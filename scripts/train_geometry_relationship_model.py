"""Train and audit Demirror's geometry-only line relationship classifier.

The protocol holds out both generator family and prompt/image identifier:

* train: DeepFloyd + Kandinsky, numeric ids 1..350;
* calibration/model selection: PixArt, ids 351..425;
* final untouched test: SDXL, ids 426..500.

Identical real images released in several generator archives are deduplicated
and never cross a numeric-id split.  The installed artifact is a portable JSON
MLP evaluated with NumPy at runtime; scikit-learn is needed only for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from image_trust.geometry_ai.contracts import DenseLayerArtifact, GeometryRelationshipModel
from image_trust.geometry_ai.features import FEATURE_SCHEMA_VERSION, extract_image_relationship_features


SEED = 20260809


@dataclass(frozen=True)
class Sample:
    path: Path
    archive: str
    scene: str
    label: int
    identifier: int
    split: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_samples(roots: list[Path]) -> tuple[list[Sample], dict[str, object]]:
    candidates: list[Sample] = []
    for root in roots:
        for label_dir in root.rglob("*"):
            if not label_dir.is_dir() or label_dir.name not in {"real", "gen"}:
                continue
            archive = label_dir.parents[2].name
            scene = "indoor" if "Indoor" in archive else "outdoor"
            for path in sorted(label_dir.iterdir()):
                if not path.is_file() or not path.stem.isdigit():
                    continue
                identifier = int(path.stem)
                generator = archive.removeprefix("Recent_").split("_", 1)[0].lower()
                if identifier <= 350 and generator in {"deepfloyd", "kandinsky"}:
                    split = "train"
                elif 351 <= identifier <= 425 and generator == "pixart":
                    split = "calibration"
                elif 426 <= identifier <= 500 and generator == "sdxl":
                    split = "test"
                else:
                    continue
                # The same real image is byte-identical across generator archives.
                # Retain one source per scene/split while keeping every generated image.
                if label_dir.name == "real":
                    preferred = {
                        "train": "deepfloyd",
                        "calibration": "pixart",
                        "test": "sdxl",
                    }[split]
                    if generator != preferred:
                        continue
                candidates.append(
                    Sample(
                        path=path,
                        archive=archive,
                        scene=scene,
                        label=1 if label_dir.name == "gen" else 0,
                        identifier=identifier,
                        split=split,
                        sha256=_sha256(path),
                    )
                )

    # Remove the occasional duplicated generated JPEG as well.
    retained: list[Sample] = []
    seen: dict[str, set[str]] = defaultdict(set)
    duplicate_counts: Counter[str] = Counter()
    for sample in sorted(candidates, key=lambda item: (item.split, item.archive, item.label, item.identifier)):
        if sample.sha256 in seen[sample.split]:
            duplicate_counts[sample.split] += 1
            continue
        seen[sample.split].add(sample.sha256)
        retained.append(sample)
    if not retained:
        raise ValueError("No geometry training samples were discovered")
    audit = {
        "candidate_count": len(candidates),
        "retained_count": len(retained),
        "duplicate_files_removed": dict(sorted(duplicate_counts.items())),
        "counts": _sample_counts(retained),
        "roots": [str(path) for path in roots],
    }
    return retained, audit


def _sample_counts(samples: list[Sample]) -> dict[str, object]:
    counts = Counter((sample.split, sample.archive, str(sample.label)) for sample in samples)
    return {
        f"{split}:{archive}:label_{label}": count
        for (split, archive, label), count in sorted(counts.items())
    }


def extract_matrix(
    samples: list[Sample],
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, object]]]:
    cv2.setNumThreads(1)

    def extract(sample: Sample) -> tuple[Sample, dict[str, float], int]:
        features, line_count, _ = extract_image_relationship_features(sample.path)
        return sample, dict(features), line_count

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="geometry-features") as executor:
        extracted = list(executor.map(extract, samples))
    names = list(extracted[0][1].keys())
    if any(list(features.keys()) != names for _, features, _ in extracted):
        raise ValueError("Geometry feature names or ordering changed within the dataset")
    matrix = np.asarray([[features[name] for name in names] for _, features, _ in extracted], dtype=np.float64)
    labels = np.asarray([sample.label for sample, _, _ in extracted], dtype=np.int64)
    records = [
        {
            "path": str(sample.path),
            "sha256": sample.sha256,
            "archive": sample.archive,
            "scene": sample.scene,
            "label": sample.label,
            "identifier": sample.identifier,
            "split": sample.split,
            "line_count": line_count,
        }
        for sample, _, line_count in extracted
    ]
    return matrix, labels, names, records


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=2)
    return np.asarray([len(labels) / (2.0 * counts[label]) for label in labels], dtype=np.float64)


def _raw_mlp_logit(model: MLPClassifier, values: np.ndarray) -> np.ndarray:
    hidden = values
    for index, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        hidden = hidden @ weights + bias
        if index < len(model.coefs_) - 1:
            hidden = np.maximum(hidden, 0.0)
    return hidden.reshape(-1)


def _calibrate(raw: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    model.fit(raw.reshape(-1, 1), labels)
    return model


def _probability(calibrator: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _threshold(real_probabilities: np.ndarray, false_positive_rate: float) -> float:
    quantile = 1.0 - false_positive_rate
    return float(np.quantile(real_probabilities, quantile, method="higher"))


def _metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = probability >= threshold
    negative = labels == 0
    positive = labels == 1
    return {
        "count": int(len(labels)),
        "real_count": int(negative.sum()),
        "generated_count": int(positive.sum()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "brier_score": float(brier_score_loss(labels, probability)),
        "threshold": float(threshold),
        "true_positive_rate": float(prediction[positive].mean()),
        "false_positive_rate": float(prediction[negative].mean()),
        "mean_probability_real": float(probability[negative].mean()),
        "mean_probability_generated": float(probability[positive].mean()),
    }


def _bootstrap_tpr_interval(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    *,
    rounds: int = 2000,
) -> list[float]:
    generated = probability[labels == 1]
    rng = np.random.default_rng(SEED)
    values = [float((rng.choice(generated, size=len(generated), replace=True) >= threshold).mean()) for _ in range(rounds)]
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def train(args: argparse.Namespace) -> tuple[GeometryRelationshipModel, dict[str, object], list[dict[str, object]]]:
    samples, dataset_audit = discover_samples(args.dataset_roots)
    matrix, labels, names, records = extract_matrix(samples, workers=args.workers)
    split_array = np.asarray([record["split"] for record in records])
    train_mask = split_array == "train"
    calibration_mask = split_array == "calibration"
    test_mask = split_array == "test"
    if any(len(set(labels[mask].tolist())) != 2 for mask in (train_mask, calibration_mask, test_mask)):
        raise ValueError("Every protocol split must contain both real and generated samples")

    scaler = StandardScaler().fit(matrix[train_mask])
    x_train = scaler.transform(matrix[train_mask])
    x_calibration = scaler.transform(matrix[calibration_mask])
    x_test = scaler.transform(matrix[test_mask])
    y_train = labels[train_mask]
    y_calibration = labels[calibration_mask]
    y_test = labels[test_mask]

    candidates: list[tuple[str, MLPClassifier, LogisticRegression, np.ndarray, dict[str, object]]] = []
    for hidden in ((64, 32), (128, 64)):
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=SEED,
        )
        classifier.fit(x_train, y_train, sample_weight=_balanced_weights(y_train))
        calibration_raw = _raw_mlp_logit(classifier, x_calibration)
        calibrator = _calibrate(calibration_raw, y_calibration)
        calibration_probability = _probability(calibrator, calibration_raw)
        threshold = _threshold(calibration_probability[y_calibration == 0], args.target_fpr)
        metrics = _metrics(y_calibration, calibration_probability, threshold)
        candidates.append((f"mlp_{hidden[0]}_{hidden[1]}", classifier, calibrator, calibration_probability, metrics))
    selected = max(
        candidates,
        key=lambda item: (
            float(item[4]["true_positive_rate"]),
            float(item[4]["roc_auc"]),
            -float(item[4]["brier_score"]),
        ),
    )
    candidate_name, classifier, calibrator, calibration_probability, calibration_metrics = selected
    ai_threshold = _threshold(calibration_probability[y_calibration == 0], args.target_fpr)
    strong_threshold = _threshold(calibration_probability[y_calibration == 0], args.strong_fpr)
    test_probability = _probability(calibrator, _raw_mlp_logit(classifier, x_test))
    test_metrics = _metrics(y_test, test_probability, ai_threshold)
    strong_calibration_metrics = _metrics(y_calibration, calibration_probability, strong_threshold)
    strong_test_metrics = _metrics(y_test, test_probability, strong_threshold)
    test_metrics["true_positive_rate_bootstrap_95_interval"] = _bootstrap_tpr_interval(
        y_test, test_probability, ai_threshold
    )
    by_archive: dict[str, object] = {}
    test_records = [record for record in records if record["split"] == "test"]
    for archive in sorted({str(record["archive"]) for record in test_records}):
        mask = np.asarray([record["archive"] == archive for record in test_records])
        archive_labels = y_test[mask]
        archive_probability = test_probability[mask]
        if len(set(archive_labels.tolist())) == 2:
            by_archive[archive] = _metrics(archive_labels, archive_probability, ai_threshold)
        else:
            by_archive[archive] = {
                "count": int(len(archive_labels)),
                "mean_probability": float(archive_probability.mean()),
                "label": int(archive_labels[0]),
            }

    candidate_audit = {
        name: metrics for name, _, _, _, metrics in candidates
    }
    evaluation = {
        "selection_split": "PixArt ids 351-425; model selected without SDXL access",
        "selected_candidate": candidate_name,
        "candidate_calibration_metrics": candidate_audit,
        "calibration_at_target_fpr": calibration_metrics,
        "held_out_test": test_metrics,
        "calibration_at_strong_fpr": strong_calibration_metrics,
        "held_out_test_at_strong_fpr": strong_test_metrics,
        "held_out_test_by_archive": by_archive,
    }
    layers = [
        DenseLayerArtifact(
            weights=weights.astype(float).tolist(),
            bias=bias.astype(float).tolist(),
            activation="relu" if index < len(classifier.coefs_) - 1 else "identity",
        )
        for index, (weights, bias) in enumerate(zip(classifier.coefs_, classifier.intercepts_))
    ]
    model = GeometryRelationshipModel(
        model_version="geometry-relationship-2026-08-11.1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        standardizer_mean=scaler.mean_.astype(float).tolist(),
        standardizer_scale=np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_).astype(float).tolist(),
        layers=layers,
        platt_coefficient=float(calibrator.coef_[0][0]),
        platt_intercept=float(calibrator.intercept_[0]),
        ai_threshold=ai_threshold,
        strong_ai_threshold=strong_threshold,
        minimum_line_count=8,
        target_definition=(
            "Balanced-review-pool probability that the image belongs to the AI-generated class, "
            "using only detected line geometry and no RGB values."
        ),
        dataset_protocol={
            **dataset_audit,
            "train": "DeepFloyd and Kandinsky, ids 1-350",
            "calibration": "PixArt, ids 351-425",
            "held_out_test": "SDXL, ids 426-500",
            "real_image_duplicate_policy": "deduplicated by SHA-256 and numeric identifier split",
            "target_false_positive_rate": args.target_fpr,
            "strong_false_positive_rate": args.strong_fpr,
        },
        evaluation=evaluation,
        limitations=[
            "geometry_score_is_calibrated_on_a_balanced_research_pool_not_real_world_prevalence",
            "geometry_relationships_can_be_ambiguous_for_fisheye_panorama_reflections_and_curved_architecture",
            "new_generators_can_improve_projective_geometry",
            "low_geometry_support_is_reported_as_not_applicable",
        ],
    )
    report = {
        "schema_version": "geometry-relationship-training-report-v1",
        "model_version": model.model_version,
        "feature_count": len(names),
        "dataset_audit": dataset_audit,
        "evaluation": evaluation,
        "decision": "candidate_requires_acceptance_gate",
        "acceptance_gate": {
            "role": "bounded_supporting_score_not_standalone_origin_verdict",
            "held_out_test_tpr_min": 0.30,
            "held_out_test_fpr_max": 0.25,
            "strong_tier_calibration_fpr": args.strong_fpr,
        },
    }
    return model, report, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=[
            Path("data/p2_projective_geometry_v1/extracted"),
            Path("data/p3_aigc_v2/extracted"),
        ],
    )
    parser.add_argument("--output-model", type=Path, default=Path("models/geometry_relationship_v2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry_relationship_v2"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.20,
        help="Calibration false-positive rate for the limited (+10) geometry tier.",
    )
    parser.add_argument(
        "--strong-fpr",
        type=float,
        default=0.05,
        help="Calibration false-positive rate for the stronger (+25) geometry tier.",
    )
    args = parser.parse_args()
    if not 0.0 < args.strong_fpr <= args.target_fpr < 1.0:
        raise ValueError("Require 0 < --strong-fpr <= --target-fpr < 1")
    model, report, records = train(args)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps(report["evaluation"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
