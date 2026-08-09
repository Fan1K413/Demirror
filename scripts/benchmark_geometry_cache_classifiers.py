"""Screen fixed geometry-only classifiers from a cached DeepLSD feature matrix.

This is a *research gate*, not a product inference path.  It deliberately
uses generator-isolated splits encoded in the cache:

* DeepFloyd + Kandinsky (IDs 1--350) for fitting;
* PixArt (IDs 351--425) for candidate selection and calibration;
* SDXL (IDs 426--500) as a one-shot untouched final test.

No RGB pixels, metadata, P3 score, or source label-derived feature is passed
to a classifier.  The cache contains line coordinates and derived geometric
relationship features only.  A candidate must pass the held-out acceptance
gate before it can be considered for a separate product-integration proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 20260809


class _ScoreEstimator(Protocol):
    def fit(self, values: np.ndarray, labels: np.ndarray) -> "_ScoreEstimator": ...

    def decision_function(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CachedSample:
    archive: str
    generator: str
    identifier: int
    label: int
    scene: str
    split: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _samples(serialized: np.ndarray) -> list[CachedSample]:
    output: list[CachedSample] = []
    for value in serialized:
        row = json.loads(str(value))
        try:
            output.append(
                CachedSample(
                    archive=str(row["archive"]),
                    generator=str(row["generator"]),
                    identifier=int(row["identifier"]),
                    label=int(row["label"]),
                    scene=str(row["scene"]),
                    split=str(row["split"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cache record has an invalid research-protocol schema") from exc
    return output


def load_cache(cache_path: Path) -> tuple[np.ndarray, list[CachedSample]]:
    """Load the relation-only matrix and reject incompatible cache layouts."""

    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"relations", "records"}
        if not required.issubset(cache.files):
            raise ValueError(f"Cache is missing required keys: {sorted(required - set(cache.files))}")
        relations = np.asarray(cache["relations"], dtype=np.float64)
        records = _samples(cache["records"])
    if relations.ndim != 2 or len(relations) != len(records):
        raise ValueError("Relation matrix and record rows do not align")
    if not np.isfinite(relations).all():
        raise ValueError("Geometry relation matrix contains non-finite values")
    return relations, records


def protocol_indices(samples: list[CachedSample]) -> dict[str, np.ndarray]:
    """Validate and return the fixed generator-isolated split protocol."""

    values = np.asarray([sample.split for sample in samples])
    result = {name: np.flatnonzero(values == name) for name in ("train", "calibration", "test")}
    if any(not len(indices) for indices in result.values()):
        raise ValueError("Cache must contain train, calibration, and test rows")
    for name, indices in result.items():
        labels = {samples[int(index)].label for index in indices}
        if labels != {0, 1}:
            raise ValueError(f"{name} split must contain both real and generated labels")
    expected_generators = {
        "train": {"deepfloyd", "kandinsky"},
        "calibration": {"pixart"},
        "test": {"sdxl"},
    }
    for name, expected in expected_generators.items():
        observed = {samples[int(index)].generator for index in result[name]}
        if observed != expected:
            raise ValueError(f"{name} split generators are {sorted(observed)}, expected {sorted(expected)}")
    return result


def _estimators() -> dict[str, _ScoreEstimator]:
    """A small, predeclared family to avoid unbounded calibration-set tuning."""

    return {
        "linear_l2_c_0_1": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.1, max_iter=3000, random_state=SEED)),
            ]
        ),
        "linear_l2_c_1": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)),
            ]
        ),
        "rbf_svm_c_0_5": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(C=0.5, kernel="rbf", gamma="scale", random_state=SEED)),
            ]
        ),
        "extra_trees_leaf_5": ExtraTreesClassifier(
            n_estimators=400,
            max_features=0.35,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=1,
            random_state=SEED,
        ),
        "hist_gradient_leaf_7": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_leaf_nodes=7,
            min_samples_leaf=30,
            l2_regularization=1.0,
            max_iter=240,
            early_stopping=True,
            random_state=SEED,
        ),
    }


def _raw_scores(estimator: _ScoreEstimator, values: np.ndarray) -> np.ndarray:
    decision = getattr(estimator, "decision_function", None)
    if callable(decision):
        scores = np.asarray(decision(values), dtype=np.float64).reshape(-1)
    else:
        probabilities = np.asarray(getattr(estimator, "predict_proba")(values), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise RuntimeError("Candidate returned an invalid probability matrix")
        clipped = np.clip(probabilities[:, 1], 1e-6, 1.0 - 1e-6)
        scores = np.log(clipped / (1.0 - clipped))
    if not np.isfinite(scores).all():
        raise RuntimeError("Candidate returned non-finite scores")
    return scores


def _probabilities(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def threshold_at_fpr(real_probabilities: np.ndarray, target_fpr: float) -> float:
    return float(np.quantile(real_probabilities, 1.0 - target_fpr, method="higher"))


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    real = labels == 0
    generated = labels == 1
    prediction = probability >= threshold
    return {
        "count": int(len(labels)),
        "real_count": int(real.sum()),
        "generated_count": int(generated.sum()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "brier_score": float(brier_score_loss(labels, probability)),
        "threshold": float(threshold),
        "true_positive_rate": float(prediction[generated].mean()),
        "false_positive_rate": float(prediction[real].mean()),
        "mean_probability_real": float(probability[real].mean()),
        "mean_probability_generated": float(probability[generated].mean()),
    }


def _by_scene(
    samples: list[CachedSample],
    test_indices: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for scene in sorted({samples[int(index)].scene for index in test_indices}):
        mask = np.asarray([samples[int(index)].scene == scene for index in test_indices])
        output[scene] = metrics(labels[mask], probability[mask], threshold)
    return output


def evaluate(
    relations: np.ndarray,
    samples: list[CachedSample],
    *,
    target_fpr: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit the fixed suite, select on PixArt, then score SDXL exactly once."""

    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be within (0, 1)")
    indices = protocol_indices(samples)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int8)
    candidate_rows: list[dict[str, Any]] = []
    fitted: list[tuple[str, _ScoreEstimator, LogisticRegression, np.ndarray, dict[str, Any]]] = []
    for name, estimator in _estimators().items():
        estimator.fit(relations[indices["train"]], labels[indices["train"]])
        calibration_scores = _raw_scores(estimator, relations[indices["calibration"]])
        calibrator = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
        calibrator.fit(calibration_scores.reshape(-1, 1), labels[indices["calibration"]])
        calibration_probability = _probabilities(calibrator, calibration_scores)
        threshold = threshold_at_fpr(
            calibration_probability[labels[indices["calibration"]] == 0], target_fpr
        )
        calibration_metrics = metrics(
            labels[indices["calibration"]], calibration_probability, threshold
        )
        candidate_rows.append({"candidate": name, "calibration": calibration_metrics})
        fitted.append((name, estimator, calibrator, calibration_probability, calibration_metrics))

    selected = max(
        fitted,
        key=lambda item: (
            float(item[4]["roc_auc"]),
            float(item[4]["true_positive_rate"]),
            -float(item[4]["brier_score"]),
        ),
    )
    name, estimator, calibrator, calibration_probability, calibration_metrics = selected
    threshold = float(calibration_metrics["threshold"])
    test_probability = _probabilities(
        calibrator, _raw_scores(estimator, relations[indices["test"]])
    )
    held_out = metrics(labels[indices["test"]], test_probability, threshold)
    acceptance = {
        "minimum_held_out_roc_auc": 0.80,
        "maximum_held_out_false_positive_rate": target_fpr,
        "minimum_held_out_true_positive_rate": 0.70,
    }
    passed = (
        float(held_out["roc_auc"]) >= acceptance["minimum_held_out_roc_auc"]
        and float(held_out["false_positive_rate"]) <= acceptance["maximum_held_out_false_positive_rate"]
        and float(held_out["true_positive_rate"]) >= acceptance["minimum_held_out_true_positive_rate"]
    )
    prediction_rows = []
    for index, probability in zip(indices["test"], test_probability):
        sample = samples[int(index)]
        prediction_rows.append(
            {
                "sample_index": int(index),
                "archive": sample.archive,
                "scene": sample.scene,
                "identifier": sample.identifier,
                "label": sample.label,
                "probability": float(probability),
                "prediction_at_frozen_threshold": bool(probability >= threshold),
            }
        )
    report: dict[str, Any] = {
        "schema_version": "geometry-cache-classifier-screen-v1",
        "purpose": "Fixed-suite geometry-only research screen; not a provenance verdict or product model.",
        "selection_protocol": {
            "fit": "DeepFloyd + Kandinsky IDs 1-350",
            "selection_and_calibration": "PixArt IDs 351-425",
            "untouched_final_test": "SDXL IDs 426-500",
            "candidate_count": len(candidate_rows),
            "target_false_positive_rate": target_fpr,
        },
        "candidate_calibration": candidate_rows,
        "selected_candidate": name,
        "selected_calibration": calibration_metrics,
        "held_out_test": held_out,
        "held_out_test_by_scene": _by_scene(
            samples, indices["test"], labels[indices["test"]], test_probability, threshold
        ),
        "acceptance_gate": acceptance,
        "acceptance_gate_passed": passed,
        "limitations": [
            "The matrix is derived from line geometry only; it contains no RGB, metadata, watermark, C2PA, or P3 inputs.",
            "A paired benchmark does not establish real-world prevalence or provenance proof.",
            "If the acceptance gate fails, no candidate is eligible for the production AI-origin decision.",
        ],
    }
    return report, prediction_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path("outputs/deeplsd_geometry_v1/line_cache.npz")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/geometry_cache_classifier_screen_v1")
    )
    parser.add_argument("--target-fpr", type=float, default=0.08)
    args = parser.parse_args()
    relations, samples = load_cache(args.cache)
    report, predictions = evaluate(relations, samples, target_fpr=args.target_fpr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["cache_sha256"] = _sha256(args.cache)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "held_out_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8"
    )
    print(json.dumps(report["held_out_test"], sort_keys=True))
    print(f"acceptance_gate_passed={report['acceptance_gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
