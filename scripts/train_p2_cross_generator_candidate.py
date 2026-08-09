"""Fit and audit a geometry-only candidate without touching the web decision.

The inputs are P0 feature records produced by ``evaluate_p2_cross_generator``.
Training, calibration, and final testing must come from different generator
families.  This script writes a candidate artifact for audit only; it never
installs that artifact as a production origin signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from image_trust.p2.contracts import P2ModelArtifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(paths: list[Path]) -> tuple[list[dict[str, object]], dict[str, int]]:
    records: list[dict[str, object]] = []
    excluded_not_applicable = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("p0_run_status") != "ok":
                excluded_not_applicable += 1
                continue
            if not isinstance(row.get("p0_features"), dict):
                raise ValueError(f"Successful P0 record has no features: {path}")
            records.append(row)
    if not records:
        raise ValueError("At least one feature record is required")
    return records, {
        "accepted_p0_measurements": len(records),
        "excluded_p0_not_applicable": excluded_not_applicable,
    }


def _matrix(rows: list[dict[str, object]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [[float(dict(row["p0_features"])[name]) for name in feature_names] for row in rows],
            dtype=float,
        ),
        np.asarray([int(row["label"]) for row in rows], dtype=int),
    )


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(1.0 / bins, 1.0, bins)):
        mask = (probabilities >= lower) & ((probabilities < upper) if upper < 1.0 else (probabilities <= upper))
        if mask.any():
            result += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return result


def _archives(rows: list[dict[str, object]]) -> list[str]:
    return sorted({str(row["archive_name"]) for row in rows})


def train_candidate(
    train_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
) -> tuple[P2ModelArtifact, dict[str, object]]:
    feature_names = list(dict(train_rows[0]["p0_features"]).keys())
    if any(list(dict(row["p0_features"]).keys()) != feature_names for row in train_rows + calibration_rows + test_rows):
        raise ValueError("All records must use the same ordered P0 feature schema")
    x_train, y_train = _matrix(train_rows, feature_names)
    x_calibration, y_calibration = _matrix(calibration_rows, feature_names)
    x_test, y_test = _matrix(test_rows, feature_names)
    if any(len(set(labels.tolist())) != 2 for labels in (y_train, y_calibration, y_test)):
        raise ValueError("Each split must contain both camera-photo and AI-generated labels")
    if set(_archives(train_rows)) & set(_archives(calibration_rows)):
        raise ValueError("Training and calibration generator archives must not overlap")
    if (set(_archives(train_rows)) | set(_archives(calibration_rows))) & set(_archives(test_rows)):
        raise ValueError("Test generator archives must be fully held out")

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    base = LogisticRegression(C=1.0, max_iter=500, random_state=20260809)
    base.fit((x_train - mean) / scale, y_train)
    calibration_logit = base.decision_function((x_calibration - mean) / scale).reshape(-1, 1)
    platt = LogisticRegression(C=1.0, max_iter=500, random_state=20260809)
    platt.fit(calibration_logit, y_calibration)
    test_logit = base.decision_function((x_test - mean) / scale).reshape(-1, 1)
    probability = platt.predict_proba(test_logit)[:, 1]
    evaluation = {
        "held_out_test_archives": _archives(test_rows),
        "held_out_test_count": int(len(y_test)),
        "held_out_test_roc_auc": float(roc_auc_score(y_test, probability)),
        "held_out_test_brier": float(brier_score_loss(y_test, probability)),
        "held_out_test_accuracy_at_0_5": float(accuracy_score(y_test, probability >= 0.5)),
        "held_out_test_expected_calibration_error_10_bins": _ece(probability, y_test),
        "held_out_test_mean_probability_by_label": {
            str(label): float(probability[y_test == label].mean()) for label in (0, 1)
        },
    }
    artifact = P2ModelArtifact(
        model_version="p2-cross-generator-candidate-2026-08-09.1",
        feature_names=feature_names,
        standardizer_mean=mean.tolist(),
        standardizer_scale=scale.tolist(),
        base_coefficients=base.coef_[0].tolist(),
        base_intercept=float(base.intercept_[0]),
        platt_coefficient=float(platt.coef_[0][0]),
        platt_intercept=float(platt.intercept_[0]),
        target_definition=(
            "Candidate probability of the AI-generated label for the registered Projective Geometry "
            "generator-family protocol; audit only and not an origin determination."
        ),
        calibration_dataset={
            "training_archives": _archives(train_rows),
            "calibration_archives": _archives(calibration_rows),
            "class_counts": {
                "train": {str(label): int((y_train == label).sum()) for label in (0, 1)},
                "calibration": {str(label): int((y_calibration == label).sum()) for label in (0, 1)},
            },
        },
        evaluation=evaluation,
        limitations=[
            "candidate_evaluation_failed_to_show_generalization_on_the_held_out_generator",
            "candidate_not_installed_as_a_web_or_origin_signal",
            "geometry_features_can_be_affected_by_scene_content_and_image_processing",
        ],
    )
    return artifact, evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration-records", type=Path, nargs="+", required=True)
    parser.add_argument("--test-records", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_rows, train_counts = _load_records(args.train_records)
    calibration_rows, calibration_counts = _load_records(args.calibration_records)
    test_rows, test_counts = _load_records(args.test_records)
    artifact, evaluation = train_candidate(train_rows, calibration_rows, test_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_model.json").write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "p2-cross-generator-candidate-report-v1",
        "input_record_sha256": {
            "train": {str(path): _sha256(path) for path in args.train_records},
            "calibration": {str(path): _sha256(path) for path in args.calibration_records},
            "test": {str(path): _sha256(path) for path in args.test_records},
        },
        "evaluation": evaluation,
        "p0_measurement_counts": {
            "train": train_counts,
            "calibration": calibration_counts,
            "test": test_counts,
        },
        "decision": "rejected_for_production_origin_use",
        "reason": "The held-out generator ROC AUC is below the 0.5 no-skill baseline.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
