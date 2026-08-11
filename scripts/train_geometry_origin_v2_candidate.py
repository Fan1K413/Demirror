"""Train an explicitly non-deployable geometry-v2 logistic candidate.

The JSONL manifest must contain one row per registered sample with:
``measurement_path``, ``sample_id``, ``base_sample_id``, ``label``, ``split``,
``scene_slice`` and ``transformation``.  An optional ``baseline_probability``
is copied to the score file for the independent replacement gate.

This script never marks a model as deployment eligible.  Run
``audit_geometry_origin_v2_gate.py`` on the frozen score file to do that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from image_trust.geometry_ai.measurement_types import GeometryMeasurementV2Result
from image_trust.geometry_ai.origin_v2 import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    GeometryOriginV2Model,
    extract_geometry_origin_features,
)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("manifest is empty")
    required = {
        "measurement_path",
        "sample_id",
        "base_sample_id",
        "label",
        "split",
        "scene_slice",
        "transformation",
    }
    for index, row in enumerate(rows, start=1):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"manifest row {index} is missing {missing}")
        if row["label"] not in (0, 1):
            raise ValueError(f"manifest row {index} label must be 0 or 1")
    return rows


def _measurement_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def _probability(matrix: np.ndarray, coefficients: np.ndarray, intercept: float) -> np.ndarray:
    logits = matrix @ coefficients + intercept
    positive = logits >= 0.0
    result = np.empty_like(logits)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = probabilities >= threshold
    positive = labels == 1
    negative = labels == 0
    return {
        "count": int(len(labels)),
        "ai_count": int(positive.sum()),
        "real_count": int(negative.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "true_positive_rate": float(prediction[positive].mean()),
        "false_positive_rate": float(prediction[negative].mean()),
        "threshold": float(threshold),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", default="geometry-origin-v2-candidate")
    parser.add_argument("--target-calibration-fpr", type=float, default=0.20)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 < args.target_calibration_fpr < 1.0:
        raise ValueError("--target-calibration-fpr must be between zero and one")

    rows = _read_manifest(args.manifest)
    vectors: list[list[float]] = []
    labels: list[int] = []
    for index, row in enumerate(rows, start=1):
        path = _measurement_path(args.manifest, str(row["measurement_path"]))
        measurement = GeometryMeasurementV2Result.model_validate_json(path.read_text(encoding="utf-8"))
        if measurement.status != "measurable":
            raise ValueError(
                f"registered sample {row['sample_id']} is not measurable; do not silently drop it"
            )
        features = extract_geometry_origin_features(measurement)
        vectors.append([features[name] for name in FEATURE_NAMES])
        labels.append(int(row["label"]))
        print(f"geometry_origin_features {index}/{len(rows)}", flush=True)

    matrix = np.asarray(vectors, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    splits = np.asarray([str(row["split"]) for row in rows])
    train = splits == "train"
    calibration = splits == "calibration"
    if set(target[train].tolist()) != {0, 1}:
        raise ValueError("train split must contain both real and AI samples")
    if set(target[calibration].tolist()) != {0, 1}:
        raise ValueError("calibration split must contain both real and AI samples")

    scaler = StandardScaler().fit(matrix[train])
    standardized = scaler.transform(matrix)
    classifier = LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=2000,
        random_state=20260811,
    )
    classifier.fit(standardized[train], target[train])
    coefficients = classifier.coef_.reshape(-1)
    intercept = float(classifier.intercept_[0])
    probabilities = _probability(standardized, coefficients, intercept)
    calibration_real = probabilities[calibration & (target == 0)]
    threshold = float(
        np.quantile(calibration_real, 1.0 - args.target_calibration_fpr, method="higher")
    )

    evaluation = {
        split: _metrics(target[splits == split], probabilities[splits == split], threshold)
        for split in sorted(set(splits.tolist()))
        if set(target[splits == split].tolist()) == {0, 1}
    }
    model = GeometryOriginV2Model(
        model_version=args.model_version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=list(FEATURE_NAMES),
        standardizer_mean=scaler.mean_.astype(float).tolist(),
        standardizer_scale=scaler.scale_.astype(float).tolist(),
        coefficients=coefficients.astype(float).tolist(),
        intercept=intercept,
        decision_threshold=threshold,
        deployment_eligible=False,
        replacement_gate={
            "eligible": False,
            "status": "not_audited",
            "reason": "candidate training never grants deployment eligibility",
        },
        training_protocol={
            "manifest": str(args.manifest),
            "fit_split": "train",
            "threshold_split": "calibration",
            "target_calibration_fpr": args.target_calibration_fpr,
            "classifier": "standardized logistic regression",
        },
        evaluation=evaluation,
        limitations=[
            "Offline candidate only until the independent replacement gate passes.",
            "Geometry measurements are associative evidence, not source provenance.",
        ],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "geometry_origin_v2_candidate.json"
    score_path = args.output_dir / "geometry_origin_v2_scores.jsonl"
    model_path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with score_path.open("w", encoding="utf-8") as handle:
        for row, probability in zip(rows, probabilities):
            score_row = {
                key: row.get(key)
                for key in (
                    "sample_id",
                    "base_sample_id",
                    "label",
                    "split",
                    "scene_slice",
                    "transformation",
                    "baseline_probability",
                )
            }
            score_row["candidate_probability"] = float(probability)
            handle.write(json.dumps(score_row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"candidate_model": str(model_path), "scores": str(score_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
