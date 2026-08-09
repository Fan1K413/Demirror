"""Fit and serialize a small, explicitly scoped P2 geometry classifier."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from image_trust.p2.contracts import P2ModelArtifact
from image_trust.p2.inference import extract_geometry_features
from image_trust.pipeline import analyze_image
from image_trust.utils.config import load_config


def _feature_rows(registry: dict[str, object], config_path: Path, work_dir: Path) -> list[dict[str, object]]:
    dataset_root = Path(str(registry["dataset_root"]))
    config = load_config(config_path)
    rows: list[dict[str, object]] = []
    cv2.setNumThreads(1)
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Registry entries must be a list")
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError("Registry entry must be an object")
        image_path = dataset_root / str(entry["relative_path"])
        result = analyze_image(image_path, config, work_dir / str(entry["sample_id"]))
        values = extract_geometry_features(result)
        rows.append({
            "sample_id": entry["sample_id"],
            "split": entry["split"],
            "label": int(entry["label"]),
            "features": values,
            "p0_run_status": result.evidence.run_status.value,
        })
        if index % 50 == 0 or index == len(entries):
            print(f"features={index}/{len(entries)}")
    return rows


def _matrix(rows: list[dict[str, object]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[float(dict(row["features"])[name]) for name in names] for row in rows], dtype=float),
        np.asarray([int(row["label"]) for row in rows], dtype=int),
    )


def _expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(1.0 / bins, 1.0, bins)):
        mask = (probabilities >= lower) & ((probabilities < upper) if upper < 1.0 else (probabilities <= upper))
        if mask.any():
            result += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return result


def train(registry: dict[str, object], config_path: Path, work_dir: Path) -> tuple[P2ModelArtifact, list[dict[str, object]]]:
    rows = _feature_rows(registry, config_path, work_dir)
    names = list(dict(rows[0]["features"]).keys())
    subsets = {split: [row for row in rows if row["split"] == split] for split in ("train", "calibration", "test")}
    if any(len(subsets[split]) == 0 for split in subsets):
        raise ValueError("Registry must contain train, calibration, and test entries")
    x_train, y_train = _matrix(subsets["train"], names)
    x_calibration, y_calibration = _matrix(subsets["calibration"], names)
    x_test, y_test = _matrix(subsets["test"], names)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    base = LogisticRegression(C=1.0, max_iter=500, random_state=20260808)
    base.fit((x_train - mean) / scale, y_train)
    calibration_logit = base.decision_function((x_calibration - mean) / scale).reshape(-1, 1)
    platt = LogisticRegression(C=1.0, max_iter=500, random_state=20260808)
    platt.fit(calibration_logit, y_calibration)
    test_logit = base.decision_function((x_test - mean) / scale).reshape(-1, 1)
    test_probability = platt.predict_proba(test_logit)[:, 1]
    counts = Counter((str(row["split"]), str(row["label"])) for row in rows)
    artifact = P2ModelArtifact(
        model_version="p2-projective-geometry-pilot-2026-08-08.1",
        feature_names=names,
        standardizer_mean=mean.tolist(),
        standardizer_scale=scale.tolist(),
        base_coefficients=base.coef_[0].tolist(),
        base_intercept=float(base.intercept_[0]),
        platt_coefficient=float(platt.coef_[0][0]),
        platt_intercept=float(platt.intercept_[0]),
        target_definition="Estimated probability of the AI-generated label in the registered Projective Geometry Recent Deepfloyd Indoor benchmark; not a determination of origin, authenticity, or editing.",
        calibration_dataset={
            "registry_schema_version": registry.get("schema_version"),
            "source": registry.get("source"),
            "split_protocol": registry.get("split_protocol"),
            "class_counts": {f"{split}:{label}": count for (split, label), count in sorted(counts.items())},
            "p0_config": str(config_path),
        },
        evaluation={
            "held_out_test_count": int(len(y_test)),
            "held_out_test_roc_auc": float(roc_auc_score(y_test, test_probability)),
            "held_out_test_brier": float(brier_score_loss(y_test, test_probability)),
            "held_out_test_accuracy_at_0_5": float(accuracy_score(y_test, test_probability >= 0.5)),
            "held_out_test_expected_calibration_error_10_bins": _expected_calibration_error(test_probability, y_test),
        },
        limitations=[
            "p2_experimental_model_not_provenance_proof",
            "p2_projective_geometry_recent_deepfloyd_indoor_scope_only",
            "p2_geometry_features_can_be_affected_by_scene_content_and_image_processing",
            "p2_no_claim_for_unseen_generators_or_real_world_prevalence",
        ],
    )
    return artifact, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    artifact, rows = train(registry, args.config, args.work_dir)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    args.output_features.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(f"model={args.output_model} test_auc={artifact.evaluation['held_out_test_roc_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
