"""Benchmark interpretable vanishing-point consistency on held-out generators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from image_trust.geometry_ai.projective_features import projective_consistency_features


SEED = 20260809


def metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    real = labels == 0
    gen = labels == 1
    predicted = probability >= threshold
    return {
        "count": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "threshold": threshold,
        "true_positive_rate": float(predicted[gen].mean()),
        "false_positive_rate": float(predicted[real].mean()),
        "real_mean": float(probability[real].mean()),
        "generated_mean": float(probability[gen].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-cache", type=Path, default=Path("outputs/deeplsd_geometry_v1/line_cache.npz"))
    parser.add_argument("--feature-cache", type=Path, default=Path("outputs/projective_consistency_v1/features.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/projective_consistency_v1"))
    parser.add_argument("--target-fpr", type=float, default=0.08)
    args = parser.parse_args()
    cache = np.load(args.line_cache, allow_pickle=False)
    records = [json.loads(value) for value in cache["records"].tolist()]
    if args.feature_cache.exists():
        feature_cache = np.load(args.feature_cache, allow_pickle=False)
        features = feature_cache["features"]
        names = feature_cache["names"].tolist()
    else:
        rows: list[np.ndarray] = []
        names: list[str] | None = None
        for index, (lines, count) in enumerate(zip(cache["lines"], cache["counts"])):
            values = projective_consistency_features(lines[: int(count)], (256, 256))
            names = list(values)
            rows.append(np.asarray(list(values.values()), dtype=np.float32))
            if (index + 1) % 100 == 0:
                print(f"features {index + 1}/{len(records)}", flush=True)
        features = np.stack(rows)
        assert names is not None
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.feature_cache, features=features, names=np.asarray(names))
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    splits = np.asarray([record["split"] for record in records])
    scenes = np.asarray([record["scene"] for record in records])
    train = np.flatnonzero(splits == "train")
    calibration = np.flatnonzero(splits == "calibration")
    test = np.flatnonzero(splits == "test")
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=9,
            min_samples_leaf=30,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=12,
            max_features=0.7,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=4,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_leaf=12,
            max_features=0.7,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=4,
        ),
    }
    calibration_results: dict[str, dict[str, float | int]] = {}
    fitted: dict[str, object] = {}
    thresholds: dict[str, float] = {}
    for name, model in models.items():
        model.fit(features[train], labels[train])
        probability = model.predict_proba(features[calibration])[:, 1]
        real_probability = probability[labels[calibration] == 0]
        threshold = float(np.quantile(real_probability, 1.0 - args.target_fpr, method="higher"))
        calibration_results[name] = metrics(labels[calibration], probability, threshold)
        fitted[name] = model
        thresholds[name] = threshold
    selected = max(calibration_results, key=lambda name: calibration_results[name]["roc_auc"])
    model = fitted[selected]
    probability = model.predict_proba(features[test])[:, 1]
    report = {
        "selected_by_calibration_auc": selected,
        "calibration_candidates": calibration_results,
        "held_out_test": metrics(labels[test], probability, thresholds[selected]),
        "held_out_by_scene": {
            scene: metrics(
                labels[test[scenes[test] == scene]],
                probability[scenes[test] == scene],
                thresholds[selected],
            )
            for scene in ("indoor", "outdoor")
        },
        "feature_names": names,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
