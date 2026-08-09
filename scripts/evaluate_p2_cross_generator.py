"""Evaluate the frozen P2 geometry model on registry-held-out generator groups.

This command intentionally never refits the model.  It extracts fresh P0
measurements for entries selected from the versioned P3 registry and scores
them using the installed P2 artifact.  The result is therefore an independent
generalization check, rather than another in-sample metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from image_trust.p2.inference import extract_geometry_features, infer_experimental_ai_confidence
from image_trust.pipeline import analyze_image
from image_trust.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(1.0 / bins, 1.0, bins)):
        mask = (probabilities >= lower) & ((probabilities < upper) if upper < 1.0 else (probabilities <= upper))
        if mask.any():
            result += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return result


def _bootstrap_auc_interval(labels: np.ndarray, probabilities: np.ndarray, *, seed: int = 20260809) -> list[float]:
    """Return a deterministic 95% bootstrap interval for a balanced test set."""

    rng = np.random.default_rng(seed)
    indices_by_label = [np.flatnonzero(labels == label) for label in (0, 1)]
    values: list[float] = []
    for _ in range(2000):
        sample = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in indices_by_label])
        values.append(float(roc_auc_score(labels[sample], probabilities[sample])))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    probabilities = np.asarray([float(row["p2_probability"]) for row in rows], dtype=float)
    metrics: dict[str, object] = {
        "count": int(len(rows)),
        "class_counts": {str(label): int((labels == label).sum()) for label in (0, 1)},
        "mean_probability_by_label": {str(label): float(probabilities[labels == label].mean()) for label in (0, 1)},
    }
    if len(set(labels.tolist())) == 2:
        metrics.update({
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "roc_auc_bootstrap_95_interval": _bootstrap_auc_interval(labels, probabilities),
            "brier_score": float(brier_score_loss(labels, probabilities)),
            "accuracy_at_0_5": float(accuracy_score(labels, probabilities >= 0.5)),
            "expected_calibration_error_10_bins": _ece(probabilities, labels),
        })
    return metrics


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Registry entries must be a list")
    selected = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("split") in args.splits
        and (not args.archives or entry.get("archive_name") in args.archives)
    ]
    if not selected:
        raise ValueError(f"No entries found for splits: {args.splits}")

    config = load_config(args.config)
    transient_dir = args.output_dir / "_transient_p0_artifacts"
    records: list[dict[str, object]] = []
    cv2.setNumThreads(1)
    try:
        for index, entry in enumerate(selected, start=1):
            image_path = Path(str(entry["relative_path"]))
            expected_sha = str(entry["sha256"])
            observed_sha = _sha256(image_path)
            if observed_sha != expected_sha:
                raise ValueError(f"SHA-256 mismatch for {image_path}")
            sample_dir = transient_dir / str(entry["sample_id"])
            p0_result = analyze_image(image_path, config, sample_dir)
            p2_result = infer_experimental_ai_confidence(p0_result, args.model)
            if p2_result.status != "available" or p2_result.calibrated_probability is None:
                raise RuntimeError("The requested P2 model artifact is unavailable")
            records.append({
                "sample_id": entry["sample_id"],
                "split": entry["split"],
                "archive_name": entry["archive_name"],
                "generator_family": entry["generator_family"],
                "scene_category": entry["scene_category"],
                "label": int(entry["label"]),
                "source_sha256": observed_sha,
                "p0_run_status": p0_result.evidence.run_status.value,
                "p0_applicability": p0_result.evidence.applicability,
                "p0_features": extract_geometry_features(p0_result),
                "p2_probability": p2_result.calibrated_probability,
            })
            shutil.rmtree(sample_dir, ignore_errors=True)
            print(f"scored={index}/{len(selected)} sample={entry['sample_id']}", flush=True)
    finally:
        if not args.keep_artifacts:
            shutil.rmtree(transient_dir, ignore_errors=True)

    by_group: dict[str, dict[str, object]] = {}
    for group in sorted({str(row["archive_name"]) for row in records}):
        by_group[group] = _metrics([row for row in records if row["archive_name"] == group])
    return {
        "schema_version": "p2-cross-generator-evaluation-v1",
        "purpose": "Frozen-model external generalization evaluation; no fitting occurs in this command.",
        "registry": str(args.registry),
        "registry_sha256": _sha256(args.registry),
        "selected_splits": list(args.splits),
        "selected_archives": sorted({str(row["archive_name"]) for row in records}),
        "p0_config": str(args.config),
        "p0_config_sha256": _sha256(args.config),
        "model": str(args.model),
        "model_sha256": _sha256(args.model),
        "overall": _metrics(records),
        "by_archive": by_group,
        "records_file": "records.jsonl",
        "limitations": [
            "This evaluates a frozen geometry-only classifier, not provenance proof.",
            "The test set contains paired benchmark images and does not estimate real-world prevalence.",
            "Camera-parameter measurements are intentionally not included until they have their own source-direction validation.",
        ],
    }, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/p2_projective_geometry_pilot_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["calibration", "external_test"])
    parser.add_argument("--archives", nargs="+", default=[])
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, records = evaluate(args)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
