"""Evaluate P0 anomaly candidates against exact procedural fixture labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2

from image_trust.pipeline import analyze_image
from image_trust.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path, splits: set[str]) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if row["split"] in splits]
    if not selected:
        raise ValueError("No fixture records selected")
    return selected


def _point_to_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.dist(point, start)
    scale = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator))
    return math.dist(point, (start[0] + scale * dx, start[1] + scale * dy))


def _segments_match(first: dict[str, object], target: dict[str, object]) -> bool:
    first_start = (float(dict(first["p1"])["x"]), float(dict(first["p1"])["y"]))
    first_end = (float(dict(first["p2"])["x"]), float(dict(first["p2"])["y"]))
    target_data = dict(target["segment"])
    target_start = tuple(float(value) for value in target_data["p1"])
    target_end = tuple(float(value) for value in target_data["p2"])
    tolerance = float(target["tolerance_px"])
    distances = (
        _point_to_segment_distance(first_start, target_start, target_end),
        _point_to_segment_distance(first_end, target_start, target_end),
        _point_to_segment_distance(target_start, first_start, first_end),
        _point_to_segment_distance(target_end, first_start, first_end),
    )
    return min(distances) <= tolerance


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    tp = sum(bool(row["true_positive"]) for row in rows)
    fp = sum(bool(row["false_positive"]) for row in rows)
    tn = sum(bool(row["true_negative"]) for row in rows)
    fn = sum(bool(row["false_negative"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "count": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "measurement_available_rate": sum(row["measurement_available"] for row in rows) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["development", "holdout"])
    parser.add_argument("--minimum-candidate-score", type=float, default=0.50)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_candidate_score <= 1.0:
        raise ValueError("minimum-candidate-score must be within [0, 1]")
    records = _records(args.benchmark_root / "fixture_registry.jsonl", set(args.splits))
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    evaluated: list[dict[str, object]] = []
    for index, row in enumerate(records, start=1):
        image_path = args.benchmark_root / str(row["relative_path"])
        if _sha256(image_path) != row["sha256"]:
            raise ValueError(f"Fixture hash mismatch: {image_path}")
        sample_dir = args.output_dir / "artifacts" / str(row["sample_id"])
        result = analyze_image(image_path, config, sample_dir)
        lines = json.loads((sample_dir / "lines.json").read_text(encoding="utf-8"))
        lines_by_id = {str(line["line_id"]): line for line in lines}
        candidates = [
            candidate
            for candidate in result.evidence.features.get("anomalous_lines", [])
            if float(candidate["anomaly_candidate_score"]) >= args.minimum_candidate_score
        ]
        annotation = dict(row["geometry_annotation"])
        targets = [dict(target) for target in annotation["target_segments"]]
        matched_ids = [
            str(candidate["line_id"])
            for candidate in candidates
            if str(candidate["line_id"]) in lines_by_id
            and any(_segments_match(lines_by_id[str(candidate["line_id"])], target) for target in targets)
        ]
        anomaly_present = bool(annotation["anomaly_present"])
        predicted = bool(matched_ids) if anomaly_present else bool(candidates)
        evaluated.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "fixture_family": row["fixture_family"],
                "anomaly_present": anomaly_present,
                "measurement_available": result.evidence.run_status.value == "ok",
                "candidate_count": len(candidates),
                "matched_candidate_ids": matched_ids,
                "true_positive": anomaly_present and predicted,
                "false_positive": not anomaly_present and predicted,
                "true_negative": not anomaly_present and not predicted,
                "false_negative": anomaly_present and not predicted,
            }
        )
        print(f"evaluated={index}/{len(records)} sample={row['sample_id']}", flush=True)
    by_family: dict[str, dict[str, object]] = {}
    for family in sorted({str(row["fixture_family"]) for row in evaluated}):
        by_family[family] = _summary([row for row in evaluated if row["fixture_family"] == family])
    report = {
        "schema_version": "p0-geometry-anomaly-evaluation-v1",
        "benchmark_manifest_sha256": _sha256(args.benchmark_root / "manifest.json"),
        "p0_config_sha256": _sha256(args.config),
        "selected_splits": args.splits,
        "minimum_candidate_score": args.minimum_candidate_score,
        "overall": _summary(evaluated),
        "by_fixture_family": by_family,
        "limitations": [
            "Controlled fixtures test known line-family violations, not natural-image prevalence.",
            "The cross-generator review registry requires blinded human geometry labels before it can support accuracy claims.",
            "No declared AI/camera label is used in this evaluation.",
        ],
    }
    (args.output_dir / "evaluated_samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluated),
        encoding="utf-8",
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
