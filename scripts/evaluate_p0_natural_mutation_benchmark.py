"""Evaluate P0 localization on paired natural-background line mutations.

See ``build_p0_natural_mutation_benchmark.py``.  This evaluator reports two
separate quantities:

* target localization: whether a P0 candidate overlaps the injected segment;
* global clean-image candidate rate: a stricter indication of review noise.

Neither is an AI-origin accuracy metric.  The clean and mutated pair both come
from the same real photograph; source labels are intentionally absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_trust.pipeline import analyze_image
from image_trust.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_points(line: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    first = dict(line["p1_analysis"])
    second = dict(line["p2_analysis"])
    return (
        np.asarray([float(first["x"]), float(first["y"])], dtype=np.float64),
        np.asarray([float(second["x"]), float(second["y"])], dtype=np.float64),
    )


def _target_points(target: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    return (
        np.asarray(target["p1"], dtype=np.float64),
        np.asarray(target["p2"], dtype=np.float64),
        float(target["tolerance_px"]),
    )


def _segment_match(line: dict[str, Any], target: dict[str, Any]) -> bool:
    """Match an LSD fragment to a known target using direction and overlap."""

    first, second = _line_points(line)
    target_first, target_second, tolerance = _target_points(target)
    target_axis = target_second - target_first
    target_length = float(np.linalg.norm(target_axis))
    candidate_axis = second - first
    candidate_length = float(np.linalg.norm(candidate_axis))
    if target_length <= 1e-6 or candidate_length <= 1e-6:
        return False
    cosine = abs(float(np.dot(target_axis, candidate_axis) / (target_length * candidate_length)))
    if math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))) > 9.0:
        return False
    normal = np.asarray([-target_axis[1], target_axis[0]], dtype=np.float64) / target_length
    perpendicular = [abs(float(np.dot(point - target_first, normal))) for point in (first, second)]
    if float(np.mean(perpendicular)) > tolerance:
        return False
    projections = [float(np.dot(point - target_first, target_axis) / target_length) for point in (first, second)]
    overlap = max(0.0, min(target_length, max(projections)) - max(0.0, min(projections)))
    return overlap >= target_length * 0.30


def _overlay(
    input_path: Path,
    lines: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target: dict[str, Any],
    output_path: Path,
) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read fixture: {input_path}")
    candidate_ids = {str(candidate["line_id"]) for candidate in candidates}
    for line in lines:
        first, second = _line_points(line)
        color = (0, 0, 255) if str(line["line_id"]) in candidate_ids else (190, 190, 190)
        thickness = 2 if str(line["line_id"]) in candidate_ids else 1
        cv2.line(
            image,
            tuple(int(value) for value in np.round(first)),
            tuple(int(value) for value in np.round(second)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    target_first, target_second, _ = _target_points(target)
    cv2.line(
        image,
        tuple(int(value) for value in np.round(target_first)),
        tuple(int(value) for value in np.round(target_second)),
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write overlay: {output_path}")


def _one_run(
    path: Path,
    target: dict[str, Any],
    config: Any,
    artifact_dir: Path,
    minimum_candidate_score: float,
) -> dict[str, Any]:
    result = analyze_image(path, config, artifact_dir)
    lines = json.loads((artifact_dir / "lines.json").read_text(encoding="utf-8"))
    candidates = [
        dict(candidate)
        for candidate in result.evidence.features.get("anomalous_lines", [])
        if float(candidate["anomaly_candidate_score"]) >= minimum_candidate_score
    ]
    by_id = {str(line["line_id"]): line for line in lines}
    matched_ids = [
        str(candidate["line_id"])
        for candidate in candidates
        if str(candidate["line_id"]) in by_id and _segment_match(by_id[str(candidate["line_id"])], target)
    ]
    _overlay(path, lines, candidates, target, artifact_dir / "target_review_overlay.png")
    return {
        "input_sha256": _sha256(path),
        "run_status": result.evidence.run_status.value,
        "applicability": result.evidence.applicability,
        "coverage": result.evidence.coverage,
        "candidate_count": len(candidates),
        "matched_target_line_ids": matched_ids,
        "target_detected": bool(matched_ids),
        "review_overlay": "target_review_overlay.png",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pair_count": 0}
    true_positive = sum(bool(row["mutated"]["target_detected"]) for row in rows)
    true_negative = sum(not bool(row["clean"]["target_detected"]) for row in rows)
    return {
        "pair_count": len(rows),
        "mutated_target_true_positive": true_positive,
        "clean_target_true_negative": true_negative,
        "target_localization_recall": true_positive / len(rows),
        "clean_target_specificity": true_negative / len(rows),
        "paired_target_localization_accuracy": (true_positive + true_negative) / (2 * len(rows)),
        "clean_images_with_any_candidate": sum(int(row["clean"]["candidate_count"]) > 0 for row in rows),
        "clean_any_candidate_rate": sum(int(row["clean"]["candidate_count"]) > 0 for row in rows) / len(rows),
        "mean_clean_candidate_count": float(np.mean([row["clean"]["candidate_count"] for row in rows])),
        "mean_mutated_candidate_count": float(np.mean([row["mutated"]["candidate_count"] for row in rows])),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = "Natural-background injected projective-line localization benchmark; not AI-origin classification."
    if manifest.get("purpose") != expected:
        raise ValueError("Unexpected benchmark manifest purpose")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark manifest has no records")
    config = load_config(args.config)
    evaluated: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError("Benchmark record is not an object")
        artifact_root = args.output_dir / "artifacts" / str(record["sample_id"])
        clean = _one_run(
            args.manifest.parent / str(record["clean_relative_path"]),
            dict(record["target_segment"]),
            config,
            artifact_root / "clean",
            args.minimum_candidate_score,
        )
        mutated = _one_run(
            args.manifest.parent / str(record["mutated_relative_path"]),
            dict(record["target_segment"]),
            config,
            artifact_root / "mutated",
            args.minimum_candidate_score,
        )
        evaluated.append(
            {
                "sample_id": record["sample_id"],
                "role": record["role"],
                "target_segment": record["target_segment"],
                "clean": clean,
                "mutated": mutated,
            }
        )
        print(f"evaluated={index}/{len(records)} sample={record['sample_id']}", flush=True)
    by_role = {
        role: _summary([row for row in evaluated if row["role"] == role])
        for role in sorted({str(row["role"]) for row in evaluated})
    }
    report = {
        "schema_version": "p0-natural-line-mutation-evaluation-v1",
        "purpose": "Paired natural-background injected-line localization evaluation; not AI-origin classification.",
        "manifest_sha256": _sha256(args.manifest),
        "p0_config_sha256": _sha256(args.config),
        "minimum_candidate_score": args.minimum_candidate_score,
        "overall": _summary(evaluated),
        "by_role": by_role,
        "limitations": [
            "All files derive from real images and contain deliberate synthetic line edits, not AI-generated source labels.",
            "Target localization does not estimate natural-image geometric-error prevalence or AI-origin accuracy.",
            "The global clean candidate rate is reported separately because P0 remains a review-candidate locator.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluated), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-candidate-score", type=float, default=0.50)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_candidate_score <= 1.0:
        raise ValueError("minimum-candidate-score must be within [0, 1]")
    report = evaluate(args)
    print(json.dumps(report["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
