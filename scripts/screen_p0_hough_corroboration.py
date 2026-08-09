"""Screen whether an independent Hough pass can corroborate P0 LSD candidates.

This is a *localization* experiment only.  It deliberately consumes the
already-produced P0 candidate lines from the frozen paired natural-background
fixture, then asks whether each candidate is geometrically supported by a
separate Canny + probabilistic-Hough transform.  It never sees AI-source
labels.  The first six development pairs select one Hough setting; the last
six holdout pairs are reported unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class HoughSetting:
    threshold: int
    minimum_length: int
    maximum_gap: int

    @property
    def key(self) -> str:
        return f"threshold={self.threshold};minimum_length={self.minimum_length};maximum_gap={self.maximum_gap}"


def _axis_angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = abs(float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12)))
    return math.degrees(math.acos(np.clip(cosine, 0.0, 1.0)))


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    axis = end - start
    denominator = float(np.dot(axis, axis))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, axis) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * axis)))


def _line_supports(candidate: np.ndarray, hough_line: np.ndarray, *, tolerance: float = 4.0) -> bool:
    """Return true for a directionally aligned, visibly overlapping Hough line."""

    first, second = candidate.reshape(2, 2)
    hough_first, hough_second = hough_line.reshape(2, 2)
    candidate_axis = second - first
    hough_axis = hough_second - hough_first
    candidate_length = float(np.linalg.norm(candidate_axis))
    hough_length = float(np.linalg.norm(hough_axis))
    if candidate_length <= 1e-6 or hough_length <= 1e-6:
        return False
    if _axis_angle(candidate_axis, hough_axis) > 7.5:
        return False
    distance = min(
        _point_to_segment_distance(first, hough_first, hough_second),
        _point_to_segment_distance(second, hough_first, hough_second),
        _point_to_segment_distance(hough_first, first, second),
        _point_to_segment_distance(hough_second, first, second),
    )
    if distance > tolerance:
        return False
    direction = candidate_axis / candidate_length
    projection = np.asarray([np.dot(hough_first - first, direction), np.dot(hough_second - first, direction)])
    overlap = max(0.0, min(candidate_length, float(projection.max())) - max(0.0, float(projection.min())))
    return overlap >= min(candidate_length, hough_length) * 0.30


def _target_match(line: dict[str, Any], target: dict[str, Any]) -> bool:
    candidate = np.asarray(
        [
            line["p1_analysis"]["x"],
            line["p1_analysis"]["y"],
            line["p2_analysis"]["x"],
            line["p2_analysis"]["y"],
        ],
        dtype=np.float64,
    )
    target_line = np.asarray([*target["p1"], *target["p2"]], dtype=np.float64)
    return _line_supports(candidate, target_line, tolerance=float(target["tolerance_px"]))


def _hough_lines(path: Path, setting: HoughSetting) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read benchmark image: {path}")
    edges = cv2.Canny(image, 50, 150, apertureSize=3, L2gradient=True)
    raw = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=setting.threshold,
        minLineLength=setting.minimum_length,
        maxLineGap=setting.maximum_gap,
    )
    return np.empty((0, 4), dtype=np.float64) if raw is None else np.asarray(raw, dtype=np.float64).reshape(-1, 4)


def _load_candidate_lines(artifact_dir: Path, minimum_score: float) -> list[dict[str, Any]]:
    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    candidates = [
        candidate
        for candidate in result["evidence"]["features"]["anomalous_lines"]
        if float(candidate["anomaly_candidate_score"]) >= minimum_score
    ]
    lines = {
        str(line["line_id"]): line
        for line in json.loads((artifact_dir / "lines.json").read_text(encoding="utf-8"))
    }
    return [lines[str(candidate["line_id"])] for candidate in candidates if str(candidate["line_id"]) in lines]


def _corroborated(lines: list[dict[str, Any]], hough_lines: np.ndarray) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for line in lines:
        candidate = np.asarray(
            [
                line["p1_analysis"]["x"],
                line["p1_analysis"]["y"],
                line["p2_analysis"]["x"],
                line["p2_analysis"]["y"],
            ],
            dtype=np.float64,
        )
        if any(_line_supports(candidate, hough_line) for hough_line in hough_lines):
            retained.append(line)
    return retained


def _metric(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    pairs = len(rows)
    true_positive = sum(bool(row["mutated_target_detected"]) for row in rows)
    return {
        "pair_count": pairs,
        "target_localization_recall": true_positive / max(pairs, 1),
        "mutated_target_true_positive": true_positive,
        "mean_clean_candidate_count": float(np.mean([row["clean_candidate_count"] for row in rows])) if rows else 0.0,
        "clean_images_with_any_candidate": sum(int(row["clean_candidate_count"]) > 0 for row in rows),
        "clean_any_candidate_rate": (
            sum(int(row["clean_candidate_count"]) > 0 for row in rows) / max(pairs, 1)
        ),
    }


def _evaluate_setting(
    manifest: dict[str, Any],
    manifest_path: Path,
    artifacts_root: Path,
    setting: HoughSetting,
    minimum_score: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in manifest["records"]:
        sample_id = str(record["sample_id"])
        clean_path = manifest_path.parent / str(record["clean_relative_path"])
        mutated_path = manifest_path.parent / str(record["mutated_relative_path"])
        clean_candidates = _load_candidate_lines(artifacts_root / sample_id / "clean", minimum_score)
        mutated_candidates = _load_candidate_lines(artifacts_root / sample_id / "mutated", minimum_score)
        clean_supported = _corroborated(clean_candidates, _hough_lines(clean_path, setting))
        mutated_supported = _corroborated(mutated_candidates, _hough_lines(mutated_path, setting))
        rows.append(
            {
                "sample_id": sample_id,
                "role": str(record["role"]),
                "clean_candidate_count": len(clean_supported),
                "mutated_candidate_count": len(mutated_supported),
                "mutated_target_detected": any(
                    _target_match(line, dict(record["target_segment"])) for line in mutated_supported
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-candidate-score", type=float, default=0.50)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_candidate_score <= 1.0:
        raise ValueError("minimum-candidate-score must be within [0, 1]")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = "Natural-background injected projective-line localization benchmark; not AI-origin classification."
    if manifest.get("purpose") != expected:
        raise ValueError("Unexpected benchmark manifest purpose")
    settings = [
        HoughSetting(threshold=threshold, minimum_length=minimum_length, maximum_gap=gap)
        for threshold in (30, 45, 60)
        for minimum_length in (16, 24, 32)
        for gap in (4, 8, 12)
    ]
    all_results: dict[str, dict[str, Any]] = {}
    for setting in settings:
        rows = _evaluate_setting(
            manifest,
            args.manifest,
            args.artifacts_root,
            setting,
            args.minimum_candidate_score,
        )
        development = [row for row in rows if row["role"] == "development"]
        holdout = [row for row in rows if row["role"] == "holdout"]
        all_results[setting.key] = {
            "setting": setting.__dict__,
            "development": _metric(development),
            "holdout": _metric(holdout),
            "overall": _metric(rows),
            "records": rows,
        }
    selected_key = max(
        all_results,
        key=lambda key: (
            float(all_results[key]["development"]["target_localization_recall"]),
            -float(all_results[key]["development"]["mean_clean_candidate_count"]),
            -int(all_results[key]["setting"]["threshold"]),
        ),
    )
    selected = all_results[selected_key]
    report = {
        "schema_version": "p0-hough-corroboration-screen-v1",
        "purpose": "Independent-line-detector corroboration for P0 candidate localization; not AI-origin classification.",
        "protocol": {
            "candidate_source": "frozen P0 artifacts before Hough filtering",
            "selection_split": "development pairs only",
            "holdout": "holdout pairs were not used for setting selection",
            "candidate_hough_match": "axis angle <=7.5 degrees; segment distance <=4 analysis pixels; >=30% shorter-segment overlap",
        },
        "selected_setting_key": selected_key,
        "selected_setting": selected["setting"],
        "selected_development": selected["development"],
        "selected_holdout": selected["holdout"],
        "selected_overall": selected["overall"],
        "all_settings": {
            key: {name: value for name, value in value.items() if name != "records"}
            for key, value in all_results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
