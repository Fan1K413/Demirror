"""Evaluate whether P0 review candidates localize X-AIGD edge/shape annotations.

X-AIGD's ``low-level-edge_shape`` polygons are human annotations of visible
edge or shape artifacts in AI-generated images.  They are not source labels
and not every such artifact is a projective-geometry violation.  This command
therefore measures *candidate localization enrichment*, never AI-origin
accuracy, recall, or a detector score.

The P0 pipeline is run before polygon masks are constructed.  An annotation is
read only after P0 has emitted its candidates, then used to compare candidate
lines with all detected lines.  A development/holdout split in the manifest is
reported separately; no P0 setting is changed from either result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _scaled_polygons(
    polygons: list[list[list[float]]],
    original_size: tuple[int, int],
    canonical_size: tuple[int, int],
) -> list[np.ndarray]:
    source_width, source_height = original_size
    width, height = canonical_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Annotation source dimensions must be positive")
    scale = np.asarray([width / source_width, height / source_height], dtype=np.float64)
    output: list[np.ndarray] = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            continue
        points = points * scale
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        output.append(np.round(points).astype(np.int32))
    return output


def polygon_mask(polygons: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """Rasterize polygons into a binary image in (height, width) order."""

    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, color=255)
    return mask


def line_mask(lines: list[dict[str, Any]], size: tuple[int, int], *, thickness: int = 2) -> np.ndarray:
    """Rasterize serialized P0 lines into a binary image."""

    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        first = dict(line["p1_analysis"])
        second = dict(line["p2_analysis"])
        start = (
            int(np.clip(round(float(first["x"])), 0, width - 1)),
            int(np.clip(round(float(first["y"])), 0, height - 1)),
        )
        end = (
            int(np.clip(round(float(second["x"])), 0, width - 1)),
            int(np.clip(round(float(second["y"])), 0, height - 1)),
        )
        cv2.line(mask, start, end, color=255, thickness=thickness, lineType=cv2.LINE_8)
    return mask


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("annotation dilation radius must not be negative")
    if radius == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask, kernel)


def _overlap_rate(line_pixels: np.ndarray, target_mask: np.ndarray) -> float | None:
    count = int(np.count_nonzero(line_pixels))
    if count == 0:
        return None
    return float(np.count_nonzero((line_pixels > 0) & (target_mask > 0)) / count)


def _annotation_instance_hits(
    polygons: list[np.ndarray],
    candidate_pixels: np.ndarray,
    size: tuple[int, int],
    radius: int,
) -> tuple[int, int]:
    hit_count = 0
    for polygon in polygons:
        target = _dilate(polygon_mask([polygon], size), radius)
        hit_count += int(np.any((candidate_pixels > 0) & (target > 0)))
    return hit_count, len(polygons)


def _write_review_overlay(
    input_path: Path,
    polygons: list[np.ndarray],
    lines: list[dict[str, Any]],
    candidate_ids: set[str],
    size: tuple[int, int],
    output_path: Path,
) -> None:
    width, height = size
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read downloaded image: {input_path}")
    if (image.shape[1], image.shape[0]) != size:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    polygon_layer = image.copy()
    if polygons:
        cv2.fillPoly(polygon_layer, polygons, color=(0, 230, 70))
        image = cv2.addWeighted(image, 0.72, polygon_layer, 0.28, 0.0)
    for line in lines:
        first = dict(line["p1_analysis"])
        second = dict(line["p2_analysis"])
        start = (int(round(float(first["x"]))), int(round(float(first["y"]))))
        end = (int(round(float(second["x"]))), int(round(float(second["y"]))))
        color = (0, 0, 255) if str(line["line_id"]) in candidate_ids else (215, 215, 215)
        thickness = 3 if str(line["line_id"]) in candidate_ids else 1
        cv2.line(image, start, end, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write review overlay: {output_path}")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["p0_run_status"] == "ok"]
    candidate_rates = [row["candidate_overlap_rate"] for row in usable if row["candidate_overlap_rate"] is not None]
    all_line_rates = [row["all_line_overlap_rate"] for row in usable if row["all_line_overlap_rate"] is not None]
    total_instances = sum(int(row["annotation_instance_count"]) for row in usable)
    hit_instances = sum(int(row["candidate_hit_annotation_instances"]) for row in usable)
    return {
        "count": len(rows),
        "measurement_available_count": len(usable),
        "measurement_available_rate": len(usable) / len(rows) if rows else None,
        "images_with_candidate_count": sum(int(row["candidate_line_count"]) > 0 for row in usable),
        "mean_candidate_line_count": (
            float(np.mean([int(row["candidate_line_count"]) for row in usable])) if usable else None
        ),
        "mean_candidate_overlap_rate": float(np.mean(candidate_rates)) if candidate_rates else None,
        "mean_all_line_overlap_rate": float(np.mean(all_line_rates)) if all_line_rates else None,
        "candidate_overlap_enrichment": (
            float(np.mean(candidate_rates) / np.mean(all_line_rates))
            if candidate_rates and all_line_rates and float(np.mean(all_line_rates)) > 0.0
            else None
        ),
        "candidate_hit_annotation_instances": hit_instances,
        "annotation_instance_count": total_instances,
        "annotation_instance_hit_rate": hit_instances / total_instances if total_instances else None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "P0 candidate localization evaluation only; not source classification or model training.":
        raise ValueError("Manifest purpose is not the expected localization-only protocol")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest has no samples")
    config = load_config(args.config)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError("Manifest sample is not an object")
        input_path = args.manifest.parent / str(sample["relative_path"])
        artifact_dir = args.output_dir / "artifacts" / str(sample["sample_id"])
        # P0 receives only pixels and P0 configuration.  Annotation polygons
        # are not touched until after this call has returned.
        result = analyze_image(input_path, config, artifact_dir)
        canonical = result.input.canonical_size if result.input is not None else None
        if canonical is None:
            raise RuntimeError("P0 result has no canonical input size")
        lines = json.loads((artifact_dir / "lines.json").read_text(encoding="utf-8"))
        line_by_id = {str(line["line_id"]): line for line in lines}
        candidates = [
            dict(candidate)
            for candidate in result.evidence.features.get("anomalous_lines", [])
            if float(candidate["anomaly_candidate_score"]) >= args.minimum_candidate_score
        ]
        candidate_ids = {str(candidate["line_id"]) for candidate in candidates if str(candidate["line_id"]) in line_by_id}
        polygons = _scaled_polygons(
            list(sample["edge_shape_polygons"]),
            (int(sample["width"]), int(sample["height"])),
            canonical,
        )
        annotation = polygon_mask(polygons, canonical)
        buffered_annotation = _dilate(annotation, args.annotation_dilation_px)
        all_pixels = line_mask(lines, canonical)
        candidate_lines = [line_by_id[line_id] for line_id in sorted(candidate_ids)]
        candidate_pixels = line_mask(candidate_lines, canonical)
        hit_instances, instance_count = _annotation_instance_hits(
            polygons, candidate_pixels, canonical, args.annotation_dilation_px
        )
        _write_review_overlay(
            input_path,
            polygons,
            lines,
            candidate_ids,
            canonical,
            artifact_dir / "edge_shape_review_overlay.png",
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "role": sample["role"],
                "uid": sample["uid"],
                "generator": sample["generator"],
                "input_sha256": _sha256(input_path),
                "p0_run_status": result.evidence.run_status.value,
                "p0_applicability": result.evidence.applicability,
                "p0_coverage": result.evidence.coverage,
                "all_line_count": len(lines),
                "candidate_line_count": len(candidate_ids),
                "candidate_overlap_rate": _overlap_rate(candidate_pixels, buffered_annotation),
                "all_line_overlap_rate": _overlap_rate(all_pixels, buffered_annotation),
                "candidate_hit_annotation_instances": hit_instances,
                "annotation_instance_count": instance_count,
                "review_overlay": f"artifacts/{sample['sample_id']}/edge_shape_review_overlay.png",
            }
        )
        print(f"evaluated={index}/{len(samples)} sample={sample['sample_id']}", flush=True)
    by_role = {
        role: _summarize([row for row in rows if row["role"] == role])
        for role in sorted({str(row["role"]) for row in rows})
    }
    report = {
        "schema_version": "p0-x-aigd-edge-shape-localization-v1",
        "purpose": "Evaluate P0 candidate localization against edge/shape annotations; not AI-origin classification.",
        "manifest_sha256": _sha256(args.manifest),
        "p0_config_sha256": _sha256(args.config),
        "minimum_candidate_score": args.minimum_candidate_score,
        "annotation_dilation_px": args.annotation_dilation_px,
        "overall": _summarize(rows),
        "by_role": by_role,
        "limitations": [
            "X-AIGD edge/shape annotations are not necessarily projective-geometry violations.",
            "All selected images are AI-generated; this evaluation cannot estimate source classification specificity or accuracy.",
            "The annotation masks are applied only after P0 inference and never modify its candidates.",
            "Candidate overlap is a localization-enrichment measure, not precision or recall for AI detection.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
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
    parser.add_argument("--annotation-dilation-px", type=int, default=6)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_candidate_score <= 1.0:
        raise ValueError("minimum-candidate-score must be within [0, 1]")
    report = evaluate(args)
    print(json.dumps(report["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
