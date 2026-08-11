"""First measurement-only stage of the local geometry reconstruction.

This module deliberately has no source or AI decision.  It produces a stable,
versioned measurement record that later stages can use to fit local line
families and consistency checks.  The existing ``geometry_relationship_v2``
classifier remains unchanged while this chain is validated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from image_trust.geometry_ai.consistency_v2 import (
    fit_region_families,
    measure_consistency_checks,
    merge_multiscale_lines,
    propose_structure_regions,
)
from image_trust.geometry_ai.features import detect_lines
from image_trust.geometry_ai.measurement_overlays import write_geometry_v2_overlays
from image_trust.geometry_ai.measurement_types import (
    CanonicalBox,
    GeometryArtifactsV2,
    GeometryCheckV2,
    GeometryGateV2,
    GeometryLineV2,
    GeometryMeasurementV2Result,
    GeometryScaleV2,
)


GLOBAL_LONG_SIDE = 960
LOCAL_LONG_SIDE = 640
GRID_SIZE = 3
MIN_GLOBAL_LINES = 12
MIN_LINES_PER_REGION = 4
MIN_SUPPORTED_REGIONS = 3


def assess_geometry_measurement_v2(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    check_callback: Callable[[list[GeometryCheckV2]], None] | None = None,
    check_started_callback: Callable[[str], None] | None = None,
) -> GeometryMeasurementV2Result:
    """Extract two-scale local line evidence without producing an AI score.

    The result is ``measurable`` only when the global image has enough line
    support, that support reaches at least three spatial cells, and at least
    three local crops independently contain usable line evidence.  These are
    deliberately conservative *measurement* gates; they are not source
    decisions and do not detect all special-imaging cases.
    """

    try:
        rgb = _load_oriented_rgb(input_path)
    except (OSError, ValueError) as error:
        return GeometryMeasurementV2Result(
            status="failed",
            summary="未能读取图片，几何测量未运行。",
            limitations=[f"geometry_measurement_image_unreadable:{type(error).__name__}"],
        )

    canonical_height, canonical_width = rgb.shape[:2]
    canonical_size = (canonical_width, canonical_height)
    whole = CanonicalBox(x=0, y=0, width=canonical_width, height=canonical_height)
    try:
        global_scale = _extract_scale(rgb, "global", "global", whole, GLOBAL_LONG_SIDE)
        local_scales = [
            _extract_scale(rgb, f"tile-{row}-{column}", "local_tile", crop, LOCAL_LONG_SIDE)
            for row, column, crop in _grid_crops(canonical_width, canonical_height)
        ]
    except (cv2.error, ValueError) as error:
        return GeometryMeasurementV2Result(
            status="failed",
            summary="线段提取失败，几何测量未完成。",
            canonical_size=canonical_size,
            limitations=[f"geometry_measurement_line_extraction_failed:{type(error).__name__}"],
        )

    merged_lines = merge_multiscale_lines(global_scale, local_scales, canonical_size)
    regions = propose_structure_regions(merged_lines, canonical_size)
    occupied_cells = _occupied_global_cells(global_scale.lines, canonical_width, canonical_height)
    supported_local_regions = sum(scale.line_count >= MIN_LINES_PER_REGION for scale in local_scales)
    stable_line_count = sum(line.cross_scale_stability >= 0.65 for line in merged_lines)
    gates = [
        GeometryGateV2(
            gate_id="global_line_support",
            passed=global_scale.line_count >= MIN_GLOBAL_LINES,
            observed=float(global_scale.line_count),
            threshold=float(MIN_GLOBAL_LINES),
            description="全图至少需要足够直线支持，避免把单一边缘当作结构。",
        ),
        GeometryGateV2(
            gate_id="spatial_region_coverage",
            passed=occupied_cells >= MIN_SUPPORTED_REGIONS,
            observed=float(occupied_cells),
            threshold=float(MIN_SUPPORTED_REGIONS),
            description="直线支持须覆盖至少三个空间区域，避免只测到一个局部物体。",
        ),
        GeometryGateV2(
            gate_id="local_region_support",
            passed=supported_local_regions >= MIN_SUPPORTED_REGIONS,
            observed=float(supported_local_regions),
            threshold=float(MIN_SUPPORTED_REGIONS),
            description="至少三个局部裁切须独立提供足够线段，供后续局部结构比较。",
        ),
        GeometryGateV2(
            gate_id="cross_scale_stability",
            passed=stable_line_count >= 8,
            observed=float(stable_line_count),
            threshold=8.0,
            description="至少八条线须在全图或多个局部尺度中稳定复现。",
        ),
        GeometryGateV2(
            gate_id="structure_region_support",
            passed=len(regions) >= 1,
            observed=float(len(regions)),
            threshold=1.0,
            description="至少形成一个具有空间连续支持的局部结构区域。",
        ),
    ]
    applicability = float(
        np.mean(
            [
                min(1.0, global_scale.line_count / MIN_GLOBAL_LINES),
                min(1.0, occupied_cells / MIN_SUPPORTED_REGIONS),
                min(1.0, supported_local_regions / MIN_SUPPORTED_REGIONS),
                min(1.0, stable_line_count / 8.0),
                min(1.0, len(regions)),
            ]
        )
    )
    measurable = all(gate.passed for gate in gates)
    families = (
        fit_region_families(
            merged_lines,
            regions,
            canonical_size,
            seed=_measurement_seed(input_path),
        )
        if measurable
        else []
    )
    checks = measure_consistency_checks(
        merged_lines,
        regions,
        families,
        rgb,
        check_callback=check_callback,
        check_started_callback=check_started_callback,
    )
    limitations = [
        "geometry_measurement_v2_has_no_source_or_ai_decision",
        "geometry_measurement_v2_special_imaging_not_assessed",
    ]
    if not measurable:
        limitations.insert(0, "geometry_measurement_v2_insufficient_structural_support")
    artifacts = GeometryArtifactsV2()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_geometry_v2_overlays(rgb, merged_lines, regions, families, checks, output_dir)
        artifacts = GeometryArtifactsV2(
            result_json="geometry_measurement_v2.json",
            regions_overlay="regions_overlay.png",
            families_overlay="families_overlay.png",
            consistency_overlay="consistency_overlay.png",
            repeat_spacing_overlay="repeat_spacing_overlay.png",
        )
    result = GeometryMeasurementV2Result(
        status="measurable" if measurable else "not_applicable",
        summary=(
            "已完成双尺度线段、局部线族与结构一致性测量。"
            if measurable
            else "画面没有足够稳定的直线结构，暂不进行局部几何比较。"
        ),
        canonical_size=canonical_size,
        applicability=applicability,
        gates=gates,
        global_scale=global_scale,
        local_scales=local_scales,
        merged_lines=merged_lines,
        regions=regions,
        families=families,
        checks=checks,
        artifacts=artifacts,
        limitations=limitations,
    )
    if output_dir is not None:
        (output_dir / "geometry_measurement_v2.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
    return result


def _load_oriented_rgb(input_path: Path) -> np.ndarray:
    try:
        with Image.open(input_path) as source:
            return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError("geometry_measurement_image_unreadable") from error


def _extract_scale(
    rgb: np.ndarray,
    scale_id: str,
    scope: str,
    crop: CanonicalBox,
    long_side: int,
) -> GeometryScaleV2:
    source = rgb[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
    analysis = _resize_for_analysis(source, long_side)
    grayscale = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)
    detected = detect_lines(grayscale)
    analysis_height, analysis_width = analysis.shape[:2]
    scale_x = crop.width / analysis_width
    scale_y = crop.height / analysis_height
    diagonal = math.hypot(crop.width, crop.height)
    lines: list[GeometryLineV2] = []
    for index, (x1, y1, x2, y2) in enumerate(detected):
        mapped_x1 = crop.x + float(x1 * scale_x)
        mapped_y1 = crop.y + float(y1 * scale_y)
        mapped_x2 = crop.x + float(x2 * scale_x)
        mapped_y2 = crop.y + float(y2 * scale_y)
        length_px = math.hypot(mapped_x2 - mapped_x1, mapped_y2 - mapped_y1)
        if length_px <= 0.0:
            continue
        lines.append(
            GeometryLineV2(
                line_id=f"{scale_id}-l{index:03d}",
                x1=mapped_x1,
                y1=mapped_y1,
                x2=mapped_x2,
                y2=mapped_y2,
                length_px=length_px,
                length_normalized=length_px / max(diagonal, 1.0),
            )
        )
    return GeometryScaleV2(
        scale_id=scale_id,
        scope=scope,
        canonical_crop=crop,
        analysis_size=(analysis_width, analysis_height),
        line_count=len(lines),
        normalized_total_length=float(sum(line.length_normalized for line in lines)),
        lines=lines,
    )


def _resize_for_analysis(rgb: np.ndarray, long_side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, long_side / max(width, height))
    if scale == 1.0:
        return rgb
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _grid_crops(width: int, height: int) -> list[tuple[int, int, CanonicalBox]]:
    """Return a deterministic 3x3 grid with a small overlap around each cell."""

    boundaries_x = np.linspace(0, width, GRID_SIZE + 1, dtype=int)
    boundaries_y = np.linspace(0, height, GRID_SIZE + 1, dtype=int)
    crops: list[tuple[int, int, CanonicalBox]] = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            left, right = int(boundaries_x[column]), int(boundaries_x[column + 1])
            top, bottom = int(boundaries_y[row]), int(boundaries_y[row + 1])
            pad_x = max(1, int(round((right - left) * 0.15)))
            pad_y = max(1, int(round((bottom - top) * 0.15)))
            expanded_left = max(0, left - pad_x)
            expanded_right = min(width, right + pad_x)
            expanded_top = max(0, top - pad_y)
            expanded_bottom = min(height, bottom + pad_y)
            crops.append(
                (
                    row,
                    column,
                    CanonicalBox(
                        x=expanded_left,
                        y=expanded_top,
                        width=expanded_right - expanded_left,
                        height=expanded_bottom - expanded_top,
                    ),
                )
            )
    return crops


def _occupied_global_cells(lines: list[GeometryLineV2], width: int, height: int) -> int:
    counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for line in lines:
        midpoint_x = (line.x1 + line.x2) * 0.5
        midpoint_y = (line.y1 + line.y2) * 0.5
        column = min(GRID_SIZE - 1, max(0, int(midpoint_x / width * GRID_SIZE)))
        row = min(GRID_SIZE - 1, max(0, int(midpoint_y / height * GRID_SIZE)))
        counts[row, column] += 1
    return int(np.count_nonzero(counts >= MIN_LINES_PER_REGION))


def _measurement_seed(input_path: Path) -> int:
    """Use stable file bytes without exposing them to the measurement model."""

    import hashlib

    digest = hashlib.sha256(input_path.read_bytes()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)
