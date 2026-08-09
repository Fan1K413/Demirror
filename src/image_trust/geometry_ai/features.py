"""Deterministic line extraction and relationship feature construction.

The classifier deliberately sees no RGB values.  It receives only line
coordinates, lengths, directions, spatial support, pairwise angles, and the
distribution of intersections of the corresponding infinite lines.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


FEATURE_SCHEMA_VERSION = "geometry-line-relations-v1"
ANALYSIS_LONG_SIDE = 256
MAX_LINES = 256
PAIRWISE_LINE_CAP = 96
VP_LINE_CAP = 64


def extract_image_relationship_features(
    input_path: Path,
) -> tuple[OrderedDict[str, float], int, tuple[int, int]]:
    """Read one image with EXIF orientation and return geometry-only features."""

    lines, image_size = extract_image_lines(input_path)
    return relationship_features(lines, image_size), int(len(lines)), image_size


def extract_image_lines(input_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Read one image and return the exact fixed-resolution line representation."""

    try:
        with Image.open(input_path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(oriented)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"geometry_image_unreadable:{type(error).__name__}") from error
    height, width = rgb.shape[:2]
    scale = min(1.0, ANALYSIS_LONG_SIDE / max(width, height))
    analysis_width = max(1, int(round(width * scale)))
    analysis_height = max(1, int(round(height * scale)))
    if (analysis_width, analysis_height) != (width, height):
        rgb = cv2.resize(rgb, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lines = detect_lines(grayscale)
    return lines, (
        analysis_width,
        analysis_height,
    )


def detect_lines(grayscale: np.ndarray) -> np.ndarray:
    """Return stable ``x1,y1,x2,y2`` rows sorted by decreasing length."""

    if grayscale.ndim != 2:
        raise ValueError("geometry line detection requires a grayscale image")
    height, width = grayscale.shape
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(grayscale)
    raw = detected[0] if detected else None
    if raw is None:
        return np.empty((0, 4), dtype=np.float64)
    lines = np.asarray(raw, dtype=np.float64).reshape(-1, 4)
    delta = lines[:, 2:4] - lines[:, 0:2]
    lengths = np.linalg.norm(delta, axis=1)
    minimum = max(8.0, 0.035 * math.hypot(width, height))
    valid = np.isfinite(lines).all(axis=1) & (lengths >= minimum)
    lines = lines[valid]
    lengths = lengths[valid]
    if not len(lines):
        return np.empty((0, 4), dtype=np.float64)
    order = np.lexsort((lines[:, 3], lines[:, 2], lines[:, 1], lines[:, 0], -lengths))
    return lines[order[:MAX_LINES]]


def relationship_features(
    lines: np.ndarray | Iterable[Iterable[float]],
    image_size: tuple[int, int],
) -> OrderedDict[str, float]:
    """Describe line-set structure without reducing it to anomaly counts."""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive dimensions")
    array = np.asarray(list(lines) if not isinstance(lines, np.ndarray) else lines, dtype=np.float64)
    if array.size == 0:
        array = np.empty((0, 4), dtype=np.float64)
    array = array.reshape(-1, 4)
    array = array[np.isfinite(array).all(axis=1)]
    if len(array) > MAX_LINES:
        delta = array[:, 2:4] - array[:, 0:2]
        order = np.argsort(-np.linalg.norm(delta, axis=1), kind="stable")
        array = array[order[:MAX_LINES]]

    result: OrderedDict[str, float] = OrderedDict()
    diagonal = math.hypot(width, height)
    if not len(array):
        _empty_features(result, width, height)
        return result

    p1 = array[:, :2]
    p2 = array[:, 2:4]
    delta_px = p2 - p1
    length = np.linalg.norm(delta_px, axis=1) / diagonal
    midpoint = (p1 + p2) * 0.5 / np.asarray([width, height], dtype=np.float64)
    angle = np.mod(np.arctan2(delta_px[:, 1], delta_px[:, 0]), np.pi)
    weight = np.maximum(length, 1e-8)
    total_weight = float(weight.sum())

    result["line_count_log"] = float(math.log1p(len(array)) / math.log1p(MAX_LINES))
    result["total_length_normalized"] = float(length.sum())
    result["mean_length_normalized"] = float(length.mean())
    for name, quantile in (("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90)):
        result[f"line_length_{name}"] = float(np.quantile(length, quantile))
    result["aspect_log_abs"] = float(abs(math.log(width / height)))
    result["horizontal_line_ratio"] = float(weight[_axis_distance(angle, 0.0) <= math.radians(7.5)].sum() / total_weight)
    result["vertical_line_ratio"] = float(weight[_axis_distance(angle, math.pi / 2) <= math.radians(7.5)].sum() / total_weight)

    orientation_weight = _histogram(angle, 0.0, math.pi, 18, weight)
    orientation_count = _histogram(angle, 0.0, math.pi, 18, None)
    _append_vector(result, "orientation_weight", orientation_weight)
    _append_vector(result, "orientation_count", orientation_count)
    result["orientation_weight_entropy"] = _entropy(orientation_weight)
    result["orientation_count_entropy"] = _entropy(orientation_count)
    result["orientation_top1_weight"] = float(np.max(orientation_weight))
    result["orientation_top2_weight"] = float(np.partition(orientation_weight, -2)[-2:].sum())
    axial = np.sum(weight * np.exp(2j * angle)) / total_weight
    result["orientation_axial_concentration"] = float(abs(axial))

    length_hist = _histogram(length, 0.0, 0.50, 10, None, clip=True)
    _append_vector(result, "length_hist", length_hist)
    spatial = _grid_histogram(midpoint, weight, 4, 4)
    _append_vector(result, "spatial_weight", spatial.ravel())
    result["spatial_weight_entropy"] = _entropy(spatial.ravel())
    result["spatial_occupied_ratio"] = float(np.mean(spatial.ravel() > 0.0))

    spatial_orientation = np.zeros((3, 3, 12), dtype=np.float64)
    cell_x = np.clip((midpoint[:, 0] * 3).astype(int), 0, 2)
    cell_y = np.clip((midpoint[:, 1] * 3).astype(int), 0, 2)
    direction_bin = np.clip((angle / math.pi * 12).astype(int), 0, 11)
    np.add.at(spatial_orientation, (cell_y, cell_x, direction_bin), weight)
    spatial_orientation /= max(float(spatial_orientation.sum()), 1e-12)
    _append_vector(result, "spatial_orientation", spatial_orientation.ravel())
    for y in range(3):
        for x in range(3):
            selected = (cell_x == x) & (cell_y == y)
            prefix = f"cell_{y}_{x}"
            if not selected.any():
                result[f"{prefix}_support"] = 0.0
                result[f"{prefix}_orientation_entropy"] = 0.0
                result[f"{prefix}_orientation_top"] = 0.0
                result[f"{prefix}_axial_concentration"] = 0.0
                continue
            local_weight = weight[selected]
            local_angle = angle[selected]
            local_hist = _histogram(local_angle, 0.0, math.pi, 12, local_weight)
            result[f"{prefix}_support"] = float(local_weight.sum() / total_weight)
            result[f"{prefix}_orientation_entropy"] = _entropy(local_hist)
            result[f"{prefix}_orientation_top"] = float(local_hist.max())
            result[f"{prefix}_axial_concentration"] = float(
                abs(np.sum(local_weight * np.exp(2j * local_angle)) / local_weight.sum())
            )

    _append_pairwise_features(result, midpoint, p1 / [width, height], p2 / [width, height], length, angle)
    return result


def feature_names() -> list[str]:
    return list(relationship_features(np.empty((0, 4)), (256, 256)).keys())


def _append_pairwise_features(
    result: OrderedDict[str, float],
    midpoint: np.ndarray,
    p1_normalized: np.ndarray,
    p2_normalized: np.ndarray,
    length: np.ndarray,
    angle: np.ndarray,
) -> None:
    count = min(len(length), PAIRWISE_LINE_CAP)
    if count < 2:
        _append_vector(result, "pair_angle", np.zeros(12))
        _append_vector(result, "near_pair_angle", np.zeros(12))
        _append_vector(result, "endpoint_pair_angle", np.zeros(12))
        _append_vector(result, "pair_midpoint_distance", np.zeros(10))
        _append_vp_features(result, p1_normalized[:0], p2_normalized[:0], length[:0], angle[:0])
        return
    indices = np.triu_indices(count, 1)
    first, second = indices
    difference = np.abs(angle[first] - angle[second])
    difference = np.minimum(difference, np.pi - difference)
    pair_weight = np.maximum(length[first] * length[second], 1e-12)
    midpoint_distance = np.linalg.norm(midpoint[first] - midpoint[second], axis=1)
    endpoint_distance = np.minimum.reduce(
        [
            np.linalg.norm(p1_normalized[first] - p1_normalized[second], axis=1),
            np.linalg.norm(p1_normalized[first] - p2_normalized[second], axis=1),
            np.linalg.norm(p2_normalized[first] - p1_normalized[second], axis=1),
            np.linalg.norm(p2_normalized[first] - p2_normalized[second], axis=1),
        ]
    )
    _append_vector(result, "pair_angle", _histogram(difference, 0.0, np.pi / 2, 12, pair_weight))
    _append_vector(
        result,
        "near_pair_angle",
        _conditional_histogram(difference, pair_weight, midpoint_distance <= 0.35, 0.0, np.pi / 2, 12),
    )
    _append_vector(
        result,
        "endpoint_pair_angle",
        _conditional_histogram(difference, pair_weight, endpoint_distance <= 0.08, 0.0, np.pi / 2, 12),
    )
    _append_vector(
        result,
        "pair_midpoint_distance",
        _histogram(midpoint_distance, 0.0, math.sqrt(2.0), 10, pair_weight, clip=True),
    )
    result["near_parallel_pair_ratio"] = float(pair_weight[difference <= math.radians(5)].sum() / pair_weight.sum())
    result["near_orthogonal_pair_ratio"] = float(pair_weight[np.abs(difference - np.pi / 2) <= math.radians(5)].sum() / pair_weight.sum())
    _append_vp_features(
        result,
        p1_normalized[:VP_LINE_CAP],
        p2_normalized[:VP_LINE_CAP],
        length[:VP_LINE_CAP],
        angle[:VP_LINE_CAP],
    )


def _append_vp_features(
    result: OrderedDict[str, float],
    p1: np.ndarray,
    p2: np.ndarray,
    length: np.ndarray,
    angle: np.ndarray,
) -> None:
    if len(length) < 2:
        _append_vector(result, "intersection_field", np.zeros(36))
        for name in ("valid_ratio", "inside_ratio", "concentration", "entropy", "median_radius"):
            result[f"intersection_{name}"] = 0.0
        return
    line_h = np.cross(
        np.column_stack([p1, np.ones(len(p1))]),
        np.column_stack([p2, np.ones(len(p2))]),
    )
    first, second = np.triu_indices(len(length), 1)
    angle_difference = np.abs(angle[first] - angle[second])
    angle_difference = np.minimum(angle_difference, np.pi - angle_difference)
    eligible = angle_difference >= math.radians(7.5)
    intersections_h = np.cross(line_h[first], line_h[second])
    finite = eligible & (np.abs(intersections_h[:, 2]) > 1e-8)
    total_eligible = max(int(eligible.sum()), 1)
    result["intersection_valid_ratio"] = float(finite.sum() / total_eligible)
    if not finite.any():
        _append_vector(result, "intersection_field", np.zeros(36))
        for name in ("inside_ratio", "concentration", "entropy", "median_radius"):
            result[f"intersection_{name}"] = 0.0
        return
    intersections = intersections_h[finite, :2] / intersections_h[finite, 2:3]
    pair_weight = length[first[finite]] * length[second[finite]] * np.sin(angle_difference[finite])
    mapped = 0.5 + np.arctan((intersections - 0.5) / 1.5) / np.pi
    mapped = np.clip(mapped, 0.0, np.nextafter(1.0, 0.0))
    bins = np.zeros((6, 6), dtype=np.float64)
    x_bin = np.clip((mapped[:, 0] * 6).astype(int), 0, 5)
    y_bin = np.clip((mapped[:, 1] * 6).astype(int), 0, 5)
    np.add.at(bins, (y_bin, x_bin), pair_weight)
    bins /= max(float(bins.sum()), 1e-12)
    _append_vector(result, "intersection_field", bins.ravel())
    inside = np.all((intersections >= 0.0) & (intersections <= 1.0), axis=1)
    result["intersection_inside_ratio"] = float(pair_weight[inside].sum() / pair_weight.sum())
    result["intersection_concentration"] = float(bins.max())
    result["intersection_entropy"] = _entropy(bins.ravel())
    # Nearly parallel image lines legitimately intersect extremely far away.
    # The mapped field already preserves that direction; cap this diagnostic so
    # a projective point near infinity cannot destabilize model standardization.
    result["intersection_median_radius"] = float(
        min(10.0, np.median(np.linalg.norm(intersections - 0.5, axis=1)))
    )


def _empty_features(result: OrderedDict[str, float], width: int, height: int) -> None:
    # Build the exact schema through a harmless synthetic line, then zero every
    # value except aspect ratio.  This keeps unavailable/low-line inputs stable.
    synthetic = np.asarray([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]])
    template = relationship_features(synthetic, (width, height))
    for name in template:
        result[name] = 0.0
    result["aspect_log_abs"] = float(abs(math.log(width / height)))


def _append_vector(result: OrderedDict[str, float], prefix: str, values: np.ndarray) -> None:
    for index, value in enumerate(np.asarray(values, dtype=np.float64).ravel()):
        result[f"{prefix}_{index:03d}"] = float(value)


def _histogram(
    values: np.ndarray,
    lower: float,
    upper: float,
    bins: int,
    weights: np.ndarray | None,
    *,
    clip: bool = False,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if clip:
        data = np.clip(data, lower, np.nextafter(upper, lower))
    histogram, _ = np.histogram(data, bins=bins, range=(lower, upper), weights=weights)
    histogram = histogram.astype(np.float64)
    return histogram / max(float(histogram.sum()), 1e-12)


def _conditional_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    selected: np.ndarray,
    lower: float,
    upper: float,
    bins: int,
) -> np.ndarray:
    if not selected.any():
        return np.zeros(bins, dtype=np.float64)
    return _histogram(values[selected], lower, upper, bins, weights[selected])


def _grid_histogram(points: np.ndarray, weights: np.ndarray, rows: int, columns: int) -> np.ndarray:
    result = np.zeros((rows, columns), dtype=np.float64)
    x = np.clip((points[:, 0] * columns).astype(int), 0, columns - 1)
    y = np.clip((points[:, 1] * rows).astype(int), 0, rows - 1)
    np.add.at(result, (y, x), weights)
    return result / max(float(result.sum()), 1e-12)


def _axis_distance(angle: np.ndarray, target: float) -> np.ndarray:
    difference = np.abs(angle - target)
    return np.minimum(difference, np.pi - difference)


def _entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[values > 0.0]
    if len(values) <= 1:
        return 0.0
    return float(-np.sum(values * np.log(values)) / math.log(len(probabilities)))
