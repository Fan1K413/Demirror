"""Geometry-only summaries for 2D line segments sampled in a MoGe point map.

This module intentionally does not import MoGe or inspect RGB values.  A
short-lived worker may supply a point map, normal map, and confidence mask;
the functions here then measure whether *already detected* image lines remain
straight and locally smooth in the predicted 3D geometry.  These measures are
research features only, not an AI-origin score.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np


MOGE_LINE_FEATURE_SCHEMA_VERSION = "moge-line-geometry-v1"
_MIN_VALID_SAMPLES = 6


def moge_line_feature_names() -> list[str]:
    """Return the stable feature schema in serialization order."""

    return [
        "moge_line_count",
        "moge_usable_line_fraction",
        "moge_valid_sample_fraction_p50",
        "moge_valid_sample_fraction_p10",
        "moge_point_line_residual_p50",
        "moge_point_line_residual_p90",
        "moge_chord_residual_p50",
        "moge_chord_residual_p90",
        "moge_curved_line_fraction",
        "moge_normal_step_change_p50",
        "moge_normal_step_change_p90",
    ]


def moge_line_geometry_features(
    lines: np.ndarray | Iterable[Iterable[float]],
    points: np.ndarray,
    normals: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    *,
    samples_per_line: int = 13,
) -> OrderedDict[str, float]:
    """Summarize 3D consistency along each detected 2D line segment.

    ``points`` and optional ``normals`` are ``(height, width, 3)`` arrays.
    The residuals are dimensionless: orthogonal error divided by the robust
    3D extent of the sampled line.  This makes them invariant to MoGe's global
    scale ambiguity.  Invalid samples are ignored; a line needs at least six
    valid samples to contribute to residual aggregates.
    """

    if samples_per_line < _MIN_VALID_SAMPLES:
        raise ValueError(f"samples_per_line must be at least {_MIN_VALID_SAMPLES}")
    point_map = _validate_vector_map(points, "points")
    height, width = point_map.shape[:2]
    normal_map = None if normals is None else _validate_vector_map(normals, "normals")
    if normal_map is not None and normal_map.shape[:2] != (height, width):
        raise ValueError("normals must have the same height and width as points")
    mask = _validate_mask(valid_mask, height, width)
    line_array = _validate_lines(lines)

    values: dict[str, list[float]] = {
        "coverage": [],
        "point_residual": [],
        "chord_residual": [],
        "normal_step_change": [],
    }
    usable_lines = 0
    curved_lines = 0
    for line in line_array:
        points_on_line, normals_on_line, coverage = _sample_line(
            line, point_map, normal_map, mask, samples_per_line
        )
        values["coverage"].append(coverage)
        if len(points_on_line) < _MIN_VALID_SAMPLES:
            continue
        usable_lines += 1
        point_residual, chord_residual = _line_residuals(points_on_line)
        values["point_residual"].append(point_residual)
        values["chord_residual"].append(chord_residual)
        curved_lines += int(chord_residual >= 0.075)
        if normals_on_line is not None and len(normals_on_line) >= 2:
            values["normal_step_change"].append(_normal_step_change(normals_on_line))

    count = len(line_array)
    return OrderedDict(
        [
            ("moge_line_count", float(count)),
            ("moge_usable_line_fraction", _ratio(usable_lines, count)),
            ("moge_valid_sample_fraction_p50", _quantile(values["coverage"], 0.50)),
            ("moge_valid_sample_fraction_p10", _quantile(values["coverage"], 0.10)),
            ("moge_point_line_residual_p50", _quantile(values["point_residual"], 0.50)),
            ("moge_point_line_residual_p90", _quantile(values["point_residual"], 0.90)),
            ("moge_chord_residual_p50", _quantile(values["chord_residual"], 0.50)),
            ("moge_chord_residual_p90", _quantile(values["chord_residual"], 0.90)),
            ("moge_curved_line_fraction", _ratio(curved_lines, usable_lines)),
            ("moge_normal_step_change_p50", _quantile(values["normal_step_change"], 0.50)),
            ("moge_normal_step_change_p90", _quantile(values["normal_step_change"], 0.90)),
        ]
    )


def _validate_vector_map(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape (height, width, 3)")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be nonempty")
    return array


def _validate_mask(
    valid_mask: np.ndarray | None, height: int, width: int
) -> np.ndarray | None:
    if valid_mask is None:
        return None
    array = np.asarray(valid_mask, dtype=bool)
    if array.shape != (height, width):
        raise ValueError("valid_mask must have shape (height, width)")
    return array


def _validate_lines(lines: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(
        list(lines) if not isinstance(lines, np.ndarray) else lines, dtype=np.float64
    )
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    array = array.reshape(-1, 4)
    return array[np.isfinite(array).all(axis=1)]


def _sample_line(
    line: np.ndarray,
    point_map: np.ndarray,
    normal_map: np.ndarray | None,
    mask: np.ndarray | None,
    samples_per_line: int,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    height, width = point_map.shape[:2]
    xs = np.rint(np.linspace(line[0], line[2], samples_per_line)).astype(np.intp)
    ys = np.rint(np.linspace(line[1], line[3], samples_per_line)).astype(np.intp)
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    sampled_points = point_map[ys, xs]
    valid = np.isfinite(sampled_points).all(axis=1)
    if mask is not None:
        valid &= mask[ys, xs]
    sampled_normals = None
    if normal_map is not None:
        sampled_normals = normal_map[ys, xs]
        valid &= np.isfinite(sampled_normals).all(axis=1)
        valid &= np.linalg.norm(sampled_normals, axis=1) > 1e-9
    coverage = float(valid.mean())
    return (
        sampled_points[valid],
        None if sampled_normals is None else sampled_normals[valid],
        coverage,
    )


def _line_residuals(points: np.ndarray) -> tuple[float, float]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    direction = right_vectors[0]
    projections = centered @ direction
    extent = max(float(np.quantile(projections, 0.95) - np.quantile(projections, 0.05)), 1e-9)
    point_error = centered - np.outer(projections, direction)
    point_residual = float(np.sqrt(np.mean(np.sum(point_error * point_error, axis=1))) / extent)

    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-9:
        chord_residual = point_residual
    else:
        chord_direction = chord / chord_length
        chord_offsets = points - points[0]
        chord_error = chord_offsets - np.outer(chord_offsets @ chord_direction, chord_direction)
        chord_residual = float(
            np.sqrt(np.mean(np.sum(chord_error * chord_error, axis=1))) / chord_length
        )
    return point_residual, chord_residual


def _normal_step_change(normals: np.ndarray) -> float:
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    cosines = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    return float(np.median(1.0 - cosines))


def _quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
