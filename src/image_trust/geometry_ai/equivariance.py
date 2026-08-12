"""Source-neutral horizontal-flip consistency measurements for geometry fields.

The functions in this module do not classify image origin.  They only compare
two predictions that should be related by a known image-plane reflection.  In
particular, they never receive a source label, generator name or calibrated
AI-score threshold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


PARAMETER_KEYS = (
    "pred_roll",
    "pred_pitch",
    "pred_general_vfov",
    "pred_rel_cx",
    "pred_rel_cy",
)


def unflip_latitude(latitude: np.ndarray) -> np.ndarray:
    """Map a latitude prediction from a flipped image to original coordinates."""

    value = np.asarray(latitude, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("latitude must be a finite HxW array")
    return np.ascontiguousarray(value[:, ::-1])


def unflip_gravity(gravity: np.ndarray) -> np.ndarray:
    """Map a 2xHxW image-plane gravity field through a horizontal reflection.

    PerspectiveFields defines channel zero as the image-plane x component and
    channel one as y.  Reversing the pixel axis is therefore not sufficient:
    the x component must also change sign.
    """

    value = np.asarray(gravity, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != 2 or not np.isfinite(value).all():
        raise ValueError("gravity must be a finite 2xHxW array")
    mapped = np.ascontiguousarray(value[:, :, ::-1])
    mapped[0] *= -1.0
    return mapped


def _percentiles(values: np.ndarray) -> dict[str, float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0 or not np.isfinite(flattened).all():
        raise ValueError("comparison values must be finite and non-empty")
    return {
        "mean": float(np.mean(flattened)),
        "p50": float(np.percentile(flattened, 50)),
        "p90": float(np.percentile(flattened, 90)),
    }


def _wrapped_abs_degrees(value: float) -> float:
    return abs((float(value) + 180.0) % 360.0 - 180.0)


def _scalar(parameters: Mapping[str, float], key: str) -> float:
    if key not in parameters:
        raise ValueError(f"missing PerspectiveFields parameter: {key}")
    value = float(parameters[key])
    if not math.isfinite(value):
        raise ValueError(f"PerspectiveFields parameter must be finite: {key}")
    return value


def compare_horizontal_flip_predictions(
    *,
    original_latitude: np.ndarray,
    original_gravity: np.ndarray,
    original_parameters: Mapping[str, float],
    flipped_latitude: np.ndarray,
    flipped_gravity: np.ndarray,
    flipped_parameters: Mapping[str, float],
) -> dict[str, float]:
    """Return fixed physical discrepancies between original and flipped runs.

    All angular outputs are degrees.  Principal-point errors use the relative
    coordinates emitted by PerspectiveFields.  Higher values mean weaker
    horizontal-flip consistency, not a higher AI probability.
    """

    latitude = np.asarray(original_latitude, dtype=np.float64)
    gravity = np.asarray(original_gravity, dtype=np.float64)
    mapped_latitude = unflip_latitude(flipped_latitude)
    mapped_gravity = unflip_gravity(flipped_gravity)
    if latitude.shape != mapped_latitude.shape:
        raise ValueError("latitude predictions must have matching shapes")
    if gravity.shape != mapped_gravity.shape:
        raise ValueError("gravity predictions must have matching shapes")
    if latitude.ndim != 2 or gravity.ndim != 3 or gravity.shape[0] != 2:
        raise ValueError("unexpected PerspectiveFields prediction shape")
    if not np.isfinite(latitude).all() or not np.isfinite(gravity).all():
        raise ValueError("PerspectiveFields predictions must be finite")

    latitude_stats = _percentiles(np.abs(latitude - mapped_latitude))
    original_norm = np.linalg.norm(gravity, axis=0)
    mapped_norm = np.linalg.norm(mapped_gravity, axis=0)
    valid = (original_norm > 1e-8) & (mapped_norm > 1e-8)
    if not bool(np.any(valid)):
        raise ValueError("gravity predictions contain no comparable vectors")
    dot = np.sum(gravity * mapped_gravity, axis=0)
    cosine = dot[valid] / (original_norm[valid] * mapped_norm[valid])
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    gravity_stats = _percentiles(angles)

    roll_error = _wrapped_abs_degrees(
        _scalar(original_parameters, "pred_roll")
        + _scalar(flipped_parameters, "pred_roll")
    )
    pitch_error = _wrapped_abs_degrees(
        _scalar(original_parameters, "pred_pitch")
        - _scalar(flipped_parameters, "pred_pitch")
    )
    vfov_error = abs(
        _scalar(original_parameters, "pred_general_vfov")
        - _scalar(flipped_parameters, "pred_general_vfov")
    )
    principal_x_error = abs(
        _scalar(original_parameters, "pred_rel_cx")
        + _scalar(flipped_parameters, "pred_rel_cx")
    )
    principal_y_error = abs(
        _scalar(original_parameters, "pred_rel_cy")
        - _scalar(flipped_parameters, "pred_rel_cy")
    )
    return {
        "latitude_abs_mean_deg": latitude_stats["mean"],
        "latitude_abs_p50_deg": latitude_stats["p50"],
        "latitude_abs_p90_deg": latitude_stats["p90"],
        "gravity_angle_mean_deg": gravity_stats["mean"],
        "gravity_angle_p50_deg": gravity_stats["p50"],
        "gravity_angle_p90_deg": gravity_stats["p90"],
        "roll_reflection_error_deg": roll_error,
        "pitch_reflection_error_deg": pitch_error,
        "vfov_reflection_error_deg": vfov_error,
        "principal_x_reflection_error": principal_x_error,
        "principal_y_reflection_error": principal_y_error,
    }
