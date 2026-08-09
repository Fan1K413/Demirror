from __future__ import annotations

import numpy as np

from image_trust.geometry_ai.moge_line_features import (
    moge_line_feature_names,
    moge_line_geometry_features,
)


def _point_map(curved: bool) -> np.ndarray:
    coordinates = np.linspace(-1.0, 1.0, 31)
    x, y = np.meshgrid(coordinates, coordinates)
    z = 1.5 * x * x if curved else np.zeros_like(x)
    return np.stack([x, y, z], axis=-1)


def test_straight_3d_line_has_negligible_residual() -> None:
    features = moge_line_geometry_features(
        np.asarray([[2, 15, 28, 15]], dtype=float), _point_map(curved=False)
    )
    assert list(features) == moge_line_feature_names()
    assert features["moge_usable_line_fraction"] == 1.0
    assert features["moge_point_line_residual_p90"] < 1e-9
    assert features["moge_chord_residual_p90"] < 1e-9


def test_curved_3d_trace_has_larger_chord_residual() -> None:
    straight = moge_line_geometry_features(
        np.asarray([[2, 15, 28, 15]], dtype=float), _point_map(curved=False)
    )
    curved = moge_line_geometry_features(
        np.asarray([[2, 15, 28, 15]], dtype=float), _point_map(curved=True)
    )
    assert curved["moge_chord_residual_p50"] > 0.1
    assert curved["moge_chord_residual_p50"] > straight["moge_chord_residual_p50"]
    assert curved["moge_curved_line_fraction"] == 1.0


def test_invalid_mask_is_safe_and_never_returns_nan() -> None:
    features = moge_line_geometry_features(
        np.asarray([[2, 15, 28, 15]], dtype=float),
        _point_map(curved=False),
        valid_mask=np.zeros((31, 31), dtype=bool),
    )
    assert features["moge_usable_line_fraction"] == 0.0
    assert np.isfinite(list(features.values())).all()
