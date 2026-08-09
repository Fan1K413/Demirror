from __future__ import annotations

import numpy as np

from image_trust.geometry_ai.projective_features import projective_consistency_features


def test_projective_features_are_finite_and_fixed_length() -> None:
    lines = np.asarray(
        [
            [10, 10, 100, 50],
            [10, 30, 100, 55],
            [10, 60, 100, 60],
            [20, 10, 80, 100],
            [40, 10, 85, 100],
            [70, 10, 90, 100],
        ],
        dtype=np.float32,
    )
    first = projective_consistency_features(lines, (128, 128))
    second = projective_consistency_features(lines[::-1], (128, 128))
    assert list(first) == list(second)
    assert np.isfinite(list(first.values())).all()
    assert len(first) == 50


def test_projective_features_handle_empty_lines() -> None:
    features = projective_consistency_features(np.empty((0, 4)), (256, 256))
    assert len(features) == 50
    assert not any(features.values())
