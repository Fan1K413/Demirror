from __future__ import annotations

import numpy as np
import pytest

from image_trust.geometry_ai.equivariance import (
    compare_horizontal_flip_predictions,
    unflip_gravity,
    unflip_latitude,
)


def _parameters(**updates: float) -> dict[str, float]:
    values = {
        "pred_roll": 12.0,
        "pred_pitch": -8.0,
        "pred_general_vfov": 55.0,
        "pred_rel_cx": 0.04,
        "pred_rel_cy": -0.02,
    }
    values.update(updates)
    return values


def test_horizontal_flip_mapping_restores_scalar_and_vector_fields() -> None:
    latitude = np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    gravity = np.asarray(
        [
            [[0.2, 0.0, -0.2], [0.4, 0.0, -0.4]],
            [[0.9, 1.0, 0.9], [0.8, 1.0, 0.8]],
        ]
    )
    flipped_latitude = latitude[:, ::-1]
    flipped_gravity = gravity[:, :, ::-1].copy()
    flipped_gravity[0] *= -1.0

    assert np.array_equal(unflip_latitude(flipped_latitude), latitude)
    assert np.allclose(unflip_gravity(flipped_gravity), gravity)

    result = compare_horizontal_flip_predictions(
        original_latitude=latitude,
        original_gravity=gravity,
        original_parameters=_parameters(),
        flipped_latitude=flipped_latitude,
        flipped_gravity=flipped_gravity,
        flipped_parameters=_parameters(
            pred_roll=-12.0,
            pred_rel_cx=-0.04,
        ),
    )
    assert all(value == pytest.approx(0.0, abs=1e-6) for value in result.values())


def test_comparison_reports_fixed_parameter_reflection_errors() -> None:
    latitude = np.zeros((2, 2), dtype=np.float64)
    gravity = np.zeros((2, 2, 2), dtype=np.float64)
    gravity[1] = 1.0

    result = compare_horizontal_flip_predictions(
        original_latitude=latitude,
        original_gravity=gravity,
        original_parameters=_parameters(),
        flipped_latitude=latitude,
        flipped_gravity=gravity,
        flipped_parameters=_parameters(
            pred_roll=-8.0,
            pred_pitch=-5.0,
            pred_general_vfov=62.0,
            pred_rel_cx=-0.01,
            pred_rel_cy=0.01,
        ),
    )

    assert result["roll_reflection_error_deg"] == pytest.approx(4.0)
    assert result["pitch_reflection_error_deg"] == pytest.approx(3.0)
    assert result["vfov_reflection_error_deg"] == pytest.approx(7.0)
    assert result["principal_x_reflection_error"] == pytest.approx(0.03)
    assert result["principal_y_reflection_error"] == pytest.approx(0.03)


def test_comparison_rejects_shape_mismatch_and_zero_gravity() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        compare_horizontal_flip_predictions(
            original_latitude=np.zeros((2, 2)),
            original_gravity=np.ones((2, 2, 2)),
            original_parameters=_parameters(),
            flipped_latitude=np.zeros((2, 3)),
            flipped_gravity=np.ones((2, 2, 2)),
            flipped_parameters=_parameters(),
        )

    with pytest.raises(ValueError, match="no comparable vectors"):
        compare_horizontal_flip_predictions(
            original_latitude=np.zeros((2, 2)),
            original_gravity=np.zeros((2, 2, 2)),
            original_parameters=_parameters(),
            flipped_latitude=np.zeros((2, 2)),
            flipped_gravity=np.zeros((2, 2, 2)),
            flipped_parameters=_parameters(),
        )
