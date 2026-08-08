from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import time

import numpy as np
import pytest

from image_trust.camera.backends import (
    CameraBackendInput,
    _resize_geocalib_input,
    resolve_camera_backend,
)
from image_trust.camera.contracts import (
    CameraBackendConfig,
    CameraBackendProvenance,
    CameraConsistencyMeasurement,
    CameraConsistencyObservation,
    CameraCropProtocolConfig,
    CameraEstimate,
    CameraEstimateStatus,
    CameraModel,
    CameraUncertainty,
    FieldOfViewOrFocal,
    HorizonLine,
    IntrinsicKind,
)
from image_trust.camera.crops import map_estimate_to_canonical, plan_overlapping_crops
from image_trust.schemas import Point


def test_unavailable_backend_is_explicit_and_never_substitutes_another_model() -> None:
    config = CameraBackendConfig(
        name="perspective_fields",
        module_name="demirror_missing_perspective_fields",
        inference_device="cpu",
    )
    result = resolve_camera_backend(config).estimate(
        CameraBackendInput(
            image_rgb=np.zeros((12, 16, 3), dtype=np.uint8),
            canonical_size=(16, 12),
        )
    )

    assert result.status is CameraEstimateStatus.UNAVAILABLE
    assert result.provenance.backend_id == "perspective_fields"
    assert result.provenance.inference_device == "not_started"
    assert result.provenance.requested_inference_device == "cpu"
    assert "dependency_not_installed:demirror_missing_perspective_fields" in result.limitations
    assert "weights_path_not_configured" in result.limitations
    assert result.roll is None
    assert result.principal_point is None


def test_geocalib_requires_a_local_weight_with_reproducibility_metadata() -> None:
    result = resolve_camera_backend(
        CameraBackendConfig(
            name="geocalib",
            module_name="geocalib",
            inference_device="cpu",
        )
    ).estimate(
        CameraBackendInput(
            image_rgb=np.zeros((12, 16, 3), dtype=np.uint8),
            canonical_size=(16, 12),
        )
    )

    assert result.status is CameraEstimateStatus.UNAVAILABLE
    assert "weights_path_not_configured" in result.limitations
    assert "model_commit_not_recorded" in result.limitations
    assert "weights_license_not_recorded" in result.limitations
    assert "expected_weights_sha256_not_recorded" in result.limitations
    assert "p1_backend_inference_not_implemented" not in result.limitations


def test_geocalib_resizes_only_its_large_input_copy() -> None:
    image = np.zeros((2000, 4000, 3), dtype=np.uint8)

    resized, did_resize = _resize_geocalib_input(image, max_edge=1280)

    assert did_resize is True
    assert resized.shape == (640, 1280, 3)
    assert _resize_geocalib_input(resized, max_edge=1280)[1] is False


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="Tensor construction is exercised only in the optional P1 runtime.",
)
def test_perspective_fields_conversion_preserves_units_and_missing_uncertainty() -> None:
    import torch

    backend = resolve_camera_backend(
        CameraBackendConfig(
            name="perspective_fields",
            module_name="perspective2d",
            weights_path=str(Path(__file__)),
            model_commit="d54be737d6eacfb9d39a2b7079a494924b45bb6c",
            weights_license="noncommercial-test-license",
            inference_device="cpu",
        )
    )
    result = backend._to_camera_estimate(  # type: ignore[attr-defined]
        {
            "pred_roll": torch.tensor(15.0),
            "pred_pitch": torch.tensor(-10.0),
            "pred_general_vfov": torch.tensor(55.0),
            "pred_rel_cx": torch.tensor(-0.1),
            "pred_rel_cy": torch.tensor(0.2),
        },
        CameraBackendInput(
            image_rgb=np.zeros((100, 200, 3), dtype=np.uint8),
            canonical_size=(200, 100),
        ),
        time.perf_counter(),
        "cpu",
    )

    assert result.status is CameraEstimateStatus.OK
    assert result.camera_model is CameraModel.PINHOLE
    assert result.roll == pytest.approx(math.radians(15.0))
    assert result.pitch == pytest.approx(math.radians(-10.0))
    assert result.vfov_or_focal is not None
    assert result.vfov_or_focal.value == pytest.approx(55.0)
    assert result.principal_point is not None
    assert result.principal_point.x == pytest.approx(80.0)
    assert result.principal_point.y == pytest.approx(70.0)
    assert result.horizon is None
    assert result.uncertainty.overall is None
    assert "perspective_fields_native_uncertainty_not_exposed" in result.limitations


@pytest.mark.skipif(
    importlib.util.find_spec("geocalib") is None,
    reason="GeoCalib is an optional P1 dependency.",
)
def test_geocalib_result_conversion_uses_the_local_api_without_loading_weights() -> None:
    import torch
    from geocalib.camera import Pinhole
    from geocalib.gravity import Gravity

    backend = resolve_camera_backend(
        CameraBackendConfig(
            name="geocalib",
            module_name="geocalib",
            weights_path=str(Path(__file__)),
            model_commit="97b8968e7798a66bf04fcf791fb535624241bda7",
            weights_license="CC-BY-4.0",
            inference_device="cpu",
        )
    )
    camera = Pinhole.from_dict(
        {
            "width": torch.tensor([640.0]),
            "height": torch.tensor([480.0]),
            "vfov": torch.tensor([math.pi / 3.0]),
        }
    )
    gravity = Gravity.from_rp(torch.tensor([0.1]), torch.tensor([-0.2]))
    result = backend._to_camera_estimate(  # type: ignore[attr-defined]
        {
            "camera": camera,
            "gravity": gravity,
            "up_confidence": torch.full((1, 12, 16), 0.9),
            "latitude_confidence": torch.full((1, 12, 16), 0.8),
            "roll_uncertainty": torch.tensor([0.05]),
            "pitch_uncertainty": torch.tensor([0.07]),
            "focal_uncertainty": torch.tensor([12.0]),
        },
        torch,
        time.perf_counter(),
        "cpu",
        output_size=(1280, 960),
        input_was_downscaled=True,
    )

    assert result.status is CameraEstimateStatus.OK
    assert result.camera_model is CameraModel.PINHOLE
    assert result.principal_point == Point(x=640.0, y=480.0)
    assert result.horizon is not None
    assert result.horizon.p2.x == 1280.0
    assert result.vfov_or_focal is not None
    assert result.vfov_or_focal.kind is IntrinsicKind.VFOV_DEG
    assert result.applicability == pytest.approx(0.8)
    assert result.coverage == 1.0
    assert result.uncertainty.overall is not None
    assert "geocalib_principal_point_is_assumed_center_not_optimized" in result.limitations
    assert "geocalib_input_downscaled_to_max_edge:1280" in result.limitations


def test_crop_mapping_returns_principal_point_horizon_and_focal_to_canonical() -> None:
    crop = plan_overlapping_crops(
        (640, 480),
        CameraCropProtocolConfig(
            crop_count=4,
            side_fraction_of_short_edge=0.52,
            target_overlap_fraction=0.25,
        ),
    )[0]
    estimate = CameraEstimate(
        status=CameraEstimateStatus.OK,
        camera_model=CameraModel.PINHOLE,
        roll=0.1,
        pitch=0.2,
        vfov_or_focal=FieldOfViewOrFocal(kind=IntrinsicKind.FOCAL_PX, value=500.0),
        principal_point=Point(x=12.0, y=34.0),
        horizon=HorizonLine(
            p1=Point(x=0.0, y=20.0),
            p2=Point(x=100.0, y=20.0),
        ),
        uncertainty=CameraUncertainty(overall=0.1),
        applicability=0.9,
        coverage=0.8,
        provenance=_provenance(),
    )

    mapped = map_estimate_to_canonical(estimate, crop)

    assert mapped.coordinate_space.value == "canonical"
    assert mapped.principal_point == Point(x=crop.x + 12.0, y=crop.y + 34.0)
    assert mapped.horizon is not None
    assert mapped.horizon.p1 == Point(x=float(crop.x), y=crop.y + 20.0)
    assert mapped.vfov_or_focal is not None
    assert mapped.vfov_or_focal.value == 500.0


def test_e_cam_contract_rejects_self_contradictory_observation_values() -> None:
    with pytest.raises(ValueError, match="require a value"):
        CameraConsistencyMeasurement(
            observation=CameraConsistencyObservation.MEASURED,
            qualified_crop_ids=["crop_1", "crop_2", "crop_3"],
            required_qualified_crops=3,
        )
    with pytest.raises(ValueError, match="must not include a value"):
        CameraConsistencyMeasurement(
            observation=CameraConsistencyObservation.NOT_OBSERVED,
            value=0.2,
            required_qualified_crops=3,
        )


def _provenance() -> CameraBackendProvenance:
    return CameraBackendProvenance(
        backend_id="fake",
        backend_version="test",
        model_commit="abc123",
        weights_sha256="0" * 64,
        inference_device="cpu",
        elapsed_ms=1.0,
    )
