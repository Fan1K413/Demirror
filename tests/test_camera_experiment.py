from __future__ import annotations

import numpy as np

from image_trust.camera.backends import CameraBackendInput, resolve_camera_backend
from image_trust.camera.contracts import (
    CameraBackendConfig,
    CameraBackendProvenance,
    CameraEstimate,
    CameraEstimateStatus,
    CameraInputSummary,
    CameraModel,
    CameraUncertainty,
    FieldOfViewOrFocal,
    HorizonLine,
    P1CameraConfig,
)
from image_trust.camera.experiment import run_camera_consistency_experiment
from image_trust.schemas import Point


class CanonicalCameraBackend:
    """Deterministic test backend that returns one consistent camera for all crops."""

    backend_id = "canonical_fake"

    def estimate(self, request: CameraBackendInput) -> CameraEstimate:
        canonical_width, canonical_height = request.canonical_size
        offset_x = request.crop.x if request.crop is not None else 0
        offset_y = request.crop.y if request.crop is not None else 0
        local_width = request.image_rgb.shape[1]
        local_horizon = canonical_height / 2.0 - offset_y
        return CameraEstimate(
            status=CameraEstimateStatus.OK,
            camera_model=CameraModel.PINHOLE,
            roll=0.1,
            pitch=-0.2,
            vfov_or_focal=FieldOfViewOrFocal(kind="vfov_deg", value=60.0),
            principal_point=Point(
                x=canonical_width / 2.0 - offset_x,
                y=canonical_height / 2.0 - offset_y,
            ),
            horizon=HorizonLine(
                p1=Point(x=0.0, y=local_horizon),
                p2=Point(x=float(local_width), y=local_horizon),
            ),
            uncertainty=CameraUncertainty(overall=0.1),
            applicability=0.9,
            coverage=0.9,
            provenance=CameraBackendProvenance(
                backend_id=self.backend_id,
                backend_version="test",
                model_commit="abc123",
                weights_sha256="0" * 64,
                inference_device="cpu",
                requested_inference_device="cpu",
                elapsed_ms=0.1,
            ),
        )


def test_e_cam_is_measured_only_after_full_and_three_qualified_crop_estimates() -> None:
    result = run_camera_consistency_experiment(
        canonical_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        input_summary=_input(),
        config=_config(),
        backend=CanonicalCameraBackend(),
    )

    assert result.e_cam.observation.value == "measured"
    assert result.e_cam.value == 0.0
    assert len(result.e_cam.qualified_crop_ids) == 6
    assert "source" not in result.model_dump(mode="json")


def test_e_cam_is_not_observed_when_requested_model_is_unavailable() -> None:
    config = _config()
    unavailable_config = config.model_copy(
        update={
            "camera_backend": CameraBackendConfig(
                name="geocalib",
                module_name="demirror_missing_geocalib",
                inference_device="cpu",
            )
        }
    )
    result = run_camera_consistency_experiment(
        canonical_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        input_summary=_input(),
        config=unavailable_config,
        backend=resolve_camera_backend(unavailable_config.camera_backend),
    )

    assert result.full_image.status is CameraEstimateStatus.UNAVAILABLE
    assert result.e_cam.observation.value == "not_observed"
    assert result.e_cam.value is None
    assert result.e_cam.qualified_crop_ids == []


def _input() -> CameraInputSummary:
    return CameraInputSummary(
        sha256="a" * 64,
        original_filename="test.png",
        canonical_size=(640, 480),
    )


def _config() -> P1CameraConfig:
    return P1CameraConfig.model_validate(
        {
            "config_version": "test-p1",
            "camera_backend": {
                "name": "perspective_fields",
                "module_name": "demirror_missing_perspective_fields",
                "inference_device": "cpu",
            },
            "crop_protocol": {
                "crop_count": 6,
                "side_fraction_of_short_edge": 0.52,
                "target_overlap_fraction": 0.25,
            },
        }
    )
