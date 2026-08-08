from __future__ import annotations

import json

import numpy as np
import pytest

from image_trust.camera.calibration import (
    load_camera_experiment_results,
    summarize_camera_calibration,
)
from image_trust.camera.backends import CameraBackendInput
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
from image_trust.cli import main
from image_trust.schemas import Point


class CanonicalCameraBackend:
    backend_id = "canonical_fake"

    def estimate(self, request: CameraBackendInput) -> CameraEstimate:
        canonical_width, canonical_height = request.canonical_size
        offset_x = request.crop.x if request.crop is not None else 0
        offset_y = request.crop.y if request.crop is not None else 0
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
                p1=Point(x=0.0, y=canonical_height / 2.0 - offset_y),
                p2=Point(x=float(request.image_rgb.shape[1]), y=canonical_height / 2.0 - offset_y),
            ),
            uncertainty=CameraUncertainty(overall=0.1),
            applicability=0.9,
            coverage=0.9,
            provenance=CameraBackendProvenance(
                backend_id=self.backend_id,
                inference_device="cpu",
                elapsed_ms=0.0,
            ),
        )


def test_calibration_summary_is_descriptive_and_audits_all_gate_outcomes() -> None:
    config = _config()
    first = _result("a" * 64, config)
    second = _result("b" * 64, config)

    summary = summarize_camera_calibration(
        [first, second],
        config,
        "independent-camera-cohort",
        result_filenames=["first.json", "second.json"],
    )

    assert summary.result_count == 2
    assert summary.unique_image_count == 2
    assert summary.full_image_gate.qualified_count == 2
    assert summary.crop_gate.estimate_count == 12
    assert summary.crop_gate.qualified_count == 12
    assert summary.e_cam_observation_counts == {"measured": 2}
    assert summary.e_cam_value_distribution.p90 == 0.0
    assert summary.decision_readiness.state.value == "descriptive_only"
    assert summary.decision_readiness.e_cam_decision_threshold is None


def test_calibration_summary_rejects_duplicate_inputs_and_config_mismatch() -> None:
    config = _config()
    result = _result("a" * 64, config)

    with pytest.raises(ValueError, match="Duplicate input SHA-256"):
        summarize_camera_calibration([result, result], config, "duplicate")

    other_config = config.model_copy(update={"config_version": "different-p1"})
    with pytest.raises(ValueError, match="config digest"):
        summarize_camera_calibration([result], other_config, "mismatch")


def test_calibration_summary_rejects_tampered_gate_or_backend_provenance() -> None:
    config = _config()
    result = _result("a" * 64, config)
    changed_crop = result.crops[0].model_copy(
        update={"qualified_for_e_cam": False}
    )
    tampered_gate = result.model_copy(
        update={"crops": [changed_crop, *result.crops[1:]]}
    )
    changed_provenance = result.full_image.provenance.model_copy(
        update={"backend_id": "unexpected_backend"}
    )
    tampered_backend = result.model_copy(
        update={
            "full_image": result.full_image.model_copy(
                update={"provenance": changed_provenance}
            )
        }
    )
    shifted_crop = result.crops[0].model_copy(
        update={"crop": result.crops[0].crop.model_copy(update={"x": 1})}
    )
    tampered_crop_protocol = result.model_copy(
        update={"crops": [shifted_crop, *result.crops[1:]]}
    )

    with pytest.raises(ValueError, match="crop gate result"):
        summarize_camera_calibration([tampered_gate], config, "gate-tampered")
    with pytest.raises(ValueError, match="backend/model/weights provenance"):
        summarize_camera_calibration([tampered_backend], config, "backend-tampered")
    with pytest.raises(ValueError, match="registered P1 crop protocol"):
        summarize_camera_calibration([tampered_crop_protocol], config, "crop-tampered")


def test_calibration_summary_cli_round_trips_result_json(tmp_path) -> None:
    config = _config()
    first = _result("a" * 64, config)
    second = _result("b" * 64, config)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(
        json.dumps(first.model_dump(mode="json")), encoding="utf-8"
    )
    second_path.write_text(
        json.dumps(second.model_dump(mode="json")), encoding="utf-8"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """config_version: test-p1
camera_backend:
  name: perspective_fields
  module_name: fake_backend
  inference_device: cpu
crop_protocol:
  crop_count: 6
  side_fraction_of_short_edge: 0.52
  target_overlap_fraction: 0.25
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    exit_code = main(
        [
            "camera-calibration-summary",
            str(first_path),
            str(second_path),
            "--config",
            str(config_path),
            "--cohort",
            "cli-cohort",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["cohort_name"] == "cli-cohort"
    assert summary["backend_identity"]["backend_id"] == "canonical_fake"
    assert summary["decision_readiness"]["e_cam_decision_threshold"] is None
    assert len(load_camera_experiment_results([first_path, second_path])) == 2


def _result(image_hash: str, config: P1CameraConfig):
    return run_camera_consistency_experiment(
        canonical_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        input_summary=CameraInputSummary(
            sha256=image_hash,
            original_filename=f"{image_hash[:8]}.png",
            canonical_size=(640, 480),
        ),
        config=config,
        backend=CanonicalCameraBackend(),
    )


def _config() -> P1CameraConfig:
    return P1CameraConfig.model_validate(
        {
            "config_version": "test-p1",
            "camera_backend": CameraBackendConfig(
                name="perspective_fields",
                module_name="fake_backend",
                inference_device="cpu",
            ).model_dump(),
            "crop_protocol": {
                "crop_count": 6,
                "side_fraction_of_short_edge": 0.52,
                "target_overlap_fraction": 0.25,
            },
        }
    )
