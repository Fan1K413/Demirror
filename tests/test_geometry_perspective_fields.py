from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from image_trust.camera.contracts import CameraExperimentResult
from image_trust.geometry_ai.measurement_types import (
    GeometryArtifactsV2,
    GeometryCheckV2,
    GeometryMeasurementV2Result,
)
from image_trust.geometry_ai.perspective_fields import (
    PERSPECTIVE_FIELDS_ENV,
    PerspectiveFieldsProcessResult,
    attach_g5_measurements,
    build_g5_check,
    run_perspective_fields_isolated,
)


def test_isolated_channel_is_disabled_without_explicit_environment_opt_in(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(PERSPECTIVE_FIELDS_ENV, raising=False)

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("subprocess must not run while the channel is disabled")

    monkeypatch.setattr(
        "image_trust.geometry_ai.perspective_fields.subprocess.run", unexpected_run
    )
    outcome = run_perspective_fields_isolated(
        tmp_path / "missing.png",
        tmp_path / "missing.yaml",
        tmp_path / "output",
    )

    assert outcome.status == "disabled"
    assert outcome.camera_result is None
    assert outcome.result_path is None


def test_isolated_channel_uses_current_python_and_limits_worker_threads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"isolated-perspective-fields-test")
    config = _write_config(tmp_path)
    output = tmp_path / "camera-output"
    expected = _camera_result(
        "perspective_fields",
        input_sha256=_sha256(image),
        filename=image.name,
        e_cam=0.24,
    )
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        output.mkdir(parents=True)
        (output / "camera_result.json").write_text(
            expected.model_dump_json(indent=2), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "image_trust.geometry_ai.perspective_fields.subprocess.run", fake_run
    )
    outcome = run_perspective_fields_isolated(
        image,
        config,
        output,
        timeout_seconds=17,
        environment={PERSPECTIVE_FIELDS_ENV: "true", "OMP_NUM_THREADS": "99"},
    )

    assert outcome.status == "completed"
    assert outcome.camera_result == expected
    assert outcome.result_path == output.resolve() / "camera_result.json"
    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == __import__("sys").executable
    assert command[1:3] == ["-m", "image_trust.geometry_ai.perspective_fields"]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 17
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert kwargs["env"]["MKL_NUM_THREADS"] == "1"
    assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert kwargs["env"]["NUMEXPR_NUM_THREADS"] == "1"


def test_isolated_channel_converts_timeout_to_nonfatal_outcome(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"timeout")
    config = _write_config(tmp_path)

    def fake_run(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, timeout=0.1)

    monkeypatch.setattr(
        "image_trust.geometry_ai.perspective_fields.subprocess.run", fake_run
    )
    outcome = run_perspective_fields_isolated(
        image,
        config,
        tmp_path / "output",
        timeout_seconds=0.1,
        environment={PERSPECTIVE_FIELDS_ENV: "1"},
    )

    assert outcome.status == "timed_out"
    assert outcome.camera_result is None
    assert outcome.limitations == ("geometry_g5_perspective_fields_worker_timed_out",)


def test_isolated_channel_rejects_stale_or_wrong_input_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"expected-image")
    config = _write_config(tmp_path)
    output = tmp_path / "output"
    stale = _camera_result(
        "perspective_fields",
        input_sha256="0" * 64,
        filename=image.name,
        e_cam=0.2,
    )

    def fake_run(command, **_kwargs):
        output.mkdir(parents=True)
        (output / "camera_result.json").write_text(
            stale.model_dump_json(), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "image_trust.geometry_ai.perspective_fields.subprocess.run", fake_run
    )
    outcome = run_perspective_fields_isolated(
        image,
        config,
        output,
        environment={PERSPECTIVE_FIELDS_ENV: "yes"},
    )

    assert outcome.status == "failed"
    assert outcome.limitations == (
        "geometry_g5_perspective_fields_result_input_mismatch",
    )


def test_g5_uses_max_source_neutral_camera_discrepancy() -> None:
    geocalib = _camera_result("geocalib", e_cam=0.18)
    perspective_fields = _camera_result("perspective_fields", e_cam=0.43)

    check = build_g5_check(geocalib, perspective_fields)

    assert check.status == "available"
    assert check.anomaly_score == 0.43
    assert check.origin_eligible is False
    assert check.measurements["measured_backend_count"] == 2
    assert check.measurements["geocalib"]["e_cam"] == 0.18
    assert check.measurements["perspective_fields"]["e_cam"] == 0.43


def test_g5_attachment_replaces_placeholder_and_records_artifact() -> None:
    measurement = GeometryMeasurementV2Result(
        status="measurable",
        summary="test",
        checks=[
            GeometryCheckV2(check_id="G1", title="G1", status="available"),
            GeometryCheckV2(check_id="G5", title="placeholder", status="not_run"),
        ],
        artifacts=GeometryArtifactsV2(result_json="geometry_measurement_v2.json"),
    )
    process = PerspectiveFieldsProcessResult(
        status="completed",
        camera_result=_camera_result("perspective_fields", e_cam=0.31),
        result_path=Path("ignored-absolute-result.json"),
    )

    updated = attach_g5_measurements(
        measurement,
        _camera_result("geocalib", e_cam=0.12),
        process,
        perspective_fields_artifact="perspective_fields/camera_result.json",
    )

    assert updated is not measurement
    assert [check.check_id for check in updated.checks] == ["G1", "G5"]
    assert updated.checks[-1].anomaly_score == 0.31
    assert (
        updated.artifacts.perspective_fields_result
        == "perspective_fields/camera_result.json"
    )
    assert measurement.artifacts.perspective_fields_result is None


def test_failed_optional_process_does_not_discard_geocalib_measurement() -> None:
    measurement = GeometryMeasurementV2Result(status="measurable", summary="test")
    process = PerspectiveFieldsProcessResult(
        status="timed_out",
        limitations=("geometry_g5_perspective_fields_worker_timed_out",),
    )

    updated = attach_g5_measurements(
        measurement,
        _camera_result("geocalib", e_cam=0.22),
        process,
    )

    g5 = updated.checks[-1]
    assert g5.status == "available"
    assert g5.anomaly_score == 0.22
    assert "geometry_g5_perspective_fields_worker_timed_out" in g5.limitations


def _write_config(root: Path) -> Path:
    config = root / "perspective-fields.yaml"
    config.write_text(
        """config_version: test-perspective-fields
camera_backend:
  name: perspective_fields
  module_name: perspective2d
  weights_path: weights/test.pth
  model_commit: abc123
  inference_device: cpu
  perspective_fields_model_version: Paramnet-360Cities-edina-uncentered
crop_protocol:
  crop_count: 6
  side_fraction_of_short_edge: 0.52
  target_overlap_fraction: 0.25
quality_gate:
  min_applicability: 0.5
  min_coverage: 0.5
  max_uncertainty: 0.5
  min_qualified_crops: 3
""",
        encoding="utf-8",
    )
    return config


def _camera_result(
    backend: str,
    *,
    input_sha256: str = "a" * 64,
    filename: str = "input.png",
    e_cam: float | None,
) -> CameraExperimentResult:
    observation = "measured" if e_cam is not None else "not_observed"
    return CameraExperimentResult.model_validate(
        {
            "run": {
                "run_id": f"test-{backend}",
                "created_at_utc": "2026-08-11T00:00:00+00:00",
                "config_version": f"test-{backend}",
                "config_digest": "b" * 64,
                "requested_backend": backend,
            },
            "input": {
                "sha256": input_sha256,
                "original_filename": filename,
                "canonical_size": [640, 480],
            },
            "full_image": {
                "status": "ok",
                "camera_model": "pinhole",
                "roll": 0.1,
                "pitch": -0.1,
                "vfov_or_focal": {
                    "kind": "vfov_deg",
                    "value": 60.0,
                    "reference": "camera",
                },
                "uncertainty": {"overall": 0.1},
                "applicability": 1.0,
                "coverage": 1.0,
                "provenance": {
                    "backend_id": backend,
                    "backend_version": "test",
                    "model_commit": "abc123",
                    "weights_sha256": "c" * 64,
                    "inference_device": "cpu",
                    "requested_inference_device": "cpu",
                    "elapsed_ms": 1.0,
                },
            },
            "crops": [],
            "e_cam": {
                "observation": observation,
                "value": e_cam,
                "qualified_crop_ids": ["c1", "c2", "c3"] if e_cam is not None else [],
                "required_qualified_crops": 3,
                "component_means": {"roll_circular_rad": e_cam}
                if e_cam is not None
                else {},
            },
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
