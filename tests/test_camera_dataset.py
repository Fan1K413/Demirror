from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_trust.camera.backends import CameraBackendInput
from image_trust.camera.contracts import (
    CameraBackendConfig,
    CameraBackendProvenance,
    CameraEstimate,
    CameraEstimateStatus,
    CameraModel,
    CameraUncertainty,
    FieldOfViewOrFocal,
    HorizonLine,
    P1CameraConfig,
)
from image_trust.camera.dataset import (
    CalibrationDatasetPurpose,
    CalibrationDatasetSplit,
    CalibrationDatasetRegistry,
    CalibrationRegistryEntry,
    audit_calibration_registry,
    run_camera_calibration_dataset,
    write_calibration_registry,
)
from image_trust.cli import main
from image_trust.schemas import Point


class CanonicalCameraBackend:
    def estimate(self, request: CameraBackendInput) -> CameraEstimate:
        width, height = request.canonical_size
        offset_x = request.crop.x if request.crop is not None else 0
        offset_y = request.crop.y if request.crop is not None else 0
        return CameraEstimate(
            status=CameraEstimateStatus.OK,
            camera_model=CameraModel.PINHOLE,
            roll=0.1,
            pitch=-0.2,
            vfov_or_focal=FieldOfViewOrFocal(kind="vfov_deg", value=60.0),
            principal_point=Point(x=width / 2 - offset_x, y=height / 2 - offset_y),
            horizon=HorizonLine(
                p1=Point(x=0.0, y=height / 2 - offset_y),
                p2=Point(x=float(request.image_rgb.shape[1]), y=height / 2 - offset_y),
            ),
            uncertainty=CameraUncertainty(overall=0.1),
            applicability=0.9,
            coverage=0.9,
            provenance=CameraBackendProvenance(
                backend_id="canonical_dataset_fake",
                backend_version="test",
                model_commit="test-commit",
                weights_sha256="0" * 64,
                inference_device="cpu",
                requested_inference_device="cpu",
                elapsed_ms=0.0,
            ),
        )


def test_registry_audit_checks_files_hashes_dimensions_and_family_splits(tmp_path) -> None:
    root = tmp_path / "dataset"
    first = _image(root, "first.jpg", (120, 90), (10, 20, 30))
    second = _image(root, "second.jpg", (120, 90), (40, 50, 60))
    registry = _registry(
        _entry("first", first, root, CalibrationDatasetSplit.CALIBRATION, "camera-a"),
        _entry("second", second, root, CalibrationDatasetSplit.HOLDOUT, "camera-b"),
    )

    audit = audit_calibration_registry(registry, root)

    assert audit.valid
    assert audit.split_counts == {"calibration": 1, "holdout": 1}
    leaked = registry.model_copy(
        update={
            "entries": [
                registry.entries[0],
                registry.entries[1].model_copy(
                    update={"capture_or_generator_family": "camera-a"}
                ),
            ]
        }
    )
    unsafe = registry.model_copy(
        update={
            "entries": [
                registry.entries[0].model_copy(update={"relative_path": "../outside.jpg"}),
                registry.entries[1],
            ]
        }
    )

    assert any("family_split_leakage" in error for error in audit_calibration_registry(leaked, root).errors)
    assert any("unsafe_relative_path" in error for error in audit_calibration_registry(unsafe, root).errors)


def test_control_registry_batch_run_requires_opt_in_and_links_artifacts(tmp_path) -> None:
    root = tmp_path / "dataset"
    first = _image(root, "first.jpg", (160, 120), (10, 20, 30))
    second = _image(root, "second.jpg", (160, 120), (40, 50, 60))
    registry = CalibrationDatasetRegistry(
        cohort_name="local-control",
        intended_use=CalibrationDatasetPurpose.CONTROL_SMOKE,
        entries=[
            _entry("first", first, root, CalibrationDatasetSplit.CONTROL, "camera-a"),
            _entry("second", second, root, CalibrationDatasetSplit.CONTROL, "camera-b"),
        ],
    )
    config = _config()
    with pytest.raises(ValueError, match="allow_control_smoke"):
        run_camera_calibration_dataset(
            registry,
            root,
            config,
            tmp_path / "outputs",
            split=CalibrationDatasetSplit.CONTROL,
            backend=CanonicalCameraBackend(),
        )

    run = run_camera_calibration_dataset(
        registry,
        root,
        config,
        tmp_path / "outputs",
        split=CalibrationDatasetSplit.CONTROL,
        allow_control_smoke=True,
        backend=CanonicalCameraBackend(),
    )

    assert run.image_ids == ["first", "second"]
    assert set(run.result_artifacts) == {"first", "second"}
    summary = json.loads((tmp_path / "outputs" / run.summary_artifact).read_text(encoding="utf-8"))
    assert summary["result_count"] == 2
    assert summary["decision_readiness"]["e_cam_decision_threshold"] is None
    run_manifest = json.loads(
        (tmp_path / "outputs" / "calibration_dataset_run.json").read_text(encoding="utf-8")
    )
    assert "control_smoke_runs_are_not_independent_calibration_data" in run_manifest["limitations"]


def test_registry_audit_command_reads_written_registry(tmp_path) -> None:
    root = tmp_path / "dataset"
    image = _image(root, "first.jpg", (120, 90), (10, 20, 30))
    registry = _registry(
        _entry("first", image, root, CalibrationDatasetSplit.CALIBRATION, "camera-a")
    )
    registry_path = tmp_path / "registry.json"
    write_calibration_registry(registry_path, registry)

    assert main(
        [
            "camera-calibration-registry-audit",
            "--registry",
            str(registry_path),
            "--dataset-root",
            str(root),
        ]
    ) == 0


def _image(root: Path, filename: str, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    path = root / "images" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def _entry(
    image_id: str,
    path: Path,
    root: Path,
    split: CalibrationDatasetSplit,
    family: str,
) -> CalibrationRegistryEntry:
    with Image.open(path) as image:
        resolution = image.size
    return CalibrationRegistryEntry(
        image_id=image_id,
        split=split,
        relative_path=path.relative_to(root).as_posix(),
        source_type="camera",
        source_url_or_internal_provenance=f"internal:{image_id}",
        license="test-only",
        capture_or_generator_family=family,
        original_file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        transformations=[],
        resolution=resolution,
        c2pa_or_metadata_state="not_checked",
    )


def _registry(*entries: CalibrationRegistryEntry) -> CalibrationDatasetRegistry:
    return CalibrationDatasetRegistry(
        cohort_name="local-independent",
        intended_use=CalibrationDatasetPurpose.INDEPENDENT_CALIBRATION,
        entries=list(entries),
    )


def _config() -> P1CameraConfig:
    return P1CameraConfig.model_validate(
        {
            "config_version": "dataset-test-p1",
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
