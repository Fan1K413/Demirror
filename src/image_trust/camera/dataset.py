"""Auditable P1 camera-calibration dataset registration and batch execution."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.camera.backends import CameraBackend, resolve_camera_backend
from image_trust.camera.calibration import (
    summarize_camera_calibration,
    write_camera_calibration_summary,
)
from image_trust.camera.contracts import CameraExperimentResult, P1CameraConfig
from image_trust.camera.pipeline import analyze_camera_image


class CalibrationDatasetPurpose(str, Enum):
    CONTROL_SMOKE = "control_smoke"
    INDEPENDENT_CALIBRATION = "independent_calibration"


class CalibrationDatasetSplit(str, Enum):
    CONTROL = "control"
    CALIBRATION = "calibration"
    HOLDOUT = "holdout"


class CalibrationRegistryEntry(BaseModel):
    """One local image with its provenance and family-level split assignment."""

    model_config = ConfigDict(frozen=True)

    image_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    split: CalibrationDatasetSplit
    relative_path: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_url_or_internal_provenance: str = Field(min_length=1)
    license: str = Field(min_length=1)
    capture_or_generator_family: str = Field(min_length=1)
    original_file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformations: list[str] = Field(default_factory=list)
    resolution: tuple[int, int]
    c2pa_or_metadata_state: str = Field(min_length=1)

    @model_validator(mode="after")
    def resolution_is_positive(self) -> "CalibrationRegistryEntry":
        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise ValueError("Registry resolution must contain positive width and height.")
        return self


class CalibrationDatasetRegistry(BaseModel):
    """Versioned local registry; it stores labels separately from P1 outputs."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "p1-camera-calibration-registry-v1"
    cohort_name: str = Field(min_length=1)
    intended_use: CalibrationDatasetPurpose
    entries: list[CalibrationRegistryEntry] = Field(min_length=1)


class CalibrationDatasetAudit(BaseModel):
    """File, metadata, hash, resolution, and family-split audit result."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "p1-camera-calibration-registry-audit-v1"
    registry_digest: str
    cohort_name: str
    intended_use: CalibrationDatasetPurpose
    entry_count: int = Field(ge=0)
    split_counts: dict[str, int] = Field(default_factory=dict)
    valid: bool
    errors: list[str] = Field(default_factory=list)


class CameraCalibrationDatasetRun(BaseModel):
    """Link a registered cohort to its immutable P1 result artifacts."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "p1-camera-calibration-dataset-run-v1"
    created_at_utc: str
    registry_digest: str
    cohort_name: str
    intended_use: CalibrationDatasetPurpose
    split: CalibrationDatasetSplit
    config_version: str
    config_digest: str
    requested_backend: str
    image_ids: list[str]
    result_artifacts: dict[str, str]
    summary_artifact: str
    limitations: list[str] = Field(default_factory=list)


class CalibrationDatasetValidationError(ValueError):
    """Raised only after every registry validation error has been collected."""

    def __init__(self, audit: CalibrationDatasetAudit) -> None:
        self.audit = audit
        super().__init__("; ".join(audit.errors))


def load_calibration_registry(path: Path) -> CalibrationDatasetRegistry:
    return CalibrationDatasetRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def write_calibration_registry(path: Path, registry: CalibrationDatasetRegistry) -> None:
    _write_json(path, registry.model_dump(mode="json"))


def audit_calibration_registry(
    registry: CalibrationDatasetRegistry,
    dataset_root: Path,
) -> CalibrationDatasetAudit:
    """Audit a local registry without running a model or making network requests."""

    root = dataset_root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"dataset_root_is_not_a_directory:{dataset_root}")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    family_splits: dict[str, set[CalibrationDatasetSplit]] = defaultdict(set)
    for entry in registry.entries:
        if entry.image_id in seen_ids:
            errors.append(f"duplicate_image_id:{entry.image_id}")
        seen_ids.add(entry.image_id)
        if entry.original_file_hash in seen_hashes:
            errors.append(f"duplicate_original_file_hash:{entry.image_id}")
        seen_hashes.add(entry.original_file_hash)
        family_splits[entry.capture_or_generator_family].add(entry.split)
        if root.is_dir():
            _audit_entry_file(entry, root, errors)
    for family, splits in sorted(family_splits.items()):
        if len(splits) > 1:
            split_names = ",".join(sorted(split.value for split in splits))
            errors.append(f"family_split_leakage:{family}:{split_names}")
    counts = Counter(entry.split.value for entry in registry.entries)
    return CalibrationDatasetAudit(
        registry_digest=calibration_registry_digest(registry),
        cohort_name=registry.cohort_name,
        intended_use=registry.intended_use,
        entry_count=len(registry.entries),
        split_counts=dict(sorted(counts.items())),
        valid=not errors,
        errors=sorted(set(errors)),
    )


def require_valid_calibration_registry(
    registry: CalibrationDatasetRegistry,
    dataset_root: Path,
) -> CalibrationDatasetAudit:
    audit = audit_calibration_registry(registry, dataset_root)
    if not audit.valid:
        raise CalibrationDatasetValidationError(audit)
    return audit


def run_camera_calibration_dataset(
    registry: CalibrationDatasetRegistry,
    dataset_root: Path,
    config: P1CameraConfig,
    output_dir: Path,
    *,
    split: CalibrationDatasetSplit = CalibrationDatasetSplit.CALIBRATION,
    allow_control_smoke: bool = False,
    backend: CameraBackend | None = None,
) -> CameraCalibrationDatasetRun:
    """Run one registered split and produce results plus a descriptive summary.

    The runner never selects thresholds.  A control-only registry is blocked
    unless the caller explicitly opts into a smoke run, keeping existing local
    fixtures separate from an independent calibration cohort.
    """

    audit = require_valid_calibration_registry(registry, dataset_root)
    if split is CalibrationDatasetSplit.CONTROL:
        if registry.intended_use is not CalibrationDatasetPurpose.CONTROL_SMOKE:
            raise ValueError("Only control_smoke registries may run the control split.")
        if not allow_control_smoke:
            raise ValueError("Control runs require allow_control_smoke=True.")
    elif split is CalibrationDatasetSplit.CALIBRATION:
        if registry.intended_use is not CalibrationDatasetPurpose.INDEPENDENT_CALIBRATION:
            raise ValueError("Calibration runs require an independent_calibration registry.")
    else:
        raise ValueError("Holdout entries cannot be used by the calibration runner.")
    selected = sorted(
        (entry for entry in registry.entries if entry.split is split),
        key=lambda entry: entry.image_id,
    )
    if not selected:
        raise ValueError(f"Registry has no entries for split={split.value}.")
    root = dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    resolved_backend = backend or resolve_camera_backend(config.camera_backend)
    results: list[CameraExperimentResult] = []
    result_artifacts: dict[str, str] = {}
    for entry in selected:
        input_path = _entry_path(entry, root)
        artifact_dir = artifacts_dir / f"{entry.image_id}-{entry.original_file_hash[:12]}"
        results.append(
            analyze_camera_image(
                input_path,
                config,
                artifact_dir,
                backend=resolved_backend,
            )
        )
        result_artifacts[entry.image_id] = (
            artifact_dir / "camera_result.json"
        ).relative_to(output_dir).as_posix()
    summary = summarize_camera_calibration(
        results,
        config,
        registry.cohort_name,
        result_filenames=[result_artifacts[entry.image_id] for entry in selected],
    )
    summary_path = output_dir / "camera_calibration_summary.json"
    write_camera_calibration_summary(summary_path, summary)
    run = CameraCalibrationDatasetRun(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        registry_digest=audit.registry_digest,
        cohort_name=registry.cohort_name,
        intended_use=registry.intended_use,
        split=split,
        config_version=config.config_version,
        config_digest=summary.config_digest,
        requested_backend=config.camera_backend.name,
        image_ids=[entry.image_id for entry in selected],
        result_artifacts=result_artifacts,
        summary_artifact=summary_path.relative_to(output_dir).as_posix(),
        limitations=[
            "dataset_registry_labels_do_not_change_p1_camera_measurements",
            "control_smoke_runs_are_not_independent_calibration_data",
            "e_cam_decision_threshold_is_not_fitted_or_emitted",
        ],
    )
    _write_json(output_dir / "calibration_dataset_run.json", run.model_dump(mode="json"))
    return run


def calibration_registry_digest(registry: CalibrationDatasetRegistry) -> str:
    return hashlib.sha256(
        registry.model_dump_json(exclude_none=False).encode("utf-8")
    ).hexdigest()


def _audit_entry_file(
    entry: CalibrationRegistryEntry,
    root: Path,
    errors: list[str],
) -> None:
    try:
        path = _entry_path(entry, root)
    except ValueError as error:
        errors.append(str(error))
        return
    if not path.is_file():
        errors.append(f"image_path_not_found:{entry.image_id}:{entry.relative_path}")
        return
    try:
        actual_hash = _sha256(path)
    except OSError:
        errors.append(f"image_hash_unavailable:{entry.image_id}")
        return
    if actual_hash != entry.original_file_hash:
        errors.append(f"image_hash_mismatch:{entry.image_id}")
    try:
        with Image.open(path) as image:
            canonical = ImageOps.exif_transpose(image)
            actual_resolution = canonical.size
    except OSError:
        errors.append(f"image_decode_failed:{entry.image_id}")
        return
    if actual_resolution != entry.resolution:
        errors.append(f"image_resolution_mismatch:{entry.image_id}")


def _entry_path(entry: CalibrationRegistryEntry, root: Path) -> Path:
    relative = PurePosixPath(entry.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe_relative_path:{entry.image_id}:{entry.relative_path}")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"unsafe_relative_path:{entry.image_id}:{entry.relative_path}") from error
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
