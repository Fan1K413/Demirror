"""Optional, process-isolated Perspective Fields measurements for geometry v2.

The Perspective Fields checkpoint is comparatively large and its runtime has a
different dependency stack from the line-geometry path.  This module therefore
keeps it disabled by default and, when explicitly enabled, runs the existing P1
camera pipeline in a short-lived child process.  The child writes the same
source-neutral ``CameraExperimentResult`` contract used by GeoCalib.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from image_trust.camera.config import load_camera_config
from image_trust.camera.contracts import (
    CameraConsistencyObservation,
    CameraEstimateStatus,
    CameraExperimentResult,
)
from image_trust.camera.pipeline import analyze_camera_image
from image_trust.geometry_ai.measurement_types import (
    GeometryArtifactsV2,
    GeometryCheckV2,
    GeometryMeasurementV2Result,
)


PERSPECTIVE_FIELDS_ENV = "DEMIRROR_GEOMETRY_PERSPECTIVE_FIELDS"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class PerspectiveFieldsProcessResult:
    """Outcome of the optional child process without propagating its failures."""

    status: Literal["disabled", "completed", "timed_out", "failed"]
    camera_result: CameraExperimentResult | None = None
    result_path: Path | None = None
    limitations: tuple[str, ...] = ()


def perspective_fields_enabled(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the heavyweight optional channel was explicitly enabled."""

    source = os.environ if environment is None else environment
    return source.get(PERSPECTIVE_FIELDS_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def run_perspective_fields_isolated(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: float = 300.0,
    environment: Mapping[str, str] | None = None,
) -> PerspectiveFieldsProcessResult:
    """Run Perspective Fields in a bounded child process when opted in.

    Timeout, import, model, and output-validation failures are converted into a
    structured status so the line-geometry and GeoCalib paths can continue.
    ``sys.executable`` is deliberately used to keep the child in the current
    project's Python environment.
    """

    source_environment = os.environ if environment is None else environment
    if not perspective_fields_enabled(source_environment):
        return PerspectiveFieldsProcessResult(
            status="disabled",
            limitations=("geometry_g5_perspective_fields_not_enabled",),
        )
    if timeout_seconds <= 0:
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=("geometry_g5_perspective_fields_invalid_timeout",),
        )

    input_path = input_path.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    result_path = output_dir / "camera_result.json"
    try:
        config = load_camera_config(config_path)
    except Exception as exc:
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=(
                f"geometry_g5_perspective_fields_config_failed:{type(exc).__name__}",
            ),
        )
    if config.camera_backend.name != "perspective_fields":
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=("geometry_g5_perspective_fields_config_backend_mismatch",),
        )
    if not input_path.is_file():
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=("geometry_g5_perspective_fields_input_not_found",),
        )

    child_environment = dict(source_environment)
    for name in _THREAD_ENVIRONMENT:
        child_environment[name] = "1"
    child_environment["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "image_trust.geometry_ai.perspective_fields",
        "--worker",
        "--input",
        str(input_path),
        "--config",
        str(config_path),
        "--output",
        str(output_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=_project_root_for_config(config_path),
            env=child_environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return PerspectiveFieldsProcessResult(
            status="timed_out",
            limitations=("geometry_g5_perspective_fields_worker_timed_out",),
        )
    except Exception as exc:
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=(
                f"geometry_g5_perspective_fields_worker_failed:{type(exc).__name__}",
            ),
        )
    if completed.returncode != 0:
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=(
                f"geometry_g5_perspective_fields_worker_exit:{completed.returncode}",
            ),
        )

    try:
        camera_result = CameraExperimentResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=(
                f"geometry_g5_perspective_fields_result_invalid:{type(exc).__name__}",
            ),
        )
    if camera_result.run.requested_backend != "perspective_fields":
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=("geometry_g5_perspective_fields_result_backend_mismatch",),
        )
    if camera_result.input.sha256 != _sha256(input_path):
        return PerspectiveFieldsProcessResult(
            status="failed",
            limitations=("geometry_g5_perspective_fields_result_input_mismatch",),
        )
    return PerspectiveFieldsProcessResult(
        status="completed",
        camera_result=camera_result,
        result_path=result_path,
    )


def build_g5_check(
    geocalib_result: CameraExperimentResult | None = None,
    perspective_fields_result: CameraExperimentResult | None = None,
    *,
    perspective_fields_status: Literal[
        "disabled", "completed", "timed_out", "failed"
    ]
    | None = None,
    additional_limitations: Sequence[str] = (),
) -> GeometryCheckV2:
    """Combine source-neutral camera consistency measurements into G5."""

    backend_results = [
        ("geocalib", geocalib_result),
        ("perspective_fields", perspective_fields_result),
    ]
    backend_results = [(name, result) for name, result in backend_results if result is not None]
    measurements: dict[str, object] = {}
    measured_values: list[float] = []
    failed_backend_count = 0
    limitations = list(additional_limitations)
    for name, result in backend_results:
        assert result is not None
        full_status = result.full_image.status
        if full_status is CameraEstimateStatus.FAILED:
            failed_backend_count += 1
        e_cam = result.e_cam
        if (
            e_cam.observation is CameraConsistencyObservation.MEASURED
            and e_cam.value is not None
        ):
            measured_values.append(e_cam.value)
        measurements[name] = {
            "requested_backend": result.run.requested_backend,
            "full_image_status": full_status.value,
            "full_image_applicability": result.full_image.applicability,
            "full_image_coverage": result.full_image.coverage,
            "full_image_uncertainty": result.full_image.uncertainty.overall,
            "e_cam_observation": e_cam.observation.value,
            "e_cam": e_cam.value,
            "qualified_crop_count": len(e_cam.qualified_crop_ids),
            "required_qualified_crops": e_cam.required_qualified_crops,
            "component_means": e_cam.component_means,
        }

    measurements["attempted_backend_count"] = len(backend_results)
    measurements["measured_backend_count"] = len(measured_values)
    if perspective_fields_status is not None:
        measurements["perspective_fields_process_status"] = perspective_fields_status

    if measured_values:
        return GeometryCheckV2(
            check_id="G5",
            title="多裁剪相机与透视场一致性",
            status="available",
            anomaly_score=max(measured_values),
            measurements=measurements,
            limitations=sorted(
                set(["geometry_g5_measurement_not_source_evidence", *limitations])
            ),
        )
    if backend_results:
        status: Literal["not_applicable", "failed"] = (
            "failed" if failed_backend_count == len(backend_results) else "not_applicable"
        )
        return GeometryCheckV2(
            check_id="G5",
            title="多裁剪相机与透视场一致性",
            status=status,
            measurements=measurements,
            limitations=sorted(
                set(["geometry_g5_no_qualified_camera_measurement", *limitations])
            ),
        )
    if perspective_fields_status in {"failed", "timed_out"}:
        return GeometryCheckV2(
            check_id="G5",
            title="多裁剪相机与透视场一致性",
            status="failed",
            measurements=measurements,
            limitations=sorted(set(limitations)),
        )
    return GeometryCheckV2(
        check_id="G5",
        title="多裁剪相机与透视场一致性",
        status="not_run",
        measurements=measurements,
        limitations=sorted(
            set(["geometry_g5_camera_measurements_not_attached", *limitations])
        ),
    )


def attach_g5_measurements(
    measurement: GeometryMeasurementV2Result,
    geocalib_result: CameraExperimentResult | None = None,
    perspective_fields_run: PerspectiveFieldsProcessResult | None = None,
    *,
    perspective_fields_artifact: str | None = None,
) -> GeometryMeasurementV2Result:
    """Return a copy with exactly one up-to-date G5 check and optional artifact."""

    process_status = perspective_fields_run.status if perspective_fields_run else None
    process_limitations = perspective_fields_run.limitations if perspective_fields_run else ()
    perspective_result = (
        perspective_fields_run.camera_result if perspective_fields_run is not None else None
    )
    g5 = build_g5_check(
        geocalib_result,
        perspective_result,
        perspective_fields_status=process_status,
        additional_limitations=process_limitations,
    )
    checks = [check for check in measurement.checks if check.check_id != "G5"]
    checks.append(g5)
    artifacts: GeometryArtifactsV2 = measurement.artifacts
    if perspective_fields_artifact is not None:
        artifacts = artifacts.model_copy(
            update={"perspective_fields_result": perspective_fields_artifact}
        )
    return measurement.model_copy(update={"checks": checks, "artifacts": artifacts})


def _project_root_for_config(config_path: Path) -> Path:
    parent = config_path.parent
    return parent.parent if parent.name.lower() == "configs" else parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_worker_threads() -> None:
    """Apply PyTorch limits inside the child in addition to BLAS env limits."""

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # The camera adapter will report a missing dependency or inference
        # failure through its normal result contract.
        return


def _run_worker(input_path: Path, config_path: Path, output_dir: Path) -> int:
    config = load_camera_config(config_path)
    if config.camera_backend.name != "perspective_fields":
        raise ValueError("Perspective Fields worker requires its dedicated configuration.")
    _configure_worker_threads()
    analyze_camera_image(input_path, config, output_dir)
    return 0


def _parse_worker_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parse_worker_arguments(sys.argv[1:] if arguments is None else arguments)
    if not parsed.worker:
        raise SystemExit("This module is an internal Perspective Fields worker.")
    return _run_worker(parsed.input, parsed.config, parsed.output)


if __name__ == "__main__":
    raise SystemExit(main())
