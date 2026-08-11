"""Persistent local analysis jobs used by the offline demonstration server."""

from __future__ import annotations

import json
import re
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from image_trust.camera.config import load_camera_config
from image_trust.camera.contracts import CameraEstimateStatus
from image_trust.camera.pipeline import analyze_camera_image
from image_trust.ai_likelihood.dda import assess_high_confidence_ai
from image_trust.origin import assess_origin
from image_trust.pipeline import analyze_image
from image_trust.provenance.c2pa import inspect_c2pa_asset, write_c2pa_record
from image_trust.provenance.config import load_c2pa_config
from image_trust.schemas import RunStatus
from image_trust.utils.config import load_config
from image_trust.watermark.suite import (
    assess_implicit_watermarks,
    build_offline_watermark_adapters,
)


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")


class WebJobStatus(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    REJECTED = "rejected"
    FAILED = "failed"


class WebJob(BaseModel):
    """Persisted local job state returned by the browser API."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "demirror-web-job-v1"
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    original_filename: str
    upload_filename: str
    created_at_utc: str
    updated_at_utc: str
    status: WebJobStatus
    stage: str
    progress_percent: int = Field(default=0, ge=0, le=100)
    limitations: list[str] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)
    result_artifact: str | None = None


class WebJobOutcome(BaseModel):
    """Result supplied by the analysis runner after a job is accepted."""

    model_config = ConfigDict(frozen=True)

    status: WebJobStatus
    result: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)


@dataclass(frozen=True)
class WebJobSnapshot:
    job: WebJob
    result: dict[str, Any] | None


ProgressReporter = Callable[[str, int], None]
JobRunner = Callable[[Path, Path, ProgressReporter], WebJobOutcome]


class LocalJobStore:
    """Thread-safe, disk-backed local analysis queue for a single demo machine."""

    def __init__(
        self,
        root: Path,
        runner: JobRunner,
        *,
        max_workers: int = 1,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="demirror")
        self._lock = threading.RLock()
        self._futures: dict[str, Future[None]] = {}
        self._mark_interrupted_jobs_failed()

    def create_job(self, filename: str, content: bytes) -> WebJob:
        """Persist an uploaded local image and schedule its analysis."""

        safe_name = _safe_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        now = _now()
        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id
        if suffix not in ALLOWED_SUFFIXES:
            job = WebJob(
                job_id=job_id,
                original_filename=filename,
                upload_filename="upload.bin",
                created_at_utc=now,
                updated_at_utc=now,
                status=WebJobStatus.REJECTED,
                stage="validation",
                progress_percent=100,
                limitations=["unsupported_file_extension"],
                errors=[
                    {
                        "code": "unsupported_file_extension",
                        "message": "Only PNG, JPEG, and static WebP uploads are accepted.",
                    }
                ],
            )
            self._write_job(job_dir, job)
            return job
        if len(content) > MAX_UPLOAD_BYTES:
            job = WebJob(
                job_id=job_id,
                original_filename=filename,
                upload_filename=f"upload{suffix}",
                created_at_utc=now,
                updated_at_utc=now,
                status=WebJobStatus.REJECTED,
                stage="validation",
                progress_percent=100,
                limitations=["upload_too_large"],
                errors=[
                    {
                        "code": "upload_too_large",
                        "message": f"Uploads must not exceed {MAX_UPLOAD_BYTES} bytes.",
                    }
                ],
            )
            self._write_job(job_dir, job)
            return job
        job_dir.mkdir(parents=True, exist_ok=False)
        upload_filename = f"upload{suffix}"
        (job_dir / upload_filename).write_bytes(content)
        job = WebJob(
            job_id=job_id,
            original_filename=filename,
            upload_filename=upload_filename,
            created_at_utc=now,
            updated_at_utc=now,
            status=WebJobStatus.QUEUED,
            stage="queued",
        )
        self._write_job(job_dir, job)
        with self._lock:
            self._futures[job_id] = self._executor.submit(self._run_job, job_id)
        return job

    def get_snapshot(self, job_id: str) -> WebJobSnapshot | None:
        """Load persisted state, so completed jobs survive a server restart."""

        job_dir = self._job_dir(job_id)
        if job_dir is None or not (job_dir / "job.json").is_file():
            return None
        job = WebJob.model_validate_json((job_dir / "job.json").read_text(encoding="utf-8"))
        result: dict[str, Any] | None = None
        if job.result_artifact:
            result_path = _safe_relative(job_dir, job.result_artifact)
            if result_path.is_file():
                raw = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    result = raw
        return WebJobSnapshot(job=job, result=result)

    def artifact_path(self, job_id: str, relative_path: str) -> Path | None:
        snapshot = self.get_snapshot(job_id)
        if snapshot is None or snapshot.result is None:
            return None
        artifacts = snapshot.result.get("artifacts")
        if not isinstance(artifacts, dict):
            return None
        published_paths = {value for value in artifacts.values() if isinstance(value, str)}
        if relative_path not in published_paths:
            return None
        job_dir = self._job_dir(job_id)
        if job_dir is None:
            return None
        try:
            path = _safe_relative(job_dir, relative_path)
        except ValueError:
            return None
        return path if path.is_file() else None

    def wait(self, job_id: str, timeout: float = 30.0) -> WebJobSnapshot | None:
        """Testing helper; the browser itself polls ``get_snapshot`` instead."""

        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.get_snapshot(job_id)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _run_job(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        if job_dir is None:
            return
        current = self.get_snapshot(job_id)
        if current is None:
            return
        self._update_progress(
            job_id,
            status=WebJobStatus.VALIDATING,
            stage="validating",
            progress_percent=5,
        )
        try:
            upload_path = job_dir / current.job.upload_filename
            self._update_progress(
                job_id,
                status=WebJobStatus.RUNNING,
                stage="starting",
                progress_percent=8,
            )

            def report_progress(stage: str, progress_percent: int) -> None:
                self._update_progress(
                    job_id,
                    status=WebJobStatus.RUNNING,
                    stage=stage,
                    progress_percent=progress_percent,
                )

            outcome = self._runner(upload_path, job_dir, report_progress)
            result_path = job_dir / "web_result.json"
            _write_json(result_path, outcome.result)
            latest = self.get_snapshot(job_id)
            if latest is None:
                return
            self._replace_job(
                latest.job,
                status=outcome.status,
                stage="complete",
                progress_percent=100,
                limitations=outcome.limitations,
                errors=outcome.errors,
                result_artifact=result_path.name,
            )
        except Exception as error:
            latest = self.get_snapshot(job_id)
            if latest is None:
                return
            self._replace_job(
                latest.job,
                status=WebJobStatus.FAILED,
                stage="failed",
                progress_percent=100,
                limitations=["web_analysis_failed"],
                errors=[{"code": type(error).__name__, "message": str(error)}],
            )

    def _replace_job(self, job: WebJob, **updates: Any) -> WebJob:
        updated = job.model_copy(
            update={"updated_at_utc": _now(), **updates},
        )
        self._write_job(self.root / updated.job_id, updated)
        return updated

    def _update_progress(
        self,
        job_id: str,
        *,
        status: WebJobStatus,
        stage: str,
        progress_percent: int,
    ) -> WebJob | None:
        """Persist a monotonic progress update for a running local job."""

        with self._lock:
            snapshot = self.get_snapshot(job_id)
            if snapshot is None:
                return None
            progress = max(snapshot.job.progress_percent, min(100, progress_percent))
            return self._replace_job(
                snapshot.job,
                status=status,
                stage=stage,
                progress_percent=progress,
            )

    def _write_job(self, job_dir: Path, job: WebJob) -> None:
        job_dir.mkdir(parents=True, exist_ok=True)
        _write_json(job_dir / "job.json", job.model_dump(mode="json"))

    def _mark_interrupted_jobs_failed(self) -> None:
        """Prevent a hard-stopped server from leaving its browser job spinning."""

        interrupted = {
            WebJobStatus.QUEUED,
            WebJobStatus.VALIDATING,
            WebJobStatus.RUNNING,
        }
        for job_path in self.root.glob("*/job.json"):
            try:
                job = WebJob.model_validate_json(job_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.status not in interrupted:
                continue
            errors = [
                *job.errors,
                {
                    "code": "server_interrupted",
                    "message": "The local analysis process stopped before this job completed.",
                },
            ]
            recovered = job.model_copy(
                update={
                    "updated_at_utc": _now(),
                    "status": WebJobStatus.FAILED,
                    "stage": "failed",
                    "progress_percent": 100,
                    "limitations": sorted({*job.limitations, "web_job_interrupted"}),
                    "errors": errors,
                }
            )
            self._write_job(job_path.parent, recovered)

    def _job_dir(self, job_id: str) -> Path | None:
        if not _JOB_ID_RE.fullmatch(job_id):
            return None
        return self.root / job_id


def build_local_runner(project_root: Path) -> JobRunner:
    """Create the actual P0 + local C2PA + configured P1 runner."""

    root = project_root.resolve()
    p0_config = load_config(root / "configs" / "p0.yaml")
    c2pa_config = load_c2pa_config(root / "configs" / "p1_c2pa.yaml")
    camera_config = load_camera_config(root / "configs" / "p1_geocalib.yaml")
    watermark_adapters = build_offline_watermark_adapters()

    def runner(
        upload_path: Path,
        job_dir: Path,
        report_progress: ProgressReporter,
    ) -> WebJobOutcome:
        report_progress("geometry", 10)
        p0_result = analyze_image(upload_path, p0_config, job_dir / "p0")
        p0_dump = p0_result.model_dump(mode="json")
        if p0_result.evidence.run_status is RunStatus.REJECTED:
            return WebJobOutcome(
                status=WebJobStatus.REJECTED,
                result={"p0": p0_dump},
                limitations=["p0_input_rejected"],
                errors=p0_result.diagnostics.errors,
            )
        if p0_result.evidence.run_status in {RunStatus.UNAVAILABLE, RunStatus.FAILED}:
            return WebJobOutcome(
                status=WebJobStatus.FAILED,
                result={"p0": p0_dump, "artifacts": {"input_image": upload_path.name}},
                limitations=["p0_analysis_not_available"],
                errors=p0_result.diagnostics.errors,
            )
        report_progress("provenance", 25)
        c2pa_record = inspect_c2pa_asset(upload_path, c2pa_config)
        write_c2pa_record(job_dir / "c2pa" / "c2pa_result.json", c2pa_record)
        report_progress("watermark", 27)
        watermark_result = assess_implicit_watermarks(upload_path, watermark_adapters)
        limitations = list(p0_result.evidence.limitations)
        p3_result = assess_high_confidence_ai(
            upload_path,
            c2pa_record,
            progress_callback=report_progress,
        )
        limitations.extend(p3_result.limitations)
        result: dict[str, Any] = {
            "schema_version": "demirror-web-result-v1",
            "p0": p0_dump,
            "p3": p3_result.model_dump(mode="json"),
            "c2pa": c2pa_record.model_dump(mode="json"),
            "watermark": watermark_result.model_dump(mode="json"),
            "artifacts": {
                "input_image": upload_path.name,
                "lines_overlay": "p0/lines_overlay.png",
                "anomalous_lines_overlay": "p0/anomalous_lines_overlay.png",
                "p0_result": "p0/result.json",
                "c2pa_result": "c2pa/c2pa_result.json",
            },
        }
        report_progress("camera", 88)
        try:
            camera_result = analyze_camera_image(upload_path, camera_config, job_dir / "camera")
            result["camera"] = camera_result.model_dump(mode="json")
            result["artifacts"]["camera_result"] = "camera/camera_result.json"
            if camera_result.full_image.status is not CameraEstimateStatus.OK:
                limitations.append("p1_camera_backend_not_available")
                status = WebJobStatus.PARTIAL
            else:
                status = WebJobStatus.COMPLETED
        except Exception as error:
            result["camera"] = {
                "status": "failed",
                "limitations": ["p1_camera_analysis_failed"],
                "error": {"code": type(error).__name__, "message": str(error)},
            }
            limitations.append("p1_camera_analysis_failed")
            status = WebJobStatus.PARTIAL
            camera_result = None
        report_progress("synthesis", 96)
        origin = assess_origin(
            upload_path,
            p3_result,
            c2pa_record,
            camera_result,
            watermark_result,
        )
        result["origin"] = origin.model_dump(mode="json")
        limitations.extend(origin.limitations)
        result["limitations"] = sorted(set(limitations))
        return WebJobOutcome(status=status, result=result, limitations=sorted(set(limitations)))

    return runner


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    return name or "upload.bin"


def _safe_relative(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe_relative_artifact_path")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("unsafe_relative_artifact_path") from error
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
