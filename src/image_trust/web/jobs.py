"""Persistent local analysis jobs used by the offline demonstration server."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from image_trust.camera.config import load_camera_config
from image_trust.camera.contracts import CameraEstimateStatus
from image_trust.camera.pipeline import analyze_camera_image
from image_trust.ai_likelihood.dda import assess_high_confidence_ai
from image_trust.geometry_ai.inference import assess_geometry_ai
from image_trust.origin import assess_origin
from image_trust.pipeline import analyze_image
from image_trust.provenance.c2pa import inspect_c2pa_asset, write_c2pa_record
from image_trust.provenance.config import load_c2pa_config
from image_trust.schemas import RunStatus
from image_trust.utils.config import load_config
from image_trust.watermark.suite import (
    aggregate_watermark_results,
    assess_implicit_watermarks,
    build_offline_watermark_adapters,
)
from image_trust.watermark.openai_provenance import OpenAIContentProvenanceAdapter
from image_trust.watermark.remote import RemoteVerificationSettings


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
    CANCELLED = "cancelled"


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
    external_checks: list[Literal["openai"]] = Field(default_factory=list)


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
        worker_project_root: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="demirror")
        self._lock = threading.RLock()
        self._futures: dict[str, Future[None]] = {}
        self._cancelled_job_ids: set[str] = set()
        self._worker_project_root = worker_project_root.resolve() if worker_project_root is not None else None
        self._worker_processes: dict[str, subprocess.Popen[bytes]] = {}
        self._mark_interrupted_jobs_failed()

    def create_job(
        self,
        filename: str,
        content: bytes,
        *,
        external_checks: Iterable[Literal["openai"]] = (),
    ) -> WebJob:
        """Persist an uploaded local image and schedule its analysis."""

        requested_external_checks = sorted(set(external_checks))
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
                external_checks=requested_external_checks,
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
                external_checks=requested_external_checks,
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
            external_checks=requested_external_checks,
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
            try:
                if result_path.is_file():
                    raw = json.loads(result_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        result = raw
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        elif job.status is WebJobStatus.RUNNING:
            partial_path = job_dir / "web_partial_result.json"
            try:
                if partial_path.is_file():
                    raw = json.loads(partial_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        result = raw
            except (OSError, ValueError, json.JSONDecodeError):
                # The worker publishes atomically. If Windows momentarily holds
                # the prior file open, keep the job running and try next poll.
                pass
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
            try:
                future.result(timeout=timeout)
            except CancelledError:
                pass
        return self.get_snapshot(job_id)

    def cancel(self, job_id: str) -> WebJob | None:
        """Cancel a queued or active local job and release the single-job queue."""

        with self._lock:
            snapshot = self.get_snapshot(job_id)
            if snapshot is None:
                return None
            if snapshot.job.status in {
                WebJobStatus.COMPLETED,
                WebJobStatus.PARTIAL,
                WebJobStatus.REJECTED,
                WebJobStatus.FAILED,
                WebJobStatus.CANCELLED,
            }:
                return snapshot.job
            self._cancelled_job_ids.add(job_id)
            future = self._futures.get(job_id)
            if future is not None:
                future.cancel()
            process = self._worker_processes.get(job_id)
            errors = [
                *snapshot.job.errors,
                {"code": "cancelled_by_user", "message": "The local analysis was cancelled by the user."},
            ]
            cancelled = self._replace_job(
                snapshot.job,
                status=WebJobStatus.CANCELLED,
                stage="cancelled",
                progress_percent=100,
                limitations=sorted({*snapshot.job.limitations, "web_job_cancelled"}),
                errors=errors,
            )
        if process is not None:
            _terminate_worker_process(process)
        return cancelled

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _run_job(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        if job_dir is None:
            return
        current = self.get_snapshot(job_id)
        if current is None or self._is_cancelled(job_id):
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

            if self._worker_project_root is None:
                outcome = self._runner(upload_path, job_dir, report_progress)
            else:
                outcome = self._run_worker_process(job_id, upload_path, job_dir)
            if outcome is None or self._is_cancelled(job_id):
                return
            result_path = job_dir / "web_result.json"
            _write_json(result_path, outcome.result)
            try:
                (job_dir / "web_partial_result.json").unlink(missing_ok=True)
            except OSError:
                # A reader can briefly retain the old partial file on Windows.
                # The final result artifact takes precedence for all snapshots.
                pass
            latest = self.get_snapshot(job_id)
            if latest is None or latest.job.status is WebJobStatus.CANCELLED:
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
            if latest is None or self._is_cancelled(job_id) or latest.job.status is WebJobStatus.CANCELLED:
                return
            self._replace_job(
                latest.job,
                status=WebJobStatus.FAILED,
                stage="failed",
                progress_percent=100,
                limitations=["web_analysis_failed"],
                errors=[{"code": type(error).__name__, "message": str(error)}],
            )

    def _run_worker_process(
        self,
        job_id: str,
        upload_path: Path,
        job_dir: Path,
    ) -> WebJobOutcome | None:
        """Run the heavyweight pipeline in a killable process for the local server."""

        if self._worker_project_root is None:
            raise RuntimeError("worker_project_root_not_configured")
        progress_path = job_dir / "worker_progress.json"
        outcome_path = job_dir / "worker_outcome.json"
        progress_path.unlink(missing_ok=True)
        outcome_path.unlink(missing_ok=True)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "image_trust.web.worker",
                "--project-root",
                str(self._worker_project_root),
                "--job-dir",
                str(job_dir),
                "--upload-path",
                str(upload_path),
            ],
            cwd=self._worker_project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        with self._lock:
            self._worker_processes[job_id] = process
        last_progress: tuple[str, int] | None = None
        try:
            while process.poll() is None:
                if self._is_cancelled(job_id):
                    _terminate_worker_process(process)
                    return None
                last_progress = self._read_worker_progress(job_id, progress_path, last_progress)
                time.sleep(0.05)
            last_progress = self._read_worker_progress(job_id, progress_path, last_progress)
            if self._is_cancelled(job_id):
                return None
            if outcome_path.is_file():
                raw = json.loads(outcome_path.read_text(encoding="utf-8"))
                return WebJobOutcome.model_validate(raw)
            raise RuntimeError(f"web_worker_exited_without_outcome:{process.returncode}")
        finally:
            if process.poll() is None and self._is_cancelled(job_id):
                _terminate_worker_process(process)
            with self._lock:
                self._worker_processes.pop(job_id, None)
            progress_path.unlink(missing_ok=True)
            outcome_path.unlink(missing_ok=True)

    def _read_worker_progress(
        self,
        job_id: str,
        progress_path: Path,
        previous: tuple[str, int] | None,
    ) -> tuple[str, int] | None:
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            stage = str(payload["stage"])
            progress_percent = int(payload["progress_percent"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return previous
        current = (stage, progress_percent)
        if current != previous:
            self._update_progress(
                job_id,
                status=WebJobStatus.RUNNING,
                stage=stage,
                progress_percent=progress_percent,
            )
        return current

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
            if snapshot is None or self._is_cancelled(job_id) or snapshot.job.status is WebJobStatus.CANCELLED:
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

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled_job_ids


def build_local_runner(
    project_root: Path,
    remote_settings: RemoteVerificationSettings | None = None,
) -> JobRunner:
    """Create the actual P0 + local C2PA + configured P1 runner."""

    root = project_root.resolve()
    p0_config = load_config(root / "configs" / "p0.yaml")
    c2pa_config = load_c2pa_config(root / "configs" / "p1_c2pa.yaml")
    camera_config = load_camera_config(root / "configs" / "p1_geocalib.yaml")
    geometry_relationship_model = root / "models" / "geometry_relationship_v2.json"
    watermark_adapters = build_offline_watermark_adapters()
    configured_remote = remote_settings or RemoteVerificationSettings()

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
        geometry_relationship = assess_geometry_ai(
            upload_path,
            model_path=geometry_relationship_model,
        )
        result: dict[str, Any] = {
            "schema_version": "demirror-web-result-v1",
            "p0": p0_dump,
            "geometry_ai": geometry_relationship.model_dump(mode="json"),
            "artifacts": {
                "input_image": upload_path.name,
                "lines_overlay": "p0/lines_overlay.png",
                "anomalous_lines_overlay": "p0/anomalous_lines_overlay.png",
                "p0_result": "p0/result.json",
            },
        }

        def write_partial_result() -> None:
            """Atomically expose every completed card while the job is running."""

            _write_json(job_dir / "web_partial_result.json", result)

        limitations = [*p0_result.evidence.limitations, *geometry_relationship.limitations]
        write_partial_result()
        report_progress("provenance", 25)
        c2pa_record = inspect_c2pa_asset(upload_path, c2pa_config)
        write_c2pa_record(job_dir / "c2pa" / "c2pa_result.json", c2pa_record)
        result["c2pa"] = c2pa_record.model_dump(mode="json")
        result["artifacts"]["c2pa_result"] = "c2pa/c2pa_result.json"
        write_partial_result()
        report_progress("watermark", 27)
        offline_watermark = assess_implicit_watermarks(upload_path, watermark_adapters)
        watermark_results = list(offline_watermark.adapters)
        requested_external_checks = _requested_external_checks(job_dir)
        if "openai" in requested_external_checks:
            report_progress("openai_provenance", 28)
            openai_result = assess_implicit_watermarks(
                upload_path,
                [OpenAIContentProvenanceAdapter(api_key=configured_remote.openai_api_key)],
            )
            watermark_results.extend(openai_result.adapters)
        watermark_result = aggregate_watermark_results(watermark_results)
        result["watermark"] = watermark_result.model_dump(mode="json")
        write_partial_result()
        p3_result = assess_high_confidence_ai(
            upload_path,
            c2pa_record,
            progress_callback=report_progress,
        )
        limitations.extend(p3_result.limitations)
        result["p3"] = p3_result.model_dump(mode="json")
        partial_origin = assess_origin(
            upload_path,
            p3_result,
            c2pa_record,
            watermark_result=watermark_result,
            geometry_result=geometry_relationship,
        )
        result["origin"] = partial_origin.model_dump(mode="json")
        limitations.extend(partial_origin.limitations)
        result["limitations"] = sorted(set(limitations))
        write_partial_result()
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
        if requested_external_checks and any(
            adapter.provider in requested_external_checks and adapter.run_status != "ok"
            for adapter in watermark_result.adapters
        ):
            status = WebJobStatus.PARTIAL
        report_progress("synthesis", 96)
        origin = assess_origin(
            upload_path,
            p3_result,
            c2pa_record,
            camera_result,
            watermark_result,
            geometry_relationship,
        )
        result["origin"] = origin.model_dump(mode="json")
        limitations.extend(origin.limitations)
        result["limitations"] = sorted(set(limitations))
        return WebJobOutcome(status=status, result=result, limitations=sorted(set(limitations)))

    return runner


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    return name or "upload.bin"


def _requested_external_checks(job_dir: Path) -> set[str]:
    try:
        job = WebJob.model_validate_json((job_dir / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set(job.external_checks)


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


def _terminate_worker_process(process: subprocess.Popen[bytes]) -> None:
    """Stop one job worker, including child processes it started on Windows."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    """Write JSON atomically, tolerating transient Windows reader locks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    last_error: OSError | None = None
    for attempt in range(12):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05 * (attempt + 1))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    if last_error is not None:
        raise last_error
