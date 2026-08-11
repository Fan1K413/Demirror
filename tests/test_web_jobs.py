from __future__ import annotations

from pathlib import Path
from threading import Event

from image_trust.web.jobs import LocalJobStore, WebJob, WebJobOutcome, WebJobStatus


def test_local_job_store_persists_completed_job_and_artifact(tmp_path) -> None:
    observed_progress: list[tuple[str, int]] = []

    def runner(upload_path: Path, job_dir: Path, report_progress) -> WebJobOutcome:
        assert upload_path.read_bytes() == b"image-bytes"
        report_progress("test_detector", 47)
        progress_job = WebJob.model_validate_json((job_dir / "job.json").read_text(encoding="utf-8"))
        observed_progress.append((progress_job.stage, progress_job.progress_percent))
        artifact = job_dir / "p0" / "lines_overlay.png"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"png")
        return WebJobOutcome(
            status=WebJobStatus.COMPLETED,
            result={"schema_version": "test", "artifacts": {"overlay": "p0/lines_overlay.png"}},
        )

    store = LocalJobStore(tmp_path / "jobs", runner)
    try:
        created = store.create_job("example.jpg", b"image-bytes")
        assert created.external_checks == []
        snapshot = store.wait(created.job_id)
        assert snapshot is not None
        assert snapshot.job.status is WebJobStatus.COMPLETED
        assert snapshot.job.progress_percent == 100
        assert observed_progress == [("test_detector", 47)]
        assert snapshot.result == {"schema_version": "test", "artifacts": {"overlay": "p0/lines_overlay.png"}}
        assert store.artifact_path(created.job_id, "p0/lines_overlay.png") is not None
        assert store.artifact_path(created.job_id, "job.json") is None
        assert store.artifact_path(created.job_id, "../job.json") is None
    finally:
        store.close()


def test_local_job_store_rejects_unsupported_or_oversized_uploads(tmp_path) -> None:
    store = LocalJobStore(tmp_path / "jobs", lambda *_: WebJobOutcome(status=WebJobStatus.COMPLETED))
    try:
        wrong_type = store.create_job("note.txt", b"hello")
        oversized = store.create_job("photo.jpg", b"x" * (25 * 1024 * 1024 + 1))
        assert wrong_type.status is WebJobStatus.REJECTED
        assert wrong_type.errors[0]["code"] == "unsupported_file_extension"
        assert oversized.status is WebJobStatus.REJECTED
        assert oversized.errors[0]["code"] == "upload_too_large"
    finally:
        store.close()


def test_local_job_store_marks_jobs_left_by_a_stopped_server_as_failed(tmp_path) -> None:
    jobs_root = tmp_path / "jobs"
    job_id = "a" * 32
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    job = WebJob(
        job_id=job_id,
        original_filename="scene.jpg",
        upload_filename="upload.jpg",
        created_at_utc="2026-08-08T13:53:33+00:00",
        updated_at_utc="2026-08-08T13:53:34+00:00",
        status=WebJobStatus.RUNNING,
        stage="running",
    )
    (job_dir / "job.json").write_text(job.model_dump_json(), encoding="utf-8")

    store = LocalJobStore(jobs_root, lambda *_: WebJobOutcome(status=WebJobStatus.COMPLETED))
    try:
        snapshot = store.get_snapshot(job_id)
        assert snapshot is not None
        assert snapshot.job.status is WebJobStatus.FAILED
        assert snapshot.job.stage == "failed"
        assert snapshot.job.progress_percent == 100
        assert "web_job_interrupted" in snapshot.job.limitations
        assert snapshot.job.errors[-1]["code"] == "server_interrupted"
    finally:
        store.close()


def test_local_job_store_cancels_only_the_named_queued_job(tmp_path) -> None:
    first_started = Event()
    release_first = Event()
    executions: list[bytes] = []

    def runner(upload_path: Path, _: Path, __) -> WebJobOutcome:
        payload = upload_path.read_bytes()
        executions.append(payload)
        if payload == b"first-image":
            first_started.set()
            assert release_first.wait(timeout=5)
        return WebJobOutcome(status=WebJobStatus.COMPLETED)

    store = LocalJobStore(tmp_path / "jobs", runner)
    try:
        first = store.create_job("first.jpg", b"first-image")
        assert first_started.wait(timeout=5)
        second = store.create_job("second.jpg", b"second-image")

        cancelled = store.cancel(second.job_id)
        assert cancelled is not None
        assert cancelled.job_id == second.job_id
        assert cancelled.status is WebJobStatus.CANCELLED
        assert store.get_snapshot(first.job_id).job.status is WebJobStatus.RUNNING

        release_first.set()
        assert store.wait(first.job_id).job.status is WebJobStatus.COMPLETED
        assert store.wait(second.job_id).job.status is WebJobStatus.CANCELLED
        assert executions == [b"first-image"]
    finally:
        release_first.set()
        store.close()
