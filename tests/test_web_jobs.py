from __future__ import annotations

from pathlib import Path

from image_trust.web.jobs import LocalJobStore, WebJob, WebJobOutcome, WebJobStatus


def test_local_job_store_persists_completed_job_and_artifact(tmp_path) -> None:
    def runner(upload_path: Path, job_dir: Path) -> WebJobOutcome:
        assert upload_path.read_bytes() == b"image-bytes"
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
        snapshot = store.wait(created.job_id)
        assert snapshot is not None
        assert snapshot.job.status is WebJobStatus.COMPLETED
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
        assert "web_job_interrupted" in snapshot.job.limitations
        assert snapshot.job.errors[-1]["code"] == "server_interrupted"
    finally:
        store.close()
