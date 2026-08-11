from __future__ import annotations

import json
from pathlib import Path
from threading import Event

from image_trust.geometry_ai.measurement_types import (
    GeometryArtifactsV2,
    GeometryCheckV2,
    GeometryMeasurementV2Result,
)
from image_trust.web.jobs import (
    LocalJobStore,
    WebJob,
    WebJobOutcome,
    WebJobStatus,
    _geometry_v2_partial_summary,
    _geometry_v2_web_summary,
    _publish_geometry_v2_artifacts,
    _write_json,
)


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


def test_local_job_store_exposes_completed_cards_before_the_job_finishes(tmp_path) -> None:
    partial_written = Event()
    release_job = Event()

    def runner(_: Path, job_dir: Path, report_progress) -> WebJobOutcome:
        report_progress("geometry_complete", 20)
        (job_dir / "web_partial_result.json").write_text(
            json.dumps({"p0": {"completed": True}}), encoding="utf-8"
        )
        partial_written.set()
        assert release_job.wait(timeout=5)
        return WebJobOutcome(status=WebJobStatus.COMPLETED, result={"complete": True})

    store = LocalJobStore(tmp_path / "jobs", runner)
    try:
        created = store.create_job("example.jpg", b"image-bytes")
        assert partial_written.wait(timeout=5)
        partial_snapshot = store.get_snapshot(created.job_id)
        assert partial_snapshot is not None
        assert partial_snapshot.job.status is WebJobStatus.RUNNING
        assert partial_snapshot.result == {"p0": {"completed": True}}

        release_job.set()
        completed_snapshot = store.wait(created.job_id)
        assert completed_snapshot is not None
        assert completed_snapshot.result == {"complete": True}
        assert not (tmp_path / "jobs" / created.job_id / "web_partial_result.json").exists()
    finally:
        release_job.set()
        store.close()


def test_geometry_v2_web_summary_is_progressive_and_keeps_raw_lines_in_artifact(
    tmp_path,
) -> None:
    checks = [
        GeometryCheckV2(check_id="G1", title="parallel", status="available", anomaly_score=0.4),
        GeometryCheckV2(check_id="G2", title="vanishing", status="not_applicable"),
    ]
    partial = _geometry_v2_partial_summary(checks[:1])
    measurement = GeometryMeasurementV2Result(
        status="measurable",
        summary="done",
        checks=checks,
        artifacts=GeometryArtifactsV2(
            result_json="geometry_measurement_v2.json",
            consistency_overlay="consistency_overlay.png",
        ),
    )
    summary = _geometry_v2_web_summary(measurement)
    job_dir = tmp_path / "job"
    geometry_dir = job_dir / "geometry_v2"
    geometry_dir.mkdir(parents=True)
    (geometry_dir / "geometry_measurement_v2.json").write_text("{}", encoding="utf-8")
    (geometry_dir / "consistency_overlay.png").write_bytes(b"png")
    artifacts: dict[str, object] = {}

    _publish_geometry_v2_artifacts(artifacts, measurement, job_dir)

    assert partial["status"] == "running"
    assert [check["check_id"] for check in partial["checks"]] == ["G1"]
    assert [check["check_id"] for check in summary["checks"]] == ["G1", "G2"]
    assert "merged_lines" not in summary
    assert artifacts == {
        "geometry_v2_result": "geometry_v2/geometry_measurement_v2.json",
        "geometry_v2_consistency_overlay": "geometry_v2/consistency_overlay.png",
    }


def test_atomic_json_writer_retries_a_transient_windows_replace_lock(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "web_partial_result.json"
    original_replace = Path.replace
    attempts = 0

    def replace_once_locked(path: Path, target: Path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("transient reader lock")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_once_locked)

    _write_json(destination, {"p0": {"completed": True}})

    assert attempts == 2
    assert json.loads(destination.read_text(encoding="utf-8")) == {"p0": {"completed": True}}


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
