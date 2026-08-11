from __future__ import annotations

import io
import json
from pathlib import Path
from threading import Event

from image_trust.web.jobs import LocalJobStore, WebJobOutcome, WebJobStatus
from image_trust.web.server import create_app
from image_trust.watermark.remote import RemoteVerificationSettings


def _request(app, method: str, path: str, body: bytes = b"", content_type: str = ""):
    response: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        response["status"] = status
        response["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    response["body"] = b"".join(app(environ, start_response))
    return response


def test_wsgi_server_uploads_polls_and_serves_evidence(tmp_path) -> None:
    def runner(upload_path: Path, job_dir: Path, report_progress) -> WebJobOutcome:
        report_progress("test_detector", 50)
        artifact = job_dir / "p0" / "lines_overlay.png"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(upload_path.read_bytes())
        return WebJobOutcome(
            status=WebJobStatus.COMPLETED,
            result={"artifacts": {"lines_overlay": "p0/lines_overlay.png"}},
        )

    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("Demirror", encoding="utf-8")
    store = LocalJobStore(tmp_path / "jobs", runner)
    app = create_app(store, static_root)
    boundary = "demirror-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="scene.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("ascii") + b"image-bytes" + f"\r\n--{boundary}--\r\n".encode("ascii")
    try:
        upload = _request(
            app,
            "POST",
            "/api/jobs",
            body,
            f"multipart/form-data; boundary={boundary}",
        )
        assert upload["status"] == "202 Accepted"
        created = json.loads(upload["body"])["job"]
        snapshot = store.wait(created["job_id"])
        assert snapshot is not None

        polled = _request(app, "GET", f"/api/jobs/{created['job_id']}")
        assert polled["status"] == "200 OK"
        assert json.loads(polled["body"])["job"]["status"] == "completed"
        assert json.loads(polled["body"])["job"]["progress_percent"] == 100

        artifact = _request(
            app,
            "GET",
            f"/api/jobs/{created['job_id']}/artifacts/p0/lines_overlay.png",
        )
        assert artifact["status"] == "200 OK"
        assert artifact["body"] == b"image-bytes"

        assert _request(app, "GET", "/")["body"] == b"Demirror"
    finally:
        store.close()


def test_wsgi_server_rejects_declared_oversized_upload(tmp_path) -> None:
    store = LocalJobStore(tmp_path / "jobs", lambda *_: WebJobOutcome(status=WebJobStatus.COMPLETED))
    app = create_app(store, tmp_path)
    response: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        response["status"] = status

    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": "/api/jobs",
        "CONTENT_TYPE": "multipart/form-data; boundary=x",
        "CONTENT_LENGTH": str(25 * 1024 * 1024 + 64 * 1024 + 1),
        "wsgi.input": io.BytesIO(),
    }
    try:
        body = b"".join(app(environ, start_response))
        assert response["status"] == "413 Payload Too Large"
        assert json.loads(body) == {"error": "upload_too_large"}
    finally:
        store.close()


def test_wsgi_capabilities_hide_key_and_upload_requires_explicit_opt_in(tmp_path) -> None:
    observed_checks: list[list[str]] = []

    def runner(_: Path, job_dir: Path, __) -> WebJobOutcome:
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        observed_checks.append(job["external_checks"])
        return WebJobOutcome(status=WebJobStatus.COMPLETED)

    store = LocalJobStore(tmp_path / "jobs", runner)
    app = create_app(
        store,
        tmp_path,
        RemoteVerificationSettings(openai_api_key="never-return-this"),
    )
    try:
        capabilities = _request(app, "GET", "/api/capabilities")
        capability_body = json.loads(capabilities["body"])
        assert capabilities["status"] == "200 OK"
        assert capability_body["external_verification"]["openai"]["configured"] is True
        assert capability_body["external_verification"]["google"]["mode"] == "manual_only"
        assert capability_body["external_verification"]["google"]["configured"] is False
        assert b"never-return-this" not in capabilities["body"]

        boundary = "demirror-opt-in-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="openai_provenance"\r\n\r\n'
            "1\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="scene.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode("ascii") + b"image-bytes" + f"\r\n--{boundary}--\r\n".encode("ascii")
        uploaded = _request(
            app,
            "POST",
            "/api/jobs",
            body,
            f"multipart/form-data; boundary={boundary}",
        )
        created = json.loads(uploaded["body"])["job"]
        store.wait(created["job_id"])
        assert observed_checks == [["openai"]]
    finally:
        store.close()


def test_wsgi_cancel_targets_only_the_job_in_the_url(tmp_path) -> None:
    first_started = Event()
    release_first = Event()

    def runner(upload_path: Path, _: Path, __) -> WebJobOutcome:
        if upload_path.read_bytes() == b"first-image":
            first_started.set()
            assert release_first.wait(timeout=5)
        return WebJobOutcome(status=WebJobStatus.COMPLETED)

    store = LocalJobStore(tmp_path / "jobs", runner)
    app = create_app(store, tmp_path)
    try:
        first = store.create_job("first.jpg", b"first-image")
        assert first_started.wait(timeout=5)
        second = store.create_job("second.jpg", b"second-image")

        cancelled = _request(app, "DELETE", f"/api/jobs/{second.job_id}")
        body = json.loads(cancelled["body"])
        assert cancelled["status"] == "200 OK"
        assert body["job"]["job_id"] == second.job_id
        assert body["job"]["status"] == "cancelled"
        assert store.get_snapshot(first.job_id).job.status == WebJobStatus.RUNNING

        release_first.set()
        assert store.wait(first.job_id).job.status is WebJobStatus.COMPLETED
        assert store.wait(second.job_id).job.status is WebJobStatus.CANCELLED
    finally:
        release_first.set()
        store.close()
