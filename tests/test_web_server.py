from __future__ import annotations

import io
import json
from pathlib import Path

from image_trust.web.jobs import LocalJobStore, WebJobOutcome, WebJobStatus
from image_trust.web.server import create_app


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
    def runner(upload_path: Path, job_dir: Path) -> WebJobOutcome:
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
