"""Dependency-free WSGI API and static-file server for the local demo."""

from __future__ import annotations

import cgi
import json
import mimetypes
from pathlib import Path
from typing import Callable
from urllib.parse import unquote
from wsgiref.simple_server import WSGIServer, make_server

from image_trust.web.jobs import (
    MAX_UPLOAD_BYTES,
    LocalJobStore,
    WebJobSnapshot,
    build_local_runner,
)
from image_trust.watermark.remote import (
    RemoteVerificationSettings,
    load_remote_verification_settings,
    remote_verification_capabilities,
)


STATIC_ROOT = Path(__file__).with_name("static")


def create_app(
    store: LocalJobStore,
    static_root: Path = STATIC_ROOT,
    remote_settings: RemoteVerificationSettings | None = None,
) -> Callable:
    """Return a small WSGI application with upload, poll, and artifact routes."""

    def app(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = unquote(environ.get("PATH_INFO", "/"))
        if method == "GET" and path == "/api/capabilities":
            return _json(
                start_response,
                "200 OK",
                {"external_verification": remote_verification_capabilities(remote_settings)},
            )
        if method == "POST" and path == "/api/jobs":
            return _create_job(environ, start_response, store)
        if method == "GET" and path.startswith("/api/jobs/"):
            return _get_job_or_artifact(path, start_response, store)
        if method == "GET":
            return _serve_static(path, start_response, static_root)
        return _json(start_response, "405 Method Not Allowed", {"error": "method_not_allowed"})

    return app


def serve_local_demo(
    project_root: Path,
    jobs_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> WSGIServer:
    """Construct, but do not start, a local-only server for CLI use and tests."""

    remote_settings = load_remote_verification_settings(project_root)
    store = LocalJobStore(jobs_root, build_local_runner(project_root, remote_settings))
    server = make_server(host, port, create_app(store, remote_settings=remote_settings))
    server.job_store = store  # type: ignore[attr-defined]
    return server


def _create_job(environ, start_response, store: LocalJobStore):
    content_type = environ.get("CONTENT_TYPE", "")
    if not content_type.startswith("multipart/form-data"):
        return _json(start_response, "400 Bad Request", {"error": "multipart_required"})
    try:
        content_length = int(environ.get("CONTENT_LENGTH") or "0")
    except ValueError:
        return _json(start_response, "400 Bad Request", {"error": "invalid_content_length"})
    if content_length <= 0:
        return _json(start_response, "400 Bad Request", {"error": "empty_upload"})
    if content_length > MAX_UPLOAD_BYTES + 64 * 1024:
        return _json(start_response, "413 Payload Too Large", {"error": "upload_too_large"})
    try:
        fields = cgi.FieldStorage(fp=environ["wsgi.input"], environ=environ, keep_blank_values=False)
    except (ValueError, OSError):
        return _json(start_response, "400 Bad Request", {"error": "invalid_multipart_upload"})
    upload = fields["file"] if "file" in fields else None
    if upload is None or not getattr(upload, "filename", None) or getattr(upload, "file", None) is None:
        return _json(start_response, "400 Bad Request", {"error": "file_field_required"})
    content = upload.file.read()
    external_checks = []
    if _field_truthy(fields, "openai_provenance"):
        external_checks.append("openai")
    job = store.create_job(str(upload.filename), content, external_checks=external_checks)
    return _json(start_response, "202 Accepted", {"job": job.model_dump(mode="json")})


def _field_truthy(fields: cgi.FieldStorage, name: str) -> bool:
    if name not in fields:
        return False
    field = fields[name]
    if isinstance(field, list):
        field = field[0]
    value = getattr(field, "value", "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_job_or_artifact(path: str, start_response, store: LocalJobStore):
    fragments = path.split("/")
    if len(fragments) < 4:
        return _json(start_response, "404 Not Found", {"error": "not_found"})
    job_id = fragments[3]
    if len(fragments) == 4:
        snapshot = store.get_snapshot(job_id)
        if snapshot is None:
            return _json(start_response, "404 Not Found", {"error": "job_not_found"})
        return _json(start_response, "200 OK", _snapshot_payload(snapshot))
    if len(fragments) >= 6 and fragments[4] == "artifacts":
        artifact = store.artifact_path(job_id, "/".join(fragments[5:]))
        if artifact is None:
            return _json(start_response, "404 Not Found", {"error": "artifact_not_found"})
        return _file(start_response, artifact)
    return _json(start_response, "404 Not Found", {"error": "not_found"})


def _snapshot_payload(snapshot: WebJobSnapshot) -> dict:
    return {
        "job": snapshot.job.model_dump(mode="json"),
        "result": snapshot.result,
    }


def _serve_static(path: str, start_response, static_root: Path):
    requested = "index.html" if path in {"", "/"} else path.lstrip("/")
    candidate = (static_root / requested).resolve()
    try:
        candidate.relative_to(static_root.resolve())
    except ValueError:
        return _json(start_response, "404 Not Found", {"error": "not_found"})
    if not candidate.is_file():
        return _json(start_response, "404 Not Found", {"error": "not_found"})
    return _file(start_response, candidate)


def _json(start_response, status: str, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    start_response(
        status,
        [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))],
    )
    return [body]


def _file(start_response, path: Path):
    body = path.read_bytes()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    start_response("200 OK", [("Content-Type", media_type), ("Content-Length", str(len(body)))])
    return [body]
