"""Dependency-free WSGI API and static-file server for the local demo."""

from __future__ import annotations

import cgi
import ipaddress
import json
import mimetypes
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Callable
from urllib.parse import unquote
from wsgiref.simple_server import WSGIServer, make_server

from image_trust.geometry_ai.review_server import (
    GeometryRelationReviewStore,
    create_relation_review_app,
)
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
GEOMETRY_REVIEW_PREFIX = "/geometry-review"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """Keep one slow browser request from blocking every local route."""

    daemon_threads = True


def create_app(
    store: LocalJobStore,
    static_root: Path = STATIC_ROOT,
    remote_settings: RemoteVerificationSettings | None = None,
    relation_review_app: Callable | None = None,
) -> Callable:
    """Return a small WSGI application with upload, poll, cancel, and artifact routes."""

    def app(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = unquote(environ.get("PATH_INFO", "/"))
        if path == GEOMETRY_REVIEW_PREFIX:
            if relation_review_app is None:
                return _json(start_response, "404 Not Found", {"error": "not_found"})
            return _redirect(start_response, f"{GEOMETRY_REVIEW_PREFIX}/")
        if path.startswith(f"{GEOMETRY_REVIEW_PREFIX}/"):
            if relation_review_app is None:
                return _json(start_response, "404 Not Found", {"error": "not_found"})
            return _mount_relation_review(environ, start_response, relation_review_app)
        if method == "GET" and path == "/api/health":
            return _json(start_response, "200 OK", {"status": "ok"})
        if method == "GET" and path == "/api/capabilities":
            return _json(
                start_response,
                "200 OK",
                {"external_verification": remote_verification_capabilities(remote_settings)},
            )
        if method == "POST" and path == "/api/jobs":
            return _create_job(environ, start_response, store)
        if method == "DELETE" and path.startswith("/api/jobs/"):
            return _cancel_job(path, start_response, store)
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
    relation_review_root: Path | None = None,
    allow_non_loopback: bool = False,
) -> WSGIServer:
    """Construct, but do not start, a local-only server for CLI use and tests."""

    if not allow_non_loopback and not _is_loopback_host(host):
        raise ValueError("Demirror local server accepts loopback hosts only")
    remote_settings = load_remote_verification_settings(project_root)
    store = LocalJobStore(
        jobs_root,
        build_local_runner(project_root, remote_settings),
        worker_project_root=project_root,
    )
    review_store = None
    review_app = None
    if relation_review_root is not None:
        resolved_review_root = relation_review_root
        if not resolved_review_root.is_absolute():
            resolved_review_root = project_root / resolved_review_root
        if (resolved_review_root / "review_manifest.jsonl").is_file():
            review_store = GeometryRelationReviewStore(resolved_review_root)
            review_app = create_relation_review_app(review_store)
    server = make_server(
        host,
        port,
        create_app(
            store,
            remote_settings=remote_settings,
            relation_review_app=review_app,
        ),
        server_class=ThreadingWSGIServer,
    )
    server.job_store = store  # type: ignore[attr-defined]
    server.relation_review_store = review_store  # type: ignore[attr-defined]
    return server


def _mount_relation_review(environ, start_response, relation_review_app: Callable):
    child_environ = dict(environ)
    child_environ["SCRIPT_NAME"] = (
        str(environ.get("SCRIPT_NAME", "")) + GEOMETRY_REVIEW_PREFIX
    )
    path = str(environ.get("PATH_INFO", "/"))
    child_environ["PATH_INFO"] = path[len(GEOMETRY_REVIEW_PREFIX) :] or "/"
    return relation_review_app(child_environ, start_response)


def _redirect(start_response, location: str):
    start_response(
        "308 Permanent Redirect",
        [
            ("Location", location),
            ("Content-Length", "0"),
            ("Cache-Control", "no-store"),
        ],
    )
    return [b""]


def _is_loopback_host(host: str) -> bool:
    if host.lower() in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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


def _cancel_job(path: str, start_response, store: LocalJobStore):
    """Cancel exactly the job named by the request path, never another queued job."""

    fragments = path.split("/")
    if len(fragments) != 4:
        return _json(start_response, "404 Not Found", {"error": "not_found"})
    job = store.cancel(fragments[3])
    if job is None:
        return _json(start_response, "404 Not Found", {"error": "job_not_found"})
    snapshot = store.get_snapshot(job.job_id)
    if snapshot is None:
        return _json(start_response, "404 Not Found", {"error": "job_not_found"})
    return _json(start_response, "200 OK", _snapshot_payload(snapshot))


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
