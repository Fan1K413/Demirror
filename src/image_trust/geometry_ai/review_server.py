"""Local-only WSGI server for source-blind geometry relation annotation."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Callable
from urllib.parse import unquote

from pydantic import ValidationError

from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
    validate_annotation_against_packet,
)


STATIC_ROOT = Path(__file__).with_name("review_static")
MAX_ANNOTATION_BYTES = 2 * 1024 * 1024
REVIEWER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class ReviewTaskEntry:
    reviewer_id: str
    packet_path: Path
    annotation_path: Path
    packet: GeometryRelationReviewPacket
    assets: dict[str, Path]


class GeometryRelationReviewStore:
    """Validated view of one blind directory, with no post-hoc key access."""

    def __init__(self, blind_root: Path):
        root = blind_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("blind review root must be a directory")
        self.root = root
        self._lock = RLock()
        self._tasks = self._load_manifest(root / "review_manifest.jsonl")
        self._by_id = {task.reviewer_id: task for task in self._tasks}

    def _load_manifest(self, manifest_path: Path) -> tuple[ReviewTaskEntry, ...]:
        if not manifest_path.is_file():
            raise ValueError("blind review manifest is missing")
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("blind review manifest is empty")
        tasks: list[ReviewTaskEntry] = []
        reviewer_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"reviewer_id", "packet", "annotation"}:
                raise ValueError("blind manifest rows must contain only reviewer_id, packet, annotation")
            reviewer_id = str(row["reviewer_id"])
            if not REVIEWER_ID_PATTERN.fullmatch(reviewer_id) or reviewer_id in reviewer_ids:
                raise ValueError("blind manifest reviewer IDs must be unique safe identifiers")
            reviewer_ids.add(reviewer_id)
            packet_path = _resolve_declared_file(self.root, str(row["packet"]))
            annotation_path = _resolve_declared_file(self.root, str(row["annotation"]))
            if packet_path.parent != annotation_path.parent:
                raise ValueError("packet and annotation must share one task directory")
            packet = GeometryRelationReviewPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
            if packet.reviewer_id != reviewer_id:
                raise ValueError("packet reviewer_id does not match blind manifest")
            annotation = GeometryRelationAnnotation.model_validate_json(
                annotation_path.read_text(encoding="utf-8")
            )
            validate_annotation_against_packet(packet, annotation)
            assets = _resolve_packet_assets(packet_path.parent, packet)
            tasks.append(
                ReviewTaskEntry(
                    reviewer_id=reviewer_id,
                    packet_path=packet_path,
                    annotation_path=annotation_path,
                    packet=packet,
                    assets=assets,
                )
            )
        return tuple(tasks)

    def state(self) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
        counts = {"pending": 0, "completed": 0, "unassessable": 0}
        with self._lock:
            for index, task in enumerate(self._tasks, start=1):
                annotation = self._read_annotation(task)
                counts[annotation.status] += 1
                reviewed = sum(
                    review.verdict != "pending"
                    for review in annotation.proposed_family_reviews
                )
                tasks.append(
                    {
                        "reviewer_id": task.reviewer_id,
                        "position": index,
                        "status": annotation.status,
                        "surface_count": len(annotation.surfaces),
                        "reviewed_family_count": reviewed,
                        "family_count": len(task.packet.family_proposals),
                    }
                )
        return {
            "schema_version": "geometry-semantic-relation-review-ui-state-v1",
            "task_count": len(tasks),
            "counts": counts,
            "tasks": tasks,
        }

    def task_payload(self, reviewer_id: str) -> dict[str, Any] | None:
        task = self._by_id.get(reviewer_id)
        if task is None:
            return None
        with self._lock:
            annotation = self._read_annotation(task)
        return {
            "packet": task.packet.model_dump(mode="json"),
            "annotation": annotation.model_dump(mode="json"),
        }

    def save_annotation(
        self,
        reviewer_id: str,
        payload: dict[str, Any],
    ) -> GeometryRelationAnnotation | None:
        task = self._by_id.get(reviewer_id)
        if task is None:
            return None
        annotation = GeometryRelationAnnotation.model_validate(payload)
        validate_annotation_against_packet(task.packet, annotation)
        with self._lock:
            _atomic_write_json(task.annotation_path, annotation.model_dump(mode="json"))
        return annotation

    def asset_path(self, reviewer_id: str, relative_path: str) -> Path | None:
        task = self._by_id.get(reviewer_id)
        if task is None:
            return None
        normalized = _normalize_relative_asset(relative_path)
        if normalized is None:
            return None
        return task.assets.get(normalized)

    @staticmethod
    def _read_annotation(task: ReviewTaskEntry) -> GeometryRelationAnnotation:
        annotation = GeometryRelationAnnotation.model_validate_json(
            task.annotation_path.read_text(encoding="utf-8")
        )
        validate_annotation_against_packet(task.packet, annotation)
        return annotation


def create_relation_review_app(
    store: GeometryRelationReviewStore,
    static_root: Path = STATIC_ROOT,
) -> Callable:
    """Create the local annotation app. It exposes only validated blind assets."""

    static_root = static_root.resolve(strict=True)

    def app(environ, start_response):
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = unquote(str(environ.get("PATH_INFO", "/")))
        if method == "GET" and path == "/api/health":
            return _json(start_response, "200 OK", {"status": "ok"})
        if method == "GET" and path == "/api/state":
            return _json(start_response, "200 OK", store.state())
        if path.startswith("/api/tasks/"):
            return _handle_task_route(environ, start_response, store, method, path)
        if method == "GET":
            return _serve_static(start_response, static_root, path)
        return _json(start_response, "405 Method Not Allowed", {"error": "method_not_allowed"})

    return app


def _handle_task_route(environ, start_response, store, method: str, path: str):
    parts = path.split("/")
    if len(parts) < 4 or not parts[3]:
        return _json(start_response, "404 Not Found", {"error": "not_found"})
    reviewer_id = parts[3]
    if not REVIEWER_ID_PATTERN.fullmatch(reviewer_id):
        return _json(start_response, "404 Not Found", {"error": "task_not_found"})
    if method == "GET" and len(parts) == 4:
        payload = store.task_payload(reviewer_id)
        if payload is None:
            return _json(start_response, "404 Not Found", {"error": "task_not_found"})
        return _json(start_response, "200 OK", payload)
    if method == "PUT" and len(parts) == 5 and parts[4] == "annotation":
        try:
            payload = _read_json_body(environ)
            annotation = store.save_annotation(reviewer_id, payload)
        except RequestBodyError as error:
            return _json(start_response, error.status, {"error": error.code})
        except (ValidationError, ValueError, TypeError):
            return _json(start_response, "422 Unprocessable Entity", {"error": "invalid_annotation"})
        except OSError:
            return _json(start_response, "503 Service Unavailable", {"error": "annotation_write_failed"})
        if annotation is None:
            return _json(start_response, "404 Not Found", {"error": "task_not_found"})
        return _json(
            start_response,
            "200 OK",
            {"annotation": annotation.model_dump(mode="json")},
        )
    if method == "GET" and len(parts) >= 6 and parts[4] == "assets":
        asset = store.asset_path(reviewer_id, "/".join(parts[5:]))
        if asset is None:
            return _json(start_response, "404 Not Found", {"error": "asset_not_found"})
        return _file(start_response, asset)
    return _json(start_response, "404 Not Found", {"error": "not_found"})


class RequestBodyError(ValueError):
    def __init__(self, status: str, code: str):
        super().__init__(code)
        self.status = status
        self.code = code


def _read_json_body(environ) -> dict[str, Any]:
    content_type = str(environ.get("CONTENT_TYPE", ""))
    if not content_type.lower().startswith("application/json"):
        raise RequestBodyError("415 Unsupported Media Type", "json_required")
    try:
        content_length = int(environ.get("CONTENT_LENGTH") or "0")
    except ValueError as error:
        raise RequestBodyError("400 Bad Request", "invalid_content_length") from error
    if content_length <= 0:
        raise RequestBodyError("400 Bad Request", "empty_body")
    if content_length > MAX_ANNOTATION_BYTES:
        raise RequestBodyError("413 Payload Too Large", "annotation_too_large")
    body = environ["wsgi.input"].read(content_length)
    if len(body) != content_length:
        raise RequestBodyError("400 Bad Request", "incomplete_body")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestBodyError("400 Bad Request", "invalid_json") from error
    if not isinstance(payload, dict):
        raise RequestBodyError("400 Bad Request", "json_object_required")
    return payload


def _resolve_declared_file(root: Path, relative_path: str) -> Path:
    normalized = _normalize_relative_asset(relative_path)
    if normalized is None:
        raise ValueError("declared blind path must be relative and normalized")
    candidate = (root / Path(*PurePosixPath(normalized).parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("declared blind path escapes the blind directory") from error
    if not candidate.is_file():
        raise ValueError("declared blind file is missing")
    return candidate


def _resolve_packet_assets(
    packet_dir: Path,
    packet: GeometryRelationReviewPacket,
) -> dict[str, Path]:
    declared = list(packet.assets.model_dump().values()) + [
        family.detail_overlay for family in packet.family_proposals
    ]
    assets: dict[str, Path] = {}
    for relative_path in declared:
        normalized = _normalize_relative_asset(str(relative_path))
        if normalized is None or normalized in assets:
            raise ValueError("packet asset paths must be unique normalized relative paths")
        candidate = (packet_dir / Path(*PurePosixPath(normalized).parts)).resolve(strict=True)
        try:
            candidate.relative_to(packet_dir.resolve())
        except ValueError as error:
            raise ValueError("packet asset escapes its task directory") from error
        if not candidate.is_file():
            raise ValueError("packet asset is missing")
        assets[normalized] = candidate
    return assets


def _normalize_relative_asset(relative_path: str) -> str | None:
    candidate = PurePosixPath(relative_path.replace("\\", "/"))
    if candidate.is_absolute() or not candidate.parts:
        return None
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    return candidate.as_posix()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _serve_static(start_response, static_root: Path, path: str):
    requested = "index.html" if path in {"", "/"} else path.removeprefix("/static/")
    if path not in {"", "/"} and not path.startswith("/static/"):
        return _json(start_response, "404 Not Found", {"error": "not_found"})
    candidate = (static_root / requested).resolve()
    try:
        candidate.relative_to(static_root)
    except ValueError:
        return _json(start_response, "404 Not Found", {"error": "not_found"})
    if not candidate.is_file():
        return _json(start_response, "404 Not Found", {"error": "not_found"})
    return _file(start_response, candidate)


def _security_headers() -> list[tuple[str, str]]:
    return [
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
        (
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        ),
    ]


def _json(start_response, status: str, payload: dict[str, Any]):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            *_security_headers(),
        ],
    )
    return [body]


def _file(start_response, path: Path):
    body = path.read_bytes()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    start_response(
        "200 OK",
        [
            ("Content-Type", media_type),
            ("Content-Length", str(len(body))),
            *_security_headers(),
        ],
    )
    return [body]
