from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

from PIL import Image

from image_trust.geometry_ai.relation_annotations import build_review_packet
from image_trust.geometry_ai.review_server import (
    GeometryRelationReviewStore,
    create_relation_review_app,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_measurement_fixture():
    module_path = REPOSITORY_ROOT / "tests/test_geometry_relation_annotations.py"
    spec = importlib.util.spec_from_file_location("review_server_measurement_fixture", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._measurement()


def _blind_root(tmp_path: Path) -> Path:
    blind_root = tmp_path / "blind"
    packet_dir = blind_root / "packets" / "grr-test"
    packet_dir.mkdir(parents=True)
    packet, annotation = build_review_packet("grr-test", _load_measurement_fixture())
    (packet_dir / "review_packet.json").write_text(
        packet.model_dump_json(indent=2), encoding="utf-8"
    )
    (packet_dir / "annotation.json").write_text(
        annotation.model_dump_json(indent=2), encoding="utf-8"
    )
    for relative_path in packet.assets.model_dump().values():
        path = packet_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".png":
            Image.new("RGB", packet.canonical_size, "white").save(path)
        else:
            path.write_text("{}\n", encoding="utf-8")
    for family in packet.family_proposals:
        path = packet_dir / family.detail_overlay
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", packet.canonical_size, "white").save(path)
    (blind_root / "review_manifest.jsonl").write_text(
        json.dumps(
            {
                "reviewer_id": "grr-test",
                "packet": "packets/grr-test/review_packet.json",
                "annotation": "packets/grr-test/annotation.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return blind_root


def _request(
    app,
    method: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    content_type: str | None = None,
):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    response: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        response["status"] = status
        response["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "CONTENT_TYPE": content_type or ("application/json" if payload is not None else ""),
        "wsgi.input": io.BytesIO(body),
    }
    response["body"] = b"".join(app(environ, start_response))
    return response


def test_state_and_task_payload_expose_only_blind_review_contract(tmp_path: Path) -> None:
    app = create_relation_review_app(GeometryRelationReviewStore(_blind_root(tmp_path)))

    state_response = _request(app, "GET", "/api/state")
    task_response = _request(app, "GET", "/api/tasks/grr-test")
    state = json.loads(state_response["body"])
    task = json.loads(task_response["body"])
    serialized = json.dumps({"state": state, "task": task})

    assert state_response["status"] == "200 OK"
    assert state["counts"] == {"pending": 1, "completed": 0, "unassessable": 0}
    assert task["packet"]["source_label_visibility"] == "forbidden"
    assert "sample_id" not in serialized
    assert "generator_family" not in serialized
    assert "posthoc" not in serialized


def test_only_packet_allowlisted_assets_are_served(tmp_path: Path) -> None:
    blind_root = _blind_root(tmp_path)
    packet_dir = blind_root / "packets/grr-test"
    (packet_dir / "undeclared.txt").write_text("secret", encoding="utf-8")
    app = create_relation_review_app(GeometryRelationReviewStore(blind_root))

    image = _request(app, "GET", "/api/tasks/grr-test/assets/image.png")
    undeclared = _request(app, "GET", "/api/tasks/grr-test/assets/undeclared.txt")
    traversal = _request(app, "GET", "/api/tasks/grr-test/assets/../annotation.json")

    assert image["status"] == "200 OK"
    assert image["headers"]["Content-Type"] == "image/png"
    assert undeclared["status"] == "404 Not Found"
    assert traversal["status"] == "404 Not Found"


def test_annotation_save_is_validated_and_atomic_from_client_view(tmp_path: Path) -> None:
    blind_root = _blind_root(tmp_path)
    annotation_path = blind_root / "packets/grr-test/annotation.json"
    app = create_relation_review_app(GeometryRelationReviewStore(blind_root))
    task = json.loads(_request(app, "GET", "/api/tasks/grr-test")["body"])
    annotation = task["annotation"]
    annotation["surfaces"] = [
        {
            "surface_id": "surface-001",
            "surface_kind": "roof",
            "polygon_normalized": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.9, "y": 0.1},
                {"x": 0.9, "y": 0.4},
            ],
            "line_ids": ["m0001"],
            "visibility": "clear",
            "note": "",
        }
    ]

    saved = _request(app, "PUT", "/api/tasks/grr-test/annotation", payload=annotation)
    persisted_before_invalid = annotation_path.read_bytes()
    invalid = dict(annotation)
    invalid["reviewer_id"] = "grr-other"
    rejected = _request(app, "PUT", "/api/tasks/grr-test/annotation", payload=invalid)

    assert saved["status"] == "200 OK"
    assert json.loads(saved["body"])["annotation"]["surfaces"][0]["surface_id"] == "surface-001"
    assert rejected["status"] == "422 Unprocessable Entity"
    assert annotation_path.read_bytes() == persisted_before_invalid
    assert not list(annotation_path.parent.glob(".annotation.json.*.tmp"))


def test_static_interface_has_security_headers_and_expected_contract(tmp_path: Path) -> None:
    app = create_relation_review_app(GeometryRelationReviewStore(_blind_root(tmp_path)))

    index = _request(app, "GET", "/")
    script = _request(app, "GET", "/static/app.js")

    assert index["status"] == "200 OK"
    assert script["status"] == "200 OK"
    assert index["headers"]["Cache-Control"] == "no-store"
    assert "script-src 'self'" in index["headers"]["Content-Security-Policy"]
    assert b'id="annotation-canvas"' in index["body"]
    assert b"api/tasks/" in script["body"]
    assert b"posthoc" not in script["body"].lower()
