from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixtures = _load(
    "deterministic_surface_audit_fixtures",
    REPOSITORY_ROOT / "tests/test_geometry_deterministic_surfaces.py",
)
auditor = _load(
    "deterministic_surface_auditor",
    REPOSITORY_ROOT / "scripts/audit_geometry_surface_baseline.py",
)


def _setup(tmp_path: Path):
    blind = tmp_path / "blind"
    packets = blind / "packets"
    rows = []
    for index in range(2):
        reviewer_id = f"grr-audit-{index}"
        packet = fixtures._packet().model_copy(update={"reviewer_id": reviewer_id})
        packet_dir = packets / reviewer_id
        packet_dir.mkdir(parents=True)
        packet_path = packet_dir / "review_packet.json"
        packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
        fixtures._image().save(packet_dir / "image.png")
        rows.append(
            {
                "reviewer_id": reviewer_id,
                "packet": f"packets/{reviewer_id}/review_packet.json",
                "annotation": "posthoc/this-path-must-never-be-opened.json",
            }
        )
    manifest = blind / "review_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    registered = json.loads(fixtures.PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": registered["schema_version"],
                "status": registered["status"],
                "input": {
                    "review_manifest_sha256": auditor._sha256(manifest),
                    "packet_count": 2,
                },
                "resource_boundary": registered["resource_boundary"],
                "fixed_configuration": registered["fixed_configuration"],
                "origin_scoring_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    return blind, protocol


def test_audit_runs_twice_without_opening_annotation_paths(tmp_path: Path) -> None:
    blind, protocol = _setup(tmp_path)
    report_path = tmp_path / "report.json"

    report, exit_code = auditor.audit_baseline(
        blind,
        tmp_path / "outputs",
        protocol,
        report_path,
    )

    assert exit_code == 0
    assert report["passed"] is True
    assert report["processed_packet_count"] == 2
    assert report["annotation_files_read"] is False
    assert report["source_key_opened"] is False
    assert report["origin_scoring_authorized"] is False
    assert report["gates"]["all_results_byte_deterministic"] is True
    assert report["gates"]["all_artifacts_complete_without_temporaries"] is True
    assert report["aggregate"]["family_partition_count"] == 2
    assert set(report["implementation_sha256"]) == {
        "scripts/audit_geometry_surface_baseline.py",
        "scripts/run_geometry_surface_baseline.py",
        "src/image_trust/geometry_ai/__init__.py",
        "src/image_trust/geometry_ai/deterministic_surfaces.py",
        "tests/test_geometry_deterministic_surfaces.py",
        "tests/test_geometry_surface_baseline_audit.py",
    }
    assert report_path.is_file()
    assert len(list((tmp_path / "outputs").rglob("deterministic_surface_baseline.json"))) == 2
    assert not list((tmp_path / "outputs").rglob("*.tmp"))


def test_audit_rejects_protocol_configuration_drift(tmp_path: Path) -> None:
    blind, protocol = _setup(tmp_path)
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    payload["fixed_configuration"]["line_affinity_minimum"] = 0.99
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    report, exit_code = auditor.audit_baseline(
        blind,
        tmp_path / "outputs",
        protocol,
        tmp_path / "report.json",
    )

    assert exit_code == 2
    assert report["passed"] is False
    assert report["gates"]["registered_protocol_matches"] is False
    assert report["protocol_errors"] == [
        "fixed configuration differs from the complete default configuration"
    ]


def test_audit_rejects_incomplete_registered_configuration(tmp_path: Path) -> None:
    blind, protocol = _setup(tmp_path)
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    del payload["fixed_configuration"]["line_affinity_minimum"]
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    report, exit_code = auditor.audit_baseline(
        blind,
        tmp_path / "outputs",
        protocol,
        tmp_path / "report.json",
    )

    assert exit_code == 2
    assert report["gates"]["registered_protocol_matches"] is False


def test_implementation_hash_normalizes_checkout_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")

    assert auditor._canonical_text_sha256(lf) == auditor._canonical_text_sha256(crlf)


def test_audit_fails_closed_on_packet_path_escape(tmp_path: Path) -> None:
    blind, protocol = _setup(tmp_path)
    manifest = blind / "review_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["packet"] = "../outside.json"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    payload["input"]["review_manifest_sha256"] = auditor._sha256(manifest)
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    report, exit_code = auditor.audit_baseline(
        blind,
        tmp_path / "outputs",
        protocol,
        tmp_path / "report.json",
    )

    assert exit_code == 2
    assert report["source_key_opened"] is False
    assert any("escapes" in error for error in report["validation_errors"])
