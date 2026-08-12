from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "surface_agent_diagnostic_tool",
    REPOSITORY_ROOT / "scripts/run_geometry_surface_agent_diagnostic.py",
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)

BLIND_ROOT = REPOSITORY_ROOT / "outputs/geometry_semantic_relation_pilot_v1/blind"
ANNOTATIONS_ROOT = (
    REPOSITORY_ROOT
    / "outputs/geometry_semantic_relation_pilot_v1/agent_annotations/v1"
)
BASELINE_ROOT = (
    REPOSITORY_ROOT
    / "outputs/geometry_semantic_relation_pilot_v1/"
    "deterministic_surface_baseline/v1/packets"
)
RECORD_ROOT = REPOSITORY_ROOT / "research/records/2026-08-12/geometry"
BASELINE_AUDIT = RECORD_ROOT / "geometry_deterministic_surface_baseline_audit_v1.json"
AGENT_AUDIT = RECORD_ROOT / "geometry_semantic_relation_agent_assisted_audit_v1.json"
AGENT_PROTOCOL = (
    RECORD_ROOT / "geometry_semantic_relation_agent_assisted_protocol_v1.json"
)
HUMAN_PROTOCOL = RECORD_ROOT / "geometry_surface_continuation_protocol_v1.json"
DIAGNOSTIC_PROTOCOL = (
    RECORD_ROOT / "geometry_surface_agent_diagnostic_protocol_v1.json"
)
DIAGNOSTIC_AUDIT = RECORD_ROOT / "geometry_surface_agent_diagnostic_audit_v1.json"


def _registered_inputs(protocol_path: Path = DIAGNOSTIC_PROTOCOL):
    protocol = tool._read_json(protocol_path)
    agent_protocol = tool._read_json(AGENT_PROTOCOL)
    agent_audit = tool._read_json(AGENT_AUDIT)
    baseline_audit = tool._read_json(BASELINE_AUDIT)
    human_protocol = tool._read_json(HUMAN_PROTOCOL)
    manifest_path = BLIND_ROOT / "review_manifest.jsonl"
    manifest_rows = tool._read_jsonl(manifest_path)
    mapping = tool._validate_registered_inputs(
        protocol,
        agent_protocol,
        agent_audit,
        baseline_audit,
        human_protocol,
        protocol_path,
        AGENT_PROTOCOL,
        AGENT_AUDIT,
        BASELINE_AUDIT,
        HUMAN_PROTOCOL,
        manifest_path,
        manifest_rows,
    )
    return mapping, manifest_rows


def test_agent_diagnostic_protocol_closes_all_blind_reviewers() -> None:
    mapping, manifest_rows = _registered_inputs()

    assert len(mapping) == 36
    assert set(mapping) == {str(row["reviewer_id"]) for row in manifest_rows}
    assert set(mapping.values()) == {
        "part_01_12_recovery",
        "part_13_24",
        "part_25_36",
    }


def test_agent_diagnostic_rejects_frozen_implementation_drift(
    tmp_path: Path,
) -> None:
    protocol = tool._read_json(DIAGNOSTIC_PROTOCOL)
    protocol["frozen_inputs"]["g1_g4_measurement_sha256"] = "0" * 64
    drifted = tmp_path / "protocol.json"
    drifted.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="g1_g4_measurement_sha256"):
        _registered_inputs(drifted)


def test_agent_diagnostic_never_overwrites_an_existing_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "existing"
    output_root.mkdir()

    with pytest.raises(ValueError, match="must not already exist"):
        tool.run_diagnostic(
            BLIND_ROOT,
            ANNOTATIONS_ROOT,
            BASELINE_ROOT,
            BASELINE_AUDIT,
            AGENT_AUDIT,
            AGENT_PROTOCOL,
            HUMAN_PROTOCOL,
            DIAGNOSTIC_PROTOCOL,
            output_root,
            tmp_path / "report.json",
        )


def test_tracked_agent_diagnostic_audit_binds_current_implementation() -> None:
    report = tool._read_json(DIAGNOSTIC_AUDIT)

    assert report["status"] == "complete"
    assert report["ai_assisted_annotations_used"] is True
    assert report["human_annotations_used"] is False
    assert report["posthoc_source_key_opened"] is False
    assert report["origin_scoring_authorized"] is False
    assert report["web_integration_authorized"] is False
    for relative_path, expected_hash in report["implementation_sha256"].items():
        assert tool._normalized_text_sha256(
            REPOSITORY_ROOT / relative_path
        ) == expected_hash
