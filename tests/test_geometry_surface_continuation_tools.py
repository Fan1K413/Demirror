from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load(
    "surface_continuation_comparison_script",
    "scripts/audit_geometry_surface_human_comparison.py",
)
readiness = _load(
    "surface_continuation_readiness_script",
    "scripts/audit_geometry_surface_continuation_readiness.py",
)
receipt_extractor = _load(
    "surface_continuation_receipt_script",
    "scripts/extract_geometry_human_quality_receipt.py",
)
replay = _load(
    "surface_continuation_replay_script",
    "scripts/run_geometry_surface_conditioned_replay.py",
)


PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "research/records/2026-08-12/geometry/geometry_surface_continuation_protocol_v1.json"
)
BASELINE_AUDIT_PATH = (
    REPOSITORY_ROOT
    / "research/records/2026-08-12/geometry/geometry_deterministic_surface_baseline_audit_v1.json"
)
BLIND_ROOT = REPOSITORY_ROOT / "outputs/geometry_semantic_relation_pilot_v1/blind"


def test_registered_protocol_matches_current_nonhuman_inputs() -> None:
    protocol = comparison._read_json(PROTOCOL_PATH)
    baseline_audit = comparison._read_json(BASELINE_AUDIT_PATH)
    manifest = BLIND_ROOT / "review_manifest.jsonl"
    rows = comparison._read_jsonl(manifest)

    config = comparison._validate_registered_inputs(
        protocol,
        baseline_audit,
        BASELINE_AUDIT_PATH,
        manifest,
        rows,
    )

    assert config.expected_packet_count == 36
    assert config.macro_same_surface_pair_retention_minimum == 0.70
    assert config.split_family_recall_minimum == 0.60


def test_readiness_never_opens_manifest_annotation_paths(tmp_path: Path) -> None:
    blind = tmp_path / "blind"
    blind.mkdir()
    manifest = blind / "review_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "reviewer_id": "grr-ready",
                "packet": "packets/missing.json",
                "annotation": "posthoc/must-not-open.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_audit = tmp_path / "baseline_audit.json"
    baseline_audit.write_text(
        json.dumps(
                {
                    "passed": True,
                    "source_key_opened": False,
                    "origin_scoring_authorized": False,
                    "records": [
                        {
                            "reviewer_id": "grr-ready",
                            "result_sha256": "a" * 64,
                        }
                    ],
                }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    frozen_components = readiness._read_json(PROTOCOL_PATH)["frozen_inputs"]
    protocol.write_text(
        json.dumps(
            {
                "schema_version": "demirror-geometry-surface-continuation-protocol-v1",
                "status": (
                    "registered_before_human_comparison_and_"
                    "surface_conditioned_replay_implementation"
                ),
                "origin_scoring_authorized": False,
                "web_integration_authorized": False,
                "frozen_inputs": {
                    **{
                        field: frozen_components[field]
                        for field in (
                            "semantic_relation_pilot_protocol_sha256",
                            "deterministic_surface_protocol_sha256",
                            "relation_annotation_contract_sha256",
                            "relation_semantic_closure_sha256",
                            "g1_g4_measurement_sha256",
                        )
                    },
                    "review_manifest_sha256": readiness._normalized_text_sha256(manifest),
                    "reviewer_id_closure_sha256": readiness._canonical_hash(
                        ["grr-ready"]
                    ),
                    "packet_count": 1,
                    "deterministic_surface_audit_sha256": (
                        readiness._normalized_text_sha256(baseline_audit)
                    ),
                    "deterministic_result_hash_closure_sha256": (
                        readiness._canonical_hash(
                            [
                                {
                                    "reviewer_id": "grr-ready",
                                    "result_sha256": "a" * 64,
                                }
                            ]
                        )
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    report, exit_code = readiness.audit_readiness(
        blind,
        baseline_audit,
        protocol,
        tmp_path / "report.json",
    )

    assert exit_code == 0
    assert report["passed"] is True
    assert report["human_annotation_files_opened"] is False
    assert report["posthoc_source_key_opened"] is False


def test_quality_receipt_file_contains_no_posthoc_summary(tmp_path: Path) -> None:
    pilot_audit = tmp_path / "pilot.json"
    pilot_audit.write_text(
        json.dumps(
            {
                "schema_version": "geometry-semantic-relation-pilot-audit-v1",
                "status": "complete",
                "packet_count": 36,
                "unique_source_count": 32,
                "source_key_opened": True,
                "origin_scoring_authorized": False,
                "quality": {
                    "completed_unique_count": 30,
                    "hidden_duplicate_group_count": 4,
                    "completed_duplicate_pair_count": 4,
                    "family_decision_agreement": 0.9,
                    "surface_line_pair_agreement": 0.9,
                    "gates": {
                        "completed_unique_ratio_at_least_0_75": True,
                        "completed_duplicate_pairs_at_least_3": True,
                        "family_decision_agreement_at_least_0_80": True,
                        "surface_line_pair_agreement_at_least_0_80": True,
                    },
                    "passed": True,
                },
                "posthoc_source_summary": {"secret_label": "must not transfer"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "receipt.json"

    receipt_extractor.extract_receipt(pilot_audit, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["source_details_transferred"] is False
    assert "posthoc_source_summary" not in payload
    assert "secret_label" not in output.read_text(encoding="utf-8")


def test_replay_gate_failure_publishes_no_output(tmp_path: Path) -> None:
    protocol = replay._read_json(PROTOCOL_PATH)
    comparison_report = tmp_path / "comparison.json"
    comparison_report.write_text(
        json.dumps(
            {
                "schema_version": "geometry-surface-human-comparison-v1",
                "status": "complete",
                "passed": False,
                "decision": "keep_as_review_visualization_without_tuning",
                "gates": {"failed": False},
                "continuation_protocol_sha256": replay._normalized_text_sha256(
                    PROTOCOL_PATH
                ),
                "baseline_audit_sha256": replay._normalized_text_sha256(
                    BASELINE_AUDIT_PATH
                ),
                "review_manifest_sha256": replay._normalized_text_sha256(
                    BLIND_ROOT / "review_manifest.jsonl"
                ),
                "source_key_opened": False,
                "source_labels_used": False,
                "ai_assisted_annotations_used": False,
                "origin_scoring_authorized": False,
                "web_integration_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "replay"

    with pytest.raises(ValueError, match="did not pass"):
        baseline_root = (
            REPOSITORY_ROOT
            / "outputs/geometry_semantic_relation_pilot_v1/"
            "deterministic_surface_baseline/v1/packets"
        )
        replay.run_replay(
            BLIND_ROOT,
            baseline_root,
            BASELINE_AUDIT_PATH,
            comparison_report,
            PROTOCOL_PATH,
            output_root,
            tmp_path / "replay_report.json",
        )

    assert output_root.exists() is False
    assert (tmp_path / "replay_report.json").exists() is False
    assert protocol["origin_scoring_authorized"] is False
