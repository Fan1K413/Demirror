"""Audit non-human geometry-surface continuation readiness without annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = (
    "scripts/audit_geometry_surface_continuation_readiness.py",
    "scripts/audit_geometry_surface_human_comparison.py",
    "scripts/extract_geometry_human_quality_receipt.py",
    "scripts/run_geometry_surface_conditioned_replay.py",
    "src/image_trust/geometry_ai/surface_comparison.py",
    "src/image_trust/geometry_ai/surface_conditioned.py",
    "src/image_trust/geometry_ai/__init__.py",
    "tests/test_geometry_surface_comparison.py",
    "tests/test_geometry_surface_conditioned.py",
    "tests/test_geometry_surface_continuation_tools.py",
)


def audit_readiness(
    blind_root: Path,
    baseline_audit_path: Path,
    protocol_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], int]:
    for path, label in (
        (blind_root, "blind root"),
        (baseline_audit_path, "baseline audit"),
        (protocol_path, "continuation protocol"),
        (report_path, "readiness report"),
    ):
        _reject_posthoc_path(path, label)
    protocol = _read_json(protocol_path)
    baseline_audit = _read_json(baseline_audit_path)
    manifest_path = blind_root / "review_manifest.jsonl"
    manifest_rows = _read_jsonl(manifest_path)
    repository_root = Path(__file__).resolve().parents[1]
    implementation_hashes = {
        path: _normalized_text_sha256(repository_root / path)
        for path in IMPLEMENTATION_PATHS
    }
    frozen = dict(protocol.get("frozen_inputs", {}))
    reviewer_ids = sorted(str(row.get("reviewer_id", "")) for row in manifest_rows)
    baseline_records = [
        {
            "reviewer_id": str(record["reviewer_id"]),
            "result_sha256": str(record["result_sha256"]),
        }
        for record in sorted(
            baseline_audit.get("records", []),
            key=lambda value: str(value["reviewer_id"]),
        )
    ]
    frozen_file_hashes = {
        "semantic_relation_pilot_protocol_sha256": (
            "research/records/2026-08-12/geometry/"
            "geometry_semantic_relation_pilot_protocol_v1.json"
        ),
        "deterministic_surface_protocol_sha256": (
            "research/records/2026-08-12/geometry/"
            "geometry_deterministic_surface_baseline_protocol_v1.json"
        ),
        "relation_annotation_contract_sha256": (
            "src/image_trust/geometry_ai/relation_annotations.py"
        ),
        "relation_semantic_closure_sha256": (
            "src/image_trust/geometry_ai/relation_validation.py"
        ),
        "g1_g4_measurement_sha256": (
            "src/image_trust/geometry_ai/consistency_v2.py"
        ),
    }
    gates = {
        "continuation_protocol_schema_matches": protocol.get("schema_version")
        == "demirror-geometry-surface-continuation-protocol-v1",
        "continuation_protocol_registered": protocol.get("status")
        == (
            "registered_before_human_comparison_and_"
            "surface_conditioned_replay_implementation"
        ),
        "blind_manifest_matches_protocol": frozen.get("review_manifest_sha256")
        == _normalized_text_sha256(manifest_path),
        "blind_packet_count_matches_protocol": len(manifest_rows)
        == int(frozen.get("packet_count", -1)),
        "reviewer_id_closure_matches_protocol": (
            reviewer_ids == sorted(set(reviewer_ids))
            and frozen.get("reviewer_id_closure_sha256")
            == _canonical_hash(reviewer_ids)
        ),
        "baseline_audit_matches_protocol": frozen.get(
            "deterministic_surface_audit_sha256"
        )
        == _normalized_text_sha256(baseline_audit_path),
        "baseline_engineering_gates_passed": baseline_audit.get("passed") is True,
        "baseline_result_closure_matches_protocol": (
            [record["reviewer_id"] for record in baseline_records] == reviewer_ids
            and frozen.get("deterministic_result_hash_closure_sha256")
            == _canonical_hash(baseline_records)
        ),
        "baseline_source_key_remained_closed": baseline_audit.get("source_key_opened")
        is False,
        "baseline_origin_scoring_remained_disabled": baseline_audit.get(
            "origin_scoring_authorized"
        )
        is False,
        "continuation_origin_scoring_remains_disabled": protocol.get(
            "origin_scoring_authorized"
        )
        is False,
        "continuation_web_integration_remains_disabled": protocol.get(
            "web_integration_authorized"
        )
        is False,
        "all_nonhuman_implementation_files_present": len(implementation_hashes)
        == len(IMPLEMENTATION_PATHS),
        "all_frozen_component_hashes_match": all(
            frozen.get(field)
            == _normalized_text_sha256(repository_root / relative_path)
            for field, relative_path in frozen_file_hashes.items()
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": "geometry-surface-continuation-readiness-v1",
        "status": "ready_waiting_for_independent_human_annotations"
        if passed
        else "nonhuman_readiness_failed",
        "continuation_protocol_sha256": _normalized_text_sha256(protocol_path),
        "baseline_audit_sha256": _normalized_text_sha256(baseline_audit_path),
        "review_manifest_sha256": _normalized_text_sha256(manifest_path),
        "packet_count": len(manifest_rows),
        "implementation_sha256": implementation_hashes,
        "gates": gates,
        "passed": passed,
        "human_annotation_files_opened": False,
        "ai_assisted_annotation_files_opened": False,
        "posthoc_source_key_opened": False,
        "source_labels_used": False,
        "origin_scoring_authorized": False,
        "web_integration_authorized": False,
        "remaining_external_input": (
            "independently completed human relation annotations and their "
            "passing source-neutral quality receipt"
        ),
        "next_steps_after_human_input": [
            "run the existing hidden-duplicate human review quality audit",
            "extract the source-neutral quality receipt",
            "run the frozen deterministic-versus-human surface comparison",
            "run surface-conditioned G1-G4 only if every comparison gate passes",
        ],
    }
    _atomic_write_json(report_path, report)
    return report, 0 if passed else 2


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("blind manifest rows must be JSON objects")
    return values


def _reject_posthoc_path(path: Path, label: str) -> None:
    if any(part.casefold() == "posthoc" for part in path.resolve().parts):
        raise ValueError(f"{label} path must not enter a posthoc directory")


def _normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, exit_code = audit_readiness(
        args.blind_root,
        args.baseline_audit,
        args.protocol,
        args.report,
    )
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
