"""Compare the frozen deterministic surfaces with independent human relations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from image_trust.geometry_ai.deterministic_surfaces import (
    DeterministicSurfaceBaselineResult,
)
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
)
from image_trust.geometry_ai.surface_comparison import (
    DEFAULT_SURFACE_COMPARISON_CONFIG,
    HumanRelationQualityReceipt,
    SurfaceComparisonConfig,
    compare_deterministic_surfaces_with_human,
)


def audit_comparison(
    blind_root: Path,
    baseline_root: Path,
    baseline_audit_path: Path,
    quality_receipt_path: Path,
    protocol_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], int]:
    for path, label in (
        (blind_root, "blind root"),
        (baseline_root, "baseline root"),
        (baseline_audit_path, "baseline audit"),
        (quality_receipt_path, "human quality receipt"),
        (protocol_path, "continuation protocol"),
        (report_path, "comparison report"),
    ):
        _reject_posthoc_path(path, label)

    protocol = _read_json(protocol_path)
    baseline_audit = _read_json(baseline_audit_path)
    manifest_path = blind_root / "review_manifest.jsonl"
    manifest_rows = _read_jsonl(manifest_path)
    config = _validate_registered_inputs(
        protocol,
        baseline_audit,
        baseline_audit_path,
        manifest_path,
        manifest_rows,
    )
    quality_receipt = HumanRelationQualityReceipt.model_validate(
        _read_json(quality_receipt_path)
    )
    packets: dict[str, GeometryRelationReviewPacket] = {}
    annotations: dict[str, GeometryRelationAnnotation] = {}
    baselines: dict[str, DeterministicSurfaceBaselineResult] = {}
    expected_results = {
        str(record["reviewer_id"]): str(record["result_sha256"])
        for record in baseline_audit["records"]
    }
    for row in manifest_rows:
        reviewer_id = str(row["reviewer_id"])
        if reviewer_id in packets:
            raise ValueError("blind manifest reviewer IDs must be unique")
        packet_path = _resolve_inside(blind_root, str(row["packet"]), "review packet")
        annotation_path = _resolve_inside(
            blind_root,
            str(row["annotation"]),
            "human annotation",
        )
        baseline_path = _resolve_inside(
            baseline_root,
            f"{reviewer_id}/deterministic_surface_baseline.json",
            "deterministic baseline result",
        )
        if _raw_sha256(baseline_path) != expected_results.get(reviewer_id):
            raise ValueError(f"deterministic result hash drift for {reviewer_id}")
        packets[reviewer_id] = GeometryRelationReviewPacket.model_validate(
            _read_json(packet_path)
        )
        annotations[reviewer_id] = GeometryRelationAnnotation.model_validate(
            _read_json(annotation_path)
        )
        baselines[reviewer_id] = DeterministicSurfaceBaselineResult.model_validate(
            _read_json(baseline_path)
        )

    comparison = compare_deterministic_surfaces_with_human(
        packets,
        annotations,
        baselines,
        quality_receipt=quality_receipt,
        config=config,
    )
    report_model = comparison.__class__.model_validate(
        {
            **comparison.model_dump(mode="json"),
            "continuation_protocol_sha256": _normalized_text_sha256(protocol_path),
            "baseline_audit_sha256": _normalized_text_sha256(baseline_audit_path),
            "review_manifest_sha256": _normalized_text_sha256(manifest_path),
        }
    )
    report = report_model.model_dump(mode="json")
    _atomic_write_json(report_path, report)
    if report_model.status == "waiting_for_human_annotations":
        return report, 2
    return report, 0 if report_model.passed else 1


def _validate_registered_inputs(
    protocol: dict[str, Any],
    baseline_audit: dict[str, Any],
    baseline_audit_path: Path,
    manifest_path: Path,
    manifest_rows: list[dict[str, Any]],
) -> SurfaceComparisonConfig:
    if protocol.get("schema_version") != (
        "demirror-geometry-surface-continuation-protocol-v1"
    ):
        raise ValueError("unexpected continuation protocol schema")
    if protocol.get("status") != (
        "registered_before_human_comparison_and_surface_conditioned_replay_implementation"
    ):
        raise ValueError("continuation protocol is not a preregistration")
    frozen = dict(protocol.get("frozen_inputs", {}))
    if frozen.get("review_manifest_sha256") != _normalized_text_sha256(
        manifest_path
    ):
        raise ValueError("blind manifest differs from the continuation protocol")
    if frozen.get("deterministic_surface_audit_sha256") != (
        _normalized_text_sha256(baseline_audit_path)
    ):
        raise ValueError("baseline audit differs from the continuation protocol")
    repository_root = Path(__file__).resolve().parents[1]
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
    for field, relative_path in frozen_file_hashes.items():
        if frozen.get(field) != _normalized_text_sha256(
            repository_root / relative_path
        ):
            raise ValueError(f"frozen continuation input drifted: {relative_path}")
    if len(manifest_rows) != int(frozen.get("packet_count", -1)):
        raise ValueError("blind packet count differs from the continuation protocol")
    reviewer_ids = sorted(str(row.get("reviewer_id", "")) for row in manifest_rows)
    if not reviewer_ids or any(not reviewer_id for reviewer_id in reviewer_ids):
        raise ValueError("blind manifest contains an empty reviewer ID")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise ValueError("blind manifest reviewer IDs must be unique")
    if frozen.get("reviewer_id_closure_sha256") != _canonical_hash(reviewer_ids):
        raise ValueError("reviewer ID closure differs from the continuation protocol")

    if baseline_audit.get("passed") is not True:
        raise ValueError("deterministic surface baseline audit did not pass")
    if baseline_audit.get("source_key_opened") is not False:
        raise ValueError("baseline audit unexpectedly opened the source key")
    if baseline_audit.get("origin_scoring_authorized") is not False:
        raise ValueError("baseline audit unexpectedly authorizes source scoring")
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
    if frozen.get("deterministic_result_hash_closure_sha256") != _canonical_hash(
        baseline_records
    ):
        raise ValueError("baseline result closure differs from the continuation protocol")
    if [record["reviewer_id"] for record in baseline_records] != reviewer_ids:
        raise ValueError("baseline audit reviewer IDs differ from the blind manifest")

    gates = dict(protocol.get("comparison_gates", {}))
    expected = SurfaceComparisonConfig(
        expected_packet_count=int(frozen["packet_count"]),
        terminal_annotation_ratio_minimum=float(
            gates["terminal_annotation_ratio_minimum"]
        ),
        completed_packet_count_minimum=int(gates["completed_packet_count_minimum"]),
        assessable_family_count_minimum=int(gates["assessable_family_count_minimum"]),
        comparable_family_fraction_minimum=float(
            gates["comparable_family_fraction_minimum"]
        ),
        active_line_assignment_fraction_minimum=float(
            gates["active_line_assignment_fraction_minimum"]
        ),
        macro_same_surface_pair_retention_minimum=float(
            gates["macro_same_surface_pair_retention_minimum"]
        ),
        macro_different_surface_pair_separation_minimum=float(
            gates["macro_different_surface_pair_separation_minimum"]
        ),
        split_family_recall_minimum=float(gates["split_family_recall_minimum"]),
        non_split_family_specificity_minimum=float(
            gates["non_split_family_specificity_minimum"]
        ),
    )
    if expected != DEFAULT_SURFACE_COMPARISON_CONFIG:
        raise ValueError("comparison defaults differ from the continuation protocol")
    return expected


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


def _resolve_inside(root: Path, relative_path: str, label: str) -> Path:
    resolved_root = root.resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{label} path must be relative")
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its declared root") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is missing")
    _reject_posthoc_path(resolved, label)
    return resolved


def _reject_posthoc_path(path: Path, label: str) -> None:
    if any(part.casefold() == "posthoc" for part in path.resolve().parts):
        raise ValueError(f"{label} path must not enter a posthoc directory")


def _normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--quality-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, exit_code = audit_comparison(
        args.blind_root,
        args.baseline_root,
        args.baseline_audit,
        args.quality_receipt,
        args.protocol,
        args.report,
    )
    print(
        json.dumps(
            {"status": report["status"], "decision": report["decision"]},
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
