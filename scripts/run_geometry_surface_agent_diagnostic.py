"""Run the preregistered AI-assisted surface comparison and G1-G4 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_trust.geometry_ai.deterministic_surfaces import (
    DeterministicSurfaceBaselineResult,
)
from image_trust.geometry_ai.measurement_types import GeometryMeasurementV2Result
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
)
from image_trust.geometry_ai.relation_validation import (
    validate_annotation_semantic_closure,
)
from image_trust.geometry_ai.surface_agent_diagnostic import (
    AgentSurfaceComparisonReport,
    assess_agent_surface_conditioned_g1_g4,
    build_agent_diagnostic_replay_authorization,
    compare_deterministic_surfaces_with_agent_annotations,
)


def run_diagnostic(
    blind_root: Path,
    annotations_root: Path,
    baseline_root: Path,
    baseline_audit_path: Path,
    agent_audit_path: Path,
    agent_protocol_path: Path,
    human_protocol_path: Path,
    diagnostic_protocol_path: Path,
    output_root: Path,
    report_path: Path,
) -> tuple[dict[str, Any], int]:
    for path, label in (
        (blind_root, "blind root"),
        (annotations_root, "agent annotations"),
        (baseline_root, "baseline root"),
        (baseline_audit_path, "baseline audit"),
        (agent_audit_path, "agent audit"),
        (agent_protocol_path, "agent protocol"),
        (human_protocol_path, "human continuation protocol"),
        (diagnostic_protocol_path, "agent diagnostic protocol"),
        (output_root, "diagnostic output"),
        (report_path, "diagnostic report"),
    ):
        _reject_posthoc_path(path, label)
    if output_root.exists():
        raise ValueError("agent diagnostic output root must not already exist")
    if report_path.exists():
        raise ValueError("agent diagnostic report must not already exist")
    staging_root = output_root.with_name(output_root.name + ".tmp")
    if staging_root.exists():
        raise ValueError("agent diagnostic staging root already exists")

    protocol = _read_json(diagnostic_protocol_path)
    agent_protocol = _read_json(agent_protocol_path)
    agent_audit = _read_json(agent_audit_path)
    baseline_audit = _read_json(baseline_audit_path)
    human_protocol = _read_json(human_protocol_path)
    manifest_path = blind_root / "review_manifest.jsonl"
    manifest_rows = _read_jsonl(manifest_path)
    role_directory_by_id = _validate_registered_inputs(
        protocol,
        agent_protocol,
        agent_audit,
        baseline_audit,
        human_protocol,
        diagnostic_protocol_path,
        agent_protocol_path,
        agent_audit_path,
        baseline_audit_path,
        human_protocol_path,
        manifest_path,
        manifest_rows,
    )
    packets, annotations = _load_and_validate_agent_annotations(
        blind_root,
        annotations_root,
        manifest_rows,
        role_directory_by_id,
        agent_audit,
    )
    baselines = _load_and_validate_baselines(
        baseline_root,
        baseline_audit,
        sorted(packets),
    )

    comparison_base = compare_deterministic_surfaces_with_agent_annotations(
        packets,
        annotations,
        baselines,
    )
    comparison = AgentSurfaceComparisonReport.model_validate(
        {
            **comparison_base.model_dump(mode="json"),
            "diagnostic_protocol_sha256": _normalized_text_sha256(
                diagnostic_protocol_path
            ),
            "agent_annotation_audit_sha256": _normalized_text_sha256(
                agent_audit_path
            ),
            "baseline_audit_sha256": _normalized_text_sha256(
                baseline_audit_path
            ),
            "review_manifest_sha256": _normalized_text_sha256(manifest_path),
        }
    )
    comparison_payload = comparison.model_dump(mode="json")
    comparison_sha256 = _canonical_hash(comparison_payload)
    authorization = build_agent_diagnostic_replay_authorization(
        comparison_payload,
        comparison_report_sha256=comparison_sha256,
        g1_g4_measurement_sha256=str(
            protocol["frozen_inputs"]["g1_g4_measurement_sha256"]
        ),
    )
    row_by_id = {str(row["reviewer_id"]): row for row in manifest_rows}

    report: dict[str, Any]
    try:
        staging_root.mkdir(parents=True)
        _atomic_write_json(staging_root / "agent_surface_comparison.json", comparison_payload)
        replay_records: list[dict[str, Any]] = []
        result_status_counts: Counter[str] = Counter()
        check_status_counts: dict[str, Counter[str]] = {
            check_id: Counter() for check_id in ("G1", "G2", "G3", "G4")
        }
        finding_counts: Counter[str] = Counter()
        for reviewer_id in authorization.reviewer_ids:
            row = row_by_id[reviewer_id]
            packet_path = _resolve_inside(
                blind_root,
                str(row["packet"]),
                "review packet",
            )
            packet = packets[reviewer_id]
            image_path = _resolve_inside(
                packet_path.parent,
                packet.assets.image,
                "anonymous packet image",
            )
            measurement_path = _resolve_inside(
                packet_path.parent,
                packet.assets.measurement,
                "packet geometry measurement",
            )
            with Image.open(image_path) as source:
                canonical_rgb = np.asarray(source.convert("RGB"))
            measurement = GeometryMeasurementV2Result.model_validate(
                _read_json(measurement_path)
            )
            diagnostic = assess_agent_surface_conditioned_g1_g4(
                canonical_rgb,
                measurement,
                packet,
                baselines[reviewer_id],
                authorization,
            )
            destination = staging_root / "packets" / reviewer_id
            destination.mkdir(parents=True)
            diagnostic_payload = diagnostic.model_dump(mode="json")
            _atomic_write_json(
                destination / "surface_conditioned_g1_g4.json",
                diagnostic_payload,
            )
            result_status_counts[diagnostic.result.status] += 1
            for check in diagnostic.result.checks:
                check_status_counts[check.check_id][check.status] += 1
                finding_counts[check.check_id] += len(check.findings)
            replay_records.append(
                {
                    "reviewer_id": reviewer_id,
                    "status": diagnostic.result.status,
                    "conditioned_region_count": len(
                        diagnostic.result.conditioned_regions
                    ),
                    "conditioned_family_count": len(
                        diagnostic.result.conditioned_families
                    ),
                    "check_statuses": {
                        check.check_id: check.status
                        for check in diagnostic.result.checks
                    },
                    "finding_counts": {
                        check.check_id: len(check.findings)
                        for check in diagnostic.result.checks
                    },
                    "result_sha256": _canonical_hash(diagnostic_payload),
                }
            )

        report = {
            "schema_version": "geometry-surface-agent-diagnostic-audit-v1",
            "status": "complete",
            "annotation_semantics": (
                "AI-assisted source-blind preannotation; not human ground truth"
            ),
            "diagnostic_protocol_sha256": _normalized_text_sha256(
                diagnostic_protocol_path
            ),
            "agent_annotation_audit_sha256": _normalized_text_sha256(
                agent_audit_path
            ),
            "baseline_audit_sha256": _normalized_text_sha256(
                baseline_audit_path
            ),
            "review_manifest_sha256": _normalized_text_sha256(manifest_path),
            "comparison_report_canonical_sha256": comparison_sha256,
            "implementation_sha256": {
                "scripts/run_geometry_surface_agent_diagnostic.py": (
                    _normalized_text_sha256(Path(__file__).resolve())
                ),
                "src/image_trust/geometry_ai/surface_agent_diagnostic.py": (
                    _normalized_text_sha256(
                        Path(__file__).resolve().parents[1]
                        / "src/image_trust/geometry_ai/"
                        "surface_agent_diagnostic.py"
                    )
                ),
            },
            "comparison": {
                key: value
                for key, value in comparison_payload.items()
                if key != "records"
            },
            "comparison_record_count": len(comparison.records),
            "authorized_diagnostic_packet_count": len(authorization.reviewer_ids),
            "processed_diagnostic_packet_count": len(replay_records),
            "result_status_counts": dict(sorted(result_status_counts.items())),
            "check_status_counts": {
                check_id: dict(sorted(counts.items()))
                for check_id, counts in check_status_counts.items()
            },
            "finding_counts": dict(sorted(finding_counts.items())),
            "records": replay_records,
            "counterfactual_human_thresholds_do_not_authorize_replay": True,
            "human_confirmation_still_required": True,
            "ai_assisted_annotations_used": True,
            "human_annotations_used": False,
            "posthoc_source_key_opened": False,
            "source_labels_used": False,
            "origin_scoring_authorized": False,
            "web_integration_authorized": False,
            "decision": "diagnostic_only_human_confirmation_still_required",
        }
        _atomic_write_json(staging_root / "audit_report.json", report)
        staging_root.replace(output_root)
        _atomic_write_json(report_path, report)
    except Exception:
        if staging_root.is_dir():
            shutil.rmtree(staging_root)
        raise
    return report, 0


def _validate_registered_inputs(
    protocol: dict[str, Any],
    agent_protocol: dict[str, Any],
    agent_audit: dict[str, Any],
    baseline_audit: dict[str, Any],
    human_protocol: dict[str, Any],
    diagnostic_protocol_path: Path,
    agent_protocol_path: Path,
    agent_audit_path: Path,
    baseline_audit_path: Path,
    human_protocol_path: Path,
    manifest_path: Path,
    manifest_rows: list[dict[str, Any]],
) -> dict[str, str]:
    if protocol.get("schema_version") != (
        "demirror-geometry-surface-agent-diagnostic-protocol-v1"
    ):
        raise ValueError("unexpected agent diagnostic protocol schema")
    if protocol.get("status") != (
        "registered_before_agent_surface_comparison_or_conditioned_replay"
    ):
        raise ValueError("agent diagnostic protocol is not preregistered")
    if protocol.get("ai_assisted_annotations_used") is not True:
        raise ValueError("agent diagnostic protocol does not disclose AI annotations")
    for name in (
        "human_annotations_used",
        "posthoc_source_key_opened",
        "source_labels_used",
        "origin_scoring_authorized",
        "web_integration_authorized",
    ):
        if protocol.get(name) is not False:
            raise ValueError(f"agent diagnostic protocol violates field {name}")

    frozen = dict(protocol.get("frozen_inputs", {}))
    expected_hashes = {
        "review_manifest_sha256": _normalized_text_sha256(manifest_path),
        "agent_assisted_protocol_sha256": _normalized_text_sha256(
            agent_protocol_path
        ),
        "agent_assisted_audit_sha256": _normalized_text_sha256(agent_audit_path),
        "deterministic_surface_audit_sha256": _normalized_text_sha256(
            baseline_audit_path
        ),
        "human_continuation_protocol_sha256": _normalized_text_sha256(
            human_protocol_path
        ),
    }
    repository_root = Path(__file__).resolve().parents[1]
    expected_hashes.update(
        {
            "surface_comparison_implementation_sha256": _normalized_text_sha256(
                repository_root
                / "src/image_trust/geometry_ai/surface_comparison.py"
            ),
            "surface_conditioned_implementation_sha256": _normalized_text_sha256(
                repository_root
                / "src/image_trust/geometry_ai/surface_conditioned.py"
            ),
            "g1_g4_measurement_sha256": _normalized_text_sha256(
                repository_root / "src/image_trust/geometry_ai/consistency_v2.py"
            ),
        }
    )
    for field, actual in expected_hashes.items():
        if frozen.get(field) != actual:
            raise ValueError(f"agent diagnostic frozen input drifted: {field}")
    if len(manifest_rows) != int(frozen.get("packet_count", -1)):
        raise ValueError("agent diagnostic packet count differs from its protocol")
    reviewer_ids = [str(row.get("reviewer_id", "")) for row in manifest_rows]
    if not reviewer_ids or len(reviewer_ids) != len(set(reviewer_ids)):
        raise ValueError("blind manifest reviewer IDs must be nonempty and unique")

    if agent_protocol.get("schema_version") != (
        "demirror-geometry-semantic-relation-agent-assisted-protocol-v1"
    ):
        raise ValueError("unexpected AI-assisted annotation protocol schema")
    role_directory_by_id: dict[str, str] = {}
    for role in agent_protocol.get("roles", []):
        first, last = role["manifest_positions"]
        output_subdirectory = str(role["output_subdirectory"])
        if output_subdirectory not in frozen.get(
            "agent_annotation_subdirectories",
            [],
        ):
            raise ValueError("AI-assisted role directory differs from protocol")
        for reviewer_id in reviewer_ids[first - 1 : last]:
            if reviewer_id in role_directory_by_id:
                raise ValueError("AI-assisted role ranges overlap")
            role_directory_by_id[reviewer_id] = output_subdirectory
    if set(role_directory_by_id) != set(reviewer_ids):
        raise ValueError("AI-assisted role ranges do not cover the manifest")

    if agent_audit.get("passed") is not True:
        raise ValueError("AI-assisted annotation audit did not pass")
    if agent_audit.get("source_key_opened") is not False:
        raise ValueError("AI-assisted annotation audit opened the source key")
    if agent_audit.get("origin_scoring_authorized") is not False:
        raise ValueError("AI-assisted annotation audit authorizes origin scoring")
    if agent_audit.get("annotation_semantics") != (
        "AI-assisted blind preannotation; not human ground truth"
    ):
        raise ValueError("AI-assisted annotation semantics changed")
    if agent_audit.get("agent_annotation_closure_sha256") != frozen.get(
        "agent_annotation_closure_sha256"
    ):
        raise ValueError("AI-assisted raw annotation closure differs")
    if agent_audit.get("semantic_annotation_closure_sha256") != frozen.get(
        "agent_semantic_annotation_closure_sha256"
    ):
        raise ValueError("AI-assisted semantic annotation closure differs")
    if agent_audit.get("annotation_status_counts") != frozen.get(
        "agent_annotation_status_counts"
    ):
        raise ValueError("AI-assisted annotation statuses differ")

    if baseline_audit.get("passed") is not True:
        raise ValueError("deterministic surface baseline audit did not pass")
    if baseline_audit.get("source_key_opened") is not False:
        raise ValueError("deterministic baseline opened the source key")
    if baseline_audit.get("origin_scoring_authorized") is not False:
        raise ValueError("deterministic baseline authorizes origin scoring")
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
        raise ValueError("deterministic result closure differs from protocol")
    if [record["reviewer_id"] for record in baseline_records] != sorted(reviewer_ids):
        raise ValueError("deterministic baseline reviewer closure differs")
    if human_protocol.get("origin_scoring_authorized") is not False:
        raise ValueError("human continuation protocol unexpectedly authorizes scoring")
    return role_directory_by_id


def _load_and_validate_agent_annotations(
    blind_root: Path,
    annotations_root: Path,
    manifest_rows: list[dict[str, Any]],
    role_directory_by_id: dict[str, str],
    agent_audit: dict[str, Any],
) -> tuple[
    dict[str, GeometryRelationReviewPacket],
    dict[str, GeometryRelationAnnotation],
]:
    packets: dict[str, GeometryRelationReviewPacket] = {}
    annotations: dict[str, GeometryRelationAnnotation] = {}
    raw_closure: list[dict[str, str]] = []
    semantic_closure: list[dict[str, str]] = []
    for row in manifest_rows:
        reviewer_id = str(row["reviewer_id"])
        packet_path = _resolve_inside(
            blind_root,
            str(row["packet"]),
            "review packet",
        )
        annotation_path = _resolve_inside(
            annotations_root,
            f"{role_directory_by_id[reviewer_id]}/{reviewer_id}.json",
            "AI-assisted annotation",
        )
        packet = GeometryRelationReviewPacket.model_validate(_read_json(packet_path))
        annotation = GeometryRelationAnnotation.model_validate(
            _read_json(annotation_path)
        )
        if packet.reviewer_id != reviewer_id or annotation.reviewer_id != reviewer_id:
            raise ValueError("AI-assisted packet or annotation identity differs")
        validate_annotation_semantic_closure(packet, annotation)
        packets[reviewer_id] = packet
        annotations[reviewer_id] = annotation
        raw_closure.append(
            {
                "reviewer_id": reviewer_id,
                "relative_path": annotation_path.relative_to(
                    annotations_root.resolve()
                ).as_posix(),
                "sha256": _raw_sha256(annotation_path),
            }
        )
        semantic_closure.append(
            {
                "reviewer_id": reviewer_id,
                "canonical_sha256": _canonical_hash(
                    annotation.model_dump(mode="json")
                ),
            }
        )
    if _canonical_hash(sorted(raw_closure, key=lambda value: value["relative_path"])) != (
        agent_audit.get("agent_annotation_closure_sha256")
    ):
        raise ValueError("AI-assisted raw annotation files drifted")
    if _canonical_hash(
        sorted(semantic_closure, key=lambda value: value["reviewer_id"])
    ) != agent_audit.get("semantic_annotation_closure_sha256"):
        raise ValueError("AI-assisted semantic annotations drifted")
    return packets, annotations


def _load_and_validate_baselines(
    baseline_root: Path,
    baseline_audit: dict[str, Any],
    reviewer_ids: list[str],
) -> dict[str, DeterministicSurfaceBaselineResult]:
    expected_hashes = {
        str(record["reviewer_id"]): str(record["result_sha256"])
        for record in baseline_audit["records"]
    }
    baselines: dict[str, DeterministicSurfaceBaselineResult] = {}
    for reviewer_id in reviewer_ids:
        path = _resolve_inside(
            baseline_root,
            f"{reviewer_id}/deterministic_surface_baseline.json",
            "deterministic baseline result",
        )
        if _raw_sha256(path) != expected_hashes.get(reviewer_id):
            raise ValueError(f"deterministic baseline result drifted: {reviewer_id}")
        baselines[reviewer_id] = DeterministicSurfaceBaselineResult.model_validate(
            _read_json(path)
        )
    return baselines


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
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--agent-audit", type=Path, required=True)
    parser.add_argument("--agent-protocol", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, exit_code = run_diagnostic(
        args.blind_root,
        args.annotations_root,
        args.baseline_root,
        args.baseline_audit,
        args.agent_audit,
        args.agent_protocol,
        args.human_protocol,
        args.diagnostic_protocol,
        args.output_root,
        args.report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
