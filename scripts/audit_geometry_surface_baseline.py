"""Run and audit the deterministic surface baseline on source-blind packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from image_trust.geometry_ai.deterministic_surfaces import (
    DEFAULT_DETERMINISTIC_SURFACE_CONFIG,
    assess_deterministic_surface_baseline,
    write_deterministic_surface_diagnostics,
)
from image_trust.geometry_ai.relation_annotations import GeometryRelationReviewPacket


def audit_baseline(
    blind_root: Path,
    output_root: Path,
    protocol_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], int]:
    """Run every blind packet twice and close determinism and artifact gates."""

    for path, label in (
        (blind_root, "blind input"),
        (output_root, "baseline output"),
        (protocol_path, "protocol"),
        (report_path, "audit report"),
    ):
        _reject_posthoc_path(path, label)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    manifest_path = blind_root / "review_manifest.jsonl"
    rows = _read_jsonl(manifest_path)
    protocol_errors = _validate_protocol(protocol, manifest_path, len(rows))
    reviewer_ids = [str(row.get("reviewer_id", "")) for row in rows]
    errors: list[str] = []
    if not reviewer_ids or any(not reviewer_id for reviewer_id in reviewer_ids):
        errors.append("blind manifest contains an empty reviewer ID")
    if len(reviewer_ids) != len(set(reviewer_ids)):
        errors.append("blind manifest reviewer IDs must be unique")

    records: list[dict[str, Any]] = []
    all_deterministic = True
    all_artifacts = True
    all_available = True
    for row in rows:
        reviewer_id = str(row.get("reviewer_id", ""))
        try:
            packet_path = _resolve_inside(
                blind_root,
                str(row["packet"]),
                "review packet",
            )
            packet = GeometryRelationReviewPacket.model_validate(
                json.loads(packet_path.read_text(encoding="utf-8-sig"))
            )
            if packet.reviewer_id != reviewer_id:
                raise ValueError("packet reviewer ID differs from the blind manifest")
            image_path = _resolve_inside(
                packet_path.parent,
                packet.assets.image,
                "packet image",
            )
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            started = time.perf_counter()
            first = assess_deterministic_surface_baseline(image, packet)
            runtime_ms = (time.perf_counter() - started) * 1000.0
            second = assess_deterministic_surface_baseline(image, packet)
            first_payload = _canonical_bytes(first.model_dump(mode="json"))
            second_payload = _canonical_bytes(second.model_dump(mode="json"))
            deterministic = first_payload == second_payload
            all_deterministic = all_deterministic and deterministic
            all_available = all_available and first.status == "available"

            destination = output_root / reviewer_id
            written, manifest = write_deterministic_surface_diagnostics(
                image,
                packet,
                first,
                destination,
            )
            declared_artifacts = [
                destination / manifest.result_json,
                destination / manifest.surface_candidates_overlay,
                destination / manifest.family_partitions_overlay,
                destination / "deterministic_surface_baseline_artifacts.json",
            ]
            artifacts_complete = all(path.is_file() for path in declared_artifacts)
            temporary_files = list(destination.glob("*.tmp"))
            if temporary_files:
                artifacts_complete = False
            all_artifacts = all_artifacts and artifacts_complete
            partition_counts = Counter(
                partition.partition_status for partition in written.family_partitions
            )
            records.append(
                {
                    "reviewer_id": reviewer_id,
                    "status": written.status,
                    "line_count": len(packet.lines),
                    "appearance_component_count": len(written.appearance_components),
                    "surface_candidate_count": len(written.surface_candidates),
                    "family_partition_count": len(written.family_partitions),
                    "partition_status_counts": dict(sorted(partition_counts.items())),
                    "accepted_line_link_count": written.accepted_line_link_count,
                    "runtime_ms": round(runtime_ms, 3),
                    "deterministic_json": deterministic,
                    "artifacts_complete": artifacts_complete,
                    "result_sha256": _sha256(destination / manifest.result_json),
                }
            )
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"{reviewer_id or '<missing-id>'}: {error}")
            all_deterministic = False
            all_artifacts = False
            all_available = False

    runtime_values = [float(record["runtime_ms"]) for record in records]
    gates = {
        "registered_protocol_matches": not protocol_errors,
        "all_manifest_packets_processed_once": (
            len(records) == len(rows) == int(protocol.get("input", {}).get("packet_count", -1))
            and len({record["reviewer_id"] for record in records}) == len(records)
        ),
        "all_results_available_within_resource_cap": all_available and not errors,
        "all_results_byte_deterministic": all_deterministic and not errors,
        "all_artifacts_complete_without_temporaries": all_artifacts and not errors,
        "annotations_and_posthoc_not_read": True,
        "origin_scoring_remained_disabled": True,
    }
    passed = all(gates.values())
    aggregate_partition_counts: Counter[str] = Counter()
    for record in records:
        aggregate_partition_counts.update(record["partition_status_counts"])
    report = {
        "schema_version": "geometry-deterministic-surface-baseline-audit-v1",
        "status": "complete" if passed else "incomplete_or_invalid",
        "protocol_sha256": _sha256(protocol_path),
        "blind_manifest_sha256": _sha256(manifest_path),
        "implementation_sha256": _implementation_hashes(),
        "expected_packet_count": int(protocol.get("input", {}).get("packet_count", -1)),
        "processed_packet_count": len(records),
        "protocol_errors": protocol_errors,
        "validation_errors": errors,
        "aggregate": {
            "line_count": sum(int(record["line_count"]) for record in records),
            "appearance_component_count": sum(
                int(record["appearance_component_count"]) for record in records
            ),
            "surface_candidate_count": sum(
                int(record["surface_candidate_count"]) for record in records
            ),
            "family_partition_count": sum(
                int(record["family_partition_count"]) for record in records
            ),
            "partition_status_counts": dict(sorted(aggregate_partition_counts.items())),
            "accepted_line_link_count": sum(
                int(record["accepted_line_link_count"]) for record in records
            ),
            "runtime_ms_p50": _percentile(runtime_values, 0.50),
            "runtime_ms_p95": _percentile(runtime_values, 0.95),
            "runtime_ms_max": round(max(runtime_values), 3) if runtime_values else None,
        },
        "records": records,
        "gates": gates,
        "passed": passed,
        "annotation_files_read": False,
        "source_key_opened": False,
        "source_labels_used": False,
        "origin_scoring_authorized": False,
        "web_integration_authorized": False,
        "human_review_required_for_quality_metrics": True,
        "decision": (
            "eligible_for_independent_human_relation_comparison"
            if passed
            else "repair_implementation_without_opening_review_answers"
        ),
    }
    _atomic_write_json(report_path, report)
    return report, 0 if passed else 2


def _validate_protocol(
    protocol: dict[str, Any],
    manifest_path: Path,
    packet_count: int,
) -> list[str]:
    errors: list[str] = []
    if protocol.get("schema_version") != (
        "demirror-geometry-deterministic-surface-baseline-protocol-v1"
    ):
        errors.append("unexpected protocol schema")
    if protocol.get("status") != "registered_before_deterministic_baseline_implementation":
        errors.append("protocol is not an implementation preregistration")
    registered_input = dict(protocol.get("input", {}))
    if registered_input.get("review_manifest_sha256") != _sha256(manifest_path):
        errors.append("blind manifest hash differs from protocol")
    if registered_input.get("packet_count") != packet_count:
        errors.append("packet count differs from protocol")
    observed_config = json.loads(DEFAULT_DETERMINISTIC_SURFACE_CONFIG.model_dump_json())
    maximum_packet_lines = observed_config.pop("maximum_packet_lines")
    registered_config = protocol.get("fixed_configuration")
    if registered_config != observed_config:
        errors.append("fixed configuration differs from the complete default configuration")
    resource = dict(protocol.get("resource_boundary", {}))
    if resource.get("maximum_packet_lines") != maximum_packet_lines:
        errors.append("line-count resource cap differs from protocol")
    if protocol.get("origin_scoring_authorized") is not False:
        errors.append("protocol does not forbid origin scoring")
    return sorted(set(errors))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


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


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[1]
    paths = [
        "src/image_trust/geometry_ai/deterministic_surfaces.py",
        "scripts/run_geometry_surface_baseline.py",
        "scripts/audit_geometry_surface_baseline.py",
        "src/image_trust/geometry_ai/__init__.py",
        "tests/test_geometry_deterministic_surfaces.py",
        "tests/test_geometry_surface_baseline_audit.py",
    ]
    return {
        relative: _canonical_text_sha256(repository_root / relative)
        for relative in paths
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, exit_code = audit_baseline(
        args.blind_root,
        args.output_root,
        args.protocol,
        args.report,
    )
    print(json.dumps({"status": report["status"], "passed": report["passed"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
