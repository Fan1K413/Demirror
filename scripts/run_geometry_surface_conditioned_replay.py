"""Run authorized surface-conditioned G1-G4 replay on the frozen blind cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_trust.geometry_ai.deterministic_surfaces import (
    DeterministicSurfaceBaselineResult,
)
from image_trust.geometry_ai.measurement_types import GeometryMeasurementV2Result
from image_trust.geometry_ai.relation_annotations import GeometryRelationReviewPacket
from image_trust.geometry_ai.surface_conditioned import (
    assess_surface_conditioned_g1_g4,
    build_surface_replay_authorization,
)


def run_replay(
    blind_root: Path,
    baseline_root: Path,
    baseline_audit_path: Path,
    comparison_report_path: Path,
    protocol_path: Path,
    output_root: Path,
    report_path: Path,
) -> tuple[dict[str, Any], int]:
    for path, label in (
        (blind_root, "blind root"),
        (baseline_root, "baseline root"),
        (baseline_audit_path, "baseline audit"),
        (comparison_report_path, "comparison report"),
        (protocol_path, "continuation protocol"),
        (output_root, "replay output"),
        (report_path, "replay report"),
    ):
        _reject_posthoc_path(path, label)
    if output_root.exists():
        raise ValueError("replay output root must not already exist")
    staging_root = output_root.with_name(output_root.name + ".tmp")
    if staging_root.exists():
        raise ValueError("replay staging root already exists")

    protocol = _read_json(protocol_path)
    comparison = _read_json(comparison_report_path)
    baseline_audit = _read_json(baseline_audit_path)
    manifest_path = blind_root / "review_manifest.jsonl"
    manifest_rows = _read_jsonl(manifest_path)
    _validate_frozen_inputs(
        protocol,
        comparison,
        baseline_audit,
        protocol_path,
        baseline_audit_path,
        manifest_path,
    )
    authorization = build_surface_replay_authorization(
        comparison,
        comparison_report_sha256=_raw_sha256(comparison_report_path),
        g1_g4_measurement_sha256=str(
            protocol["frozen_inputs"]["g1_g4_measurement_sha256"]
        ),
    )
    row_by_id = {str(row["reviewer_id"]): row for row in manifest_rows}
    if len(row_by_id) != len(manifest_rows):
        raise ValueError("blind manifest reviewer IDs must be unique")
    expected_results = {
        str(record["reviewer_id"]): str(record["result_sha256"])
        for record in baseline_audit["records"]
    }

    prepared = []
    for reviewer_id in authorization.reviewer_ids:
        row = row_by_id.get(reviewer_id)
        if row is None:
            raise ValueError(f"authorized reviewer is missing from blind manifest: {reviewer_id}")
        packet_path = _resolve_inside(blind_root, str(row["packet"]), "review packet")
        packet = GeometryRelationReviewPacket.model_validate(_read_json(packet_path))
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
        baseline_path = _resolve_inside(
            baseline_root,
            f"{reviewer_id}/deterministic_surface_baseline.json",
            "deterministic baseline result",
        )
        if _raw_sha256(baseline_path) != expected_results.get(reviewer_id):
            raise ValueError(f"deterministic result hash drift for {reviewer_id}")
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        measurement = GeometryMeasurementV2Result.model_validate(
            _read_json(measurement_path)
        )
        baseline = DeterministicSurfaceBaselineResult.model_validate(
            _read_json(baseline_path)
        )
        result = assess_surface_conditioned_g1_g4(
            rgb,
            measurement,
            packet,
            baseline,
            authorization,
        )
        prepared.append(result)

    status_counts = Counter(result.status for result in prepared)
    check_status_counts: dict[str, Counter[str]] = {
        check_id: Counter() for check_id in ("G1", "G2", "G3", "G4")
    }
    finding_counts: Counter[str] = Counter()
    for result in prepared:
        for check in result.checks:
            check_status_counts[check.check_id][check.status] += 1
            finding_counts[check.check_id] += len(check.findings)
    report = {
        "schema_version": "geometry-surface-conditioned-replay-audit-v1",
        "status": "complete",
        "continuation_protocol_sha256": _normalized_text_sha256(protocol_path),
        "comparison_report_sha256": _raw_sha256(comparison_report_path),
        "baseline_audit_sha256": _normalized_text_sha256(baseline_audit_path),
        "authorized_packet_count": len(authorization.reviewer_ids),
        "processed_packet_count": len(prepared),
        "result_status_counts": dict(sorted(status_counts.items())),
        "check_status_counts": {
            check_id: dict(sorted(counts.items()))
            for check_id, counts in check_status_counts.items()
        },
        "finding_counts": dict(sorted(finding_counts.items())),
        "records": [
            {
                "reviewer_id": result.reviewer_id,
                "status": result.status,
                "conditioned_region_count": len(result.conditioned_regions),
                "conditioned_family_count": len(result.conditioned_families),
                "check_statuses": {
                    check.check_id: check.status for check in result.checks
                },
                "finding_counts": {
                    check.check_id: len(check.findings) for check in result.checks
                },
                "result_sha256": _canonical_hash(result.model_dump(mode="json")),
            }
            for result in prepared
        ],
        "source_key_opened": False,
        "source_labels_used": False,
        "human_annotations_used": False,
        "origin_scoring_authorized": False,
        "web_integration_authorized": False,
        "decision": "source_neutral_diagnostics_only",
    }

    staging_root.mkdir(parents=True)
    for result in prepared:
        destination = staging_root / result.reviewer_id
        destination.mkdir()
        _atomic_write_json(
            destination / "surface_conditioned_g1_g4.json",
            result.model_dump(mode="json"),
        )
    staging_root.replace(output_root)
    _atomic_write_json(report_path, report)
    return report, 0


def _validate_frozen_inputs(
    protocol: dict[str, Any],
    comparison: dict[str, Any],
    baseline_audit: dict[str, Any],
    protocol_path: Path,
    baseline_audit_path: Path,
    manifest_path: Path,
) -> None:
    if protocol.get("schema_version") != (
        "demirror-geometry-surface-continuation-protocol-v1"
    ):
        raise ValueError("unexpected continuation protocol schema")
    frozen = dict(protocol.get("frozen_inputs", {}))
    protocol_sha = _normalized_text_sha256(protocol_path)
    if comparison.get("continuation_protocol_sha256") != protocol_sha:
        raise ValueError("comparison report is bound to another continuation protocol")
    baseline_audit_sha = _normalized_text_sha256(baseline_audit_path)
    if frozen.get("deterministic_surface_audit_sha256") != baseline_audit_sha:
        raise ValueError("baseline audit differs from the continuation protocol")
    if comparison.get("baseline_audit_sha256") != baseline_audit_sha:
        raise ValueError("comparison report is bound to another baseline audit")
    manifest_sha = _normalized_text_sha256(manifest_path)
    if frozen.get("review_manifest_sha256") != manifest_sha:
        raise ValueError("blind manifest differs from the continuation protocol")
    if comparison.get("review_manifest_sha256") != manifest_sha:
        raise ValueError("comparison report is bound to another blind manifest")
    if baseline_audit.get("passed") is not True:
        raise ValueError("deterministic surface baseline audit did not pass")
    if baseline_audit.get("origin_scoring_authorized") is not False:
        raise ValueError("deterministic surface baseline unexpectedly authorizes scoring")
    consistency_path = (
        Path(__file__).resolve().parents[1]
        / "src/image_trust/geometry_ai/consistency_v2.py"
    )
    if frozen.get("g1_g4_measurement_sha256") != _normalized_text_sha256(
        consistency_path
    ):
        raise ValueError("frozen G1-G4 measurement implementation drifted")


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
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, exit_code = run_replay(
        args.blind_root,
        args.baseline_root,
        args.baseline_audit,
        args.comparison_report,
        args.protocol,
        args.output_root,
        args.report,
    )
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
