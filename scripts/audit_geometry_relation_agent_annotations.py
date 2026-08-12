"""Validate AI-assisted blind annotations without opening the post-hoc source key."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
)
from image_trust.geometry_ai.relation_validation import (
    validate_annotation_semantic_closure,
)


FORBIDDEN_KEYS = {
    "sample_id",
    "relative_path",
    "original_relative_path",
    "original_sha256",
    "declared_source_slice",
    "archive_name",
    "generator_family",
    "label_name",
    "source_group",
    "hidden_duplicate",
}
FORBIDDEN_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:pixart|sdxl|stable[ -]diffusion|midjourney|dall[ -]?e|imagen|flux)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:real|generated|synthetic|camera)[ -]?(?:photo|image)\b",
        re.IGNORECASE,
    ),
    re.compile(r"实拍|生成图|相机图|AI[ -]?生成", re.IGNORECASE),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _human_review_reasons(annotation: GeometryRelationAnnotation) -> list[str]:
    reasons: set[str] = set()
    if annotation.status == "unassessable":
        reasons.add("packet_unassessable")
    for surface in annotation.surfaces:
        if surface.visibility == "uncertain":
            reasons.add("uncertain_surface_visibility")
        if surface.surface_kind == "uncertain":
            reasons.add("uncertain_surface_kind")
    for review in annotation.proposed_family_reviews:
        if review.verdict == "unassessable":
            reasons.add("family_unassessable")
        if review.verdict == "geometry_inconsistent_within_surface":
            reasons.add("geometry_inconsistent_candidate")
        if review.outlier_line_ids:
            reasons.add("family_outliers_recorded")
    for relation in annotation.additional_relations:
        if relation.verdict == "uncertain":
            reasons.add("additional_relation_uncertain")
        if relation.verdict == "inconsistent":
            reasons.add("additional_relation_inconsistent")
        if relation.outlier_line_ids:
            reasons.add("additional_relation_outliers_recorded")
    return sorted(reasons)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit(
    blind_root: Path,
    annotations_root: Path,
    template_hashes_path: Path,
    report_path: Path,
    protocol_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Close annotation identity, schema, blindness, and template preservation."""

    manifest = _read_jsonl(blind_root / "review_manifest.jsonl")
    expected_ids = [str(row["reviewer_id"]) for row in manifest]
    expected_set = set(expected_ids)
    if len(expected_set) != len(expected_ids):
        raise ValueError("blind manifest reviewer IDs must be unique")
    # PowerShell 5.1 may emit a UTF-8 BOM when the frozen hash closure is
    # generated.  Accepting that marker does not weaken the byte-level hashes
    # of the human templates themselves.
    template_hashes = json.loads(template_hashes_path.read_text(encoding="utf-8-sig"))
    if set(template_hashes) != expected_set:
        raise ValueError("template hash closure must match the blind manifest")
    registration_errors: list[str] = []
    protocol_sha256: str | None = None
    protocol_canonical_sha256: str | None = None
    expected_subdirectory_by_id: dict[str, str] = {}
    if protocol_path is not None:
        protocol_sha256 = _sha256(protocol_path)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
        protocol_canonical_sha256 = _canonical_sha256(protocol)
        registered_input = protocol.get("input", {})
        registered_output = protocol.get("output", {})
        if registered_input.get("review_manifest_sha256") != _sha256(
            blind_root / "review_manifest.jsonl"
        ):
            registration_errors.append("blind manifest hash differs from protocol")
        if registered_input.get(
            "original_annotation_template_hash_closure_sha256"
        ) != _sha256(template_hashes_path):
            registration_errors.append(
                "human template hash closure differs from protocol"
            )
        if registered_input.get("packet_count") != len(expected_ids):
            registration_errors.append("packet count differs from protocol")
        registered_root = str(registered_output.get("root", "")).strip("/")
        resolved_annotations = annotations_root.resolve().as_posix().rstrip("/")
        if not registered_root or not resolved_annotations.endswith(
            f"/{registered_root}"
        ):
            registration_errors.append("annotation root differs from protocol")
        if protocol.get("origin_scoring_authorized") is not False:
            registration_errors.append("protocol does not forbid origin scoring")
        roles = protocol.get("roles", [])
        for role in roles:
            positions = role.get("manifest_positions", [])
            output_subdirectory = str(role.get("output_subdirectory", ""))
            if (
                not isinstance(positions, list)
                or len(positions) != 2
                or not all(isinstance(value, int) for value in positions)
                or positions[0] < 1
                or positions[0] > positions[1]
                or positions[1] > len(expected_ids)
            ):
                registration_errors.append("protocol role positions are invalid")
                continue
            if (
                not output_subdirectory
                or Path(output_subdirectory).name != output_subdirectory
                or output_subdirectory in {".", ".."}
            ):
                registration_errors.append("protocol role output directory is invalid")
                continue
            for reviewer_id in expected_ids[positions[0] - 1 : positions[1]]:
                if reviewer_id in expected_subdirectory_by_id:
                    registration_errors.append(
                        f"protocol roles overlap at {reviewer_id}"
                    )
                expected_subdirectory_by_id[reviewer_id] = output_subdirectory
        if roles and set(expected_subdirectory_by_id) != expected_set:
            registration_errors.append("protocol roles do not cover the manifest")

    annotation_paths = sorted(
        path
        for path in annotations_root.rglob("*.json")
        if path.stem in expected_set
    )
    annotation_file_closure = [
        {
            "reviewer_id": path.stem,
            "relative_path": path.relative_to(annotations_root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in annotation_paths
    ]
    annotation_path_counts = Counter(path.stem for path in annotation_paths)
    duplicate_reviewer_ids = sorted(
        reviewer_id
        for reviewer_id, count in annotation_path_counts.items()
        if count > 1
    )
    for annotation_path in annotation_paths:
        expected_subdirectory = expected_subdirectory_by_id.get(annotation_path.stem)
        if expected_subdirectory is None:
            continue
        relative_parts = annotation_path.relative_to(annotations_root).parts
        if len(relative_parts) < 2 or relative_parts[0] != expected_subdirectory:
            registration_errors.append(
                f"annotation {annotation_path.stem} is outside its registered role directory"
            )
    by_id: dict[str, Path] = {}
    errors: list[str] = []
    status_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    surface_count = 0
    priority_human_review: list[dict[str, Any]] = []
    semantic_annotation_closure: list[dict[str, str]] = []
    for annotation_path in annotation_paths:
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
            forbidden = sorted(set(_walk_keys(payload)) & FORBIDDEN_KEYS)
            if forbidden:
                raise ValueError(f"forbidden fields: {forbidden}")
            forbidden_values = sorted(
                {
                    match.group(0)
                    for value in _walk_strings(payload)
                    for pattern in FORBIDDEN_VALUE_PATTERNS
                    for match in pattern.finditer(value)
                }
            )
            if forbidden_values:
                raise ValueError(
                    f"forbidden source-label text: {forbidden_values}"
                )
            annotation = GeometryRelationAnnotation.model_validate(payload)
            if annotation.reviewer_id != annotation_path.stem:
                raise ValueError("annotation reviewer ID must match its filename")
            if annotation.reviewer_id in by_id:
                raise ValueError("duplicate reviewer annotation")
            if annotation.reviewer_id not in expected_set:
                raise ValueError("reviewer ID is not in the blind manifest")
            row = manifest[expected_ids.index(annotation.reviewer_id)]
            packet = GeometryRelationReviewPacket.model_validate_json(
                (blind_root / str(row["packet"])).read_text(encoding="utf-8-sig")
            )
            validate_annotation_semantic_closure(packet, annotation)
            by_id[annotation.reviewer_id] = annotation_path
            semantic_annotation_closure.append(
                {
                    "reviewer_id": annotation.reviewer_id,
                    "canonical_sha256": _canonical_sha256(
                        annotation.model_dump(mode="json")
                    ),
                }
            )
            status_counts[annotation.status] += 1
            surface_count += len(annotation.surfaces)
            verdict_counts.update(
                review.verdict for review in annotation.proposed_family_reviews
            )
            reasons = _human_review_reasons(annotation)
            if reasons:
                priority_human_review.append(
                    {"reviewer_id": annotation.reviewer_id, "reasons": reasons}
                )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{annotation_path.as_posix()}: {error}")

    template_drift: list[str] = []
    for row in manifest:
        reviewer_id = str(row["reviewer_id"])
        actual = _sha256(blind_root / str(row["annotation"]))
        if actual != str(template_hashes[reviewer_id]):
            template_drift.append(reviewer_id)
    missing = sorted(expected_set - set(by_id))
    extra = sorted(set(by_id) - expected_set)
    registration_errors = sorted(set(registration_errors))
    gates = {
        "all_36_annotations_present_once": (
            not missing
            and not extra
            and not duplicate_reviewer_ids
            and len(by_id) == 36
        ),
        "all_annotations_contract_valid": not errors,
        "all_annotations_frozen": status_counts.get("pending", 0) == 0 and len(by_id) == 36,
        "human_templates_unchanged": not template_drift,
        "source_key_remained_closed": True,
        "registered_input_closure_matches": not registration_errors,
    }
    passed = all(gates.values())
    report = {
        "schema_version": "geometry-semantic-relation-agent-annotation-audit-v1",
        "status": "complete" if passed else "incomplete_or_invalid",
        "annotation_semantics": "AI-assisted blind preannotation; not human ground truth",
        "expected_packet_count": len(expected_ids),
        "validated_annotation_count": len(by_id),
        "blind_manifest_sha256": _sha256(blind_root / "review_manifest.jsonl"),
        "human_template_hash_closure_sha256": _sha256(template_hashes_path),
        "agent_annotation_closure_sha256": _canonical_sha256(
            annotation_file_closure
        ),
        "semantic_annotation_closure_sha256": _canonical_sha256(
            sorted(
                semantic_annotation_closure,
                key=lambda item: item["reviewer_id"],
            )
        ),
        "protocol_sha256": protocol_sha256,
        "protocol_canonical_sha256": protocol_canonical_sha256,
        "protocol_verified": protocol_path is not None,
        "registration_errors": registration_errors,
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "surface_count": surface_count,
        "missing_reviewer_ids": missing,
        "extra_reviewer_ids": extra,
        "duplicate_reviewer_ids": duplicate_reviewer_ids,
        "template_drift_reviewer_ids": template_drift,
        "validation_errors": errors,
        "priority_human_review": sorted(
            priority_human_review, key=lambda item: item["reviewer_id"]
        ),
        "human_review_required_for_ground_truth": True,
        "source_key_opened": False,
        "gates": gates,
        "passed": passed,
        "decision": (
            "eligible_for_relation_graph_proposal_and_later_human_review"
            if passed
            else "continue_source_blind_agent_annotation"
        ),
        "origin_scoring_authorized": False,
    }
    _write_json(report_path, report)
    return report, 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--template-hashes", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, exit_code = audit(
        args.blind_root,
        args.annotations_root,
        args.template_hashes,
        args.report,
        args.protocol,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "validated_annotation_count": report["validated_annotation_count"],
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
