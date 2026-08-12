"""Audit completed semantic-surface review packets and hidden duplicates.

The source key is not opened until every blind annotation is frozen as either
``completed`` or ``unassessable``.  Hidden duplicates are used only for review
quality; source summaries use the non-duplicate occurrence exactly once.
Passing this pilot permits a larger blinded annotation round, never runtime
origin scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
    surface_pair_signature,
    validate_annotation_against_packet,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if count == 0:
        return {"rate": None, "wilson_95_lower": None, "wilson_95_upper": None}
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return {
        "rate": rate,
        "wilson_95_lower": max(0.0, center - margin),
        "wilson_95_upper": min(1.0, center + margin),
    }


def _load_blind_annotations(
    blind_root: Path,
) -> dict[str, tuple[GeometryRelationReviewPacket, GeometryRelationAnnotation]]:
    manifest = _read_jsonl(blind_root / "review_manifest.jsonl")
    loaded: dict[str, tuple[GeometryRelationReviewPacket, GeometryRelationAnnotation]] = {}
    for row in manifest:
        reviewer_id = str(row["reviewer_id"])
        if reviewer_id in loaded:
            raise ValueError("blind manifest reviewer_id values must be unique")
        packet = GeometryRelationReviewPacket.model_validate_json(
            (blind_root / str(row["packet"])).read_text(encoding="utf-8")
        )
        annotation = GeometryRelationAnnotation.model_validate_json(
            (blind_root / str(row["annotation"])).read_text(encoding="utf-8")
        )
        validate_annotation_against_packet(packet, annotation)
        loaded[reviewer_id] = (packet, annotation)
    return loaded


def _family_agreement(
    first: GeometryRelationAnnotation,
    second: GeometryRelationAnnotation,
) -> tuple[int, int]:
    first_verdicts = {
        review.proposed_family_id: review.verdict
        for review in first.proposed_family_reviews
    }
    second_verdicts = {
        review.proposed_family_id: review.verdict
        for review in second.proposed_family_reviews
    }
    common = sorted(set(first_verdicts) & set(second_verdicts))
    return sum(first_verdicts[key] == second_verdicts[key] for key in common), len(common)


def _surface_pair_agreement(
    first: GeometryRelationAnnotation,
    second: GeometryRelationAnnotation,
) -> tuple[int, int]:
    first_pairs = surface_pair_signature(first)
    second_pairs = surface_pair_signature(second)
    common = sorted(set(first_pairs) & set(second_pairs))
    return sum(first_pairs[key] == second_pairs[key] for key in common), len(common)


def _has_geometry_conflict(annotation: GeometryRelationAnnotation) -> bool:
    return any(
        review.verdict == "geometry_inconsistent_within_surface"
        for review in annotation.proposed_family_reviews
    ) or any(
        relation.scope == "within_surface" and relation.verdict == "inconsistent"
        for relation in annotation.additional_relations
    )


def _has_algorithm_overmerge(annotation: GeometryRelationAnnotation) -> bool:
    return any(
        review.verdict == "split_across_surfaces"
        for review in annotation.proposed_family_reviews
    )


def _summarize_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    conflicts = sum(bool(row["geometry_conflict"]) for row in completed)
    overmerges = sum(bool(row["algorithm_overmerge"]) for row in completed)
    return {
        "source_count": len(rows),
        "completed_count": len(completed),
        "unassessable_count": sum(row["status"] == "unassessable" for row in rows),
        "within_surface_geometry_conflict": {
            "count": conflicts,
            **_wilson(conflicts, len(completed)),
        },
        "algorithm_family_overmerge": {
            "count": overmerges,
            **_wilson(overmerges, len(completed)),
        },
    }


def audit(
    blind_root: Path,
    key_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], int]:
    loaded = _load_blind_annotations(blind_root)
    status_counts = Counter(annotation.status for _, annotation in loaded.values())
    if status_counts["pending"]:
        report = {
            "schema_version": "geometry-semantic-relation-pilot-audit-v1",
            "status": "incomplete",
            "packet_count": len(loaded),
            "annotation_status_counts": dict(sorted(status_counts.items())),
            "source_key_opened": False,
            "origin_scoring_authorized": False,
            "decision": "continue_blind_annotation",
        }
        _write_json(report_path, report)
        return report, 2

    key_rows = _read_jsonl(key_path)
    key_by_id = {str(row["reviewer_id"]): row for row in key_rows}
    if len(key_by_id) != len(key_rows) or set(key_by_id) != set(loaded):
        raise ValueError("posthoc key must match blind reviewer IDs exactly")
    by_group: dict[str, list[str]] = defaultdict(list)
    for reviewer_id, row in key_by_id.items():
        by_group[str(row["source_group"])].append(reviewer_id)
    invalid_group_sizes = sorted(
        (group, len(ids)) for group, ids in by_group.items() if len(ids) not in {1, 2}
    )
    if invalid_group_sizes:
        raise ValueError(f"source groups must contain one or two packets: {invalid_group_sizes}")

    duplicate_groups = [ids for ids in by_group.values() if len(ids) == 2]
    completed_duplicate_pairs = 0
    family_matches = family_total = 0
    surface_matches = surface_total = 0
    duplicate_rows: list[dict[str, Any]] = []
    for ids in duplicate_groups:
        first = loaded[ids[0]][1]
        second = loaded[ids[1]][1]
        both_completed = first.status == second.status == "completed"
        if both_completed:
            completed_duplicate_pairs += 1
            family_pair_matches, family_pair_total = _family_agreement(first, second)
            surface_pair_matches, surface_pair_total = _surface_pair_agreement(first, second)
            family_matches += family_pair_matches
            family_total += family_pair_total
            surface_matches += surface_pair_matches
            surface_total += surface_pair_total
        duplicate_rows.append(
            {
                "both_completed": both_completed,
                "family_decision_agreement": (
                    family_pair_matches / family_pair_total
                    if both_completed and family_pair_total
                    else None
                ),
                "surface_line_pair_agreement": (
                    surface_pair_matches / surface_pair_total
                    if both_completed and surface_pair_total
                    else None
                ),
            }
        )

    primary_rows: list[dict[str, Any]] = []
    for reviewer_id, key in key_by_id.items():
        if bool(key["hidden_duplicate"]):
            continue
        annotation = loaded[reviewer_id][1]
        source = dict(key["declared_source_slice"])
        primary_rows.append(
            {
                "status": annotation.status,
                "geometry_conflict": (
                    _has_geometry_conflict(annotation) if annotation.status == "completed" else None
                ),
                "algorithm_overmerge": (
                    _has_algorithm_overmerge(annotation) if annotation.status == "completed" else None
                ),
                "label": str(source["label_name"]),
                "archive": str(source["archive_name"]),
            }
        )

    completed_unique = sum(row["status"] == "completed" for row in primary_rows)
    completed_unique_ratio = completed_unique / len(primary_rows) if primary_rows else 0.0
    family_agreement = family_matches / family_total if family_total else None
    surface_agreement = surface_matches / surface_total if surface_total else None
    gates = {
        "completed_unique_ratio_at_least_0_75": completed_unique_ratio >= 0.75,
        "completed_duplicate_pairs_at_least_3": completed_duplicate_pairs >= 3,
        "family_decision_agreement_at_least_0_80": (
            family_agreement is not None and family_agreement >= 0.80
        ),
        "surface_line_pair_agreement_at_least_0_80": (
            surface_agreement is not None and surface_agreement >= 0.80
        ),
    }
    quality_passed = all(gates.values())
    by_label = {
        label: _summarize_source_rows([row for row in primary_rows if row["label"] == label])
        for label in sorted({str(row["label"]) for row in primary_rows})
    }
    by_archive = {
        archive: _summarize_source_rows([row for row in primary_rows if row["archive"] == archive])
        for archive in sorted({str(row["archive"]) for row in primary_rows})
    }
    report = {
        "schema_version": "geometry-semantic-relation-pilot-audit-v1",
        "status": "complete",
        "packet_count": len(loaded),
        "unique_source_count": len(primary_rows),
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "source_key_opened": True,
        "quality": {
            "completed_unique_count": completed_unique,
            "completed_unique_ratio": completed_unique_ratio,
            "hidden_duplicate_group_count": len(duplicate_groups),
            "completed_duplicate_pair_count": completed_duplicate_pairs,
            "family_decision_agreement": family_agreement,
            "surface_line_pair_agreement": surface_agreement,
            "duplicate_groups": duplicate_rows,
            "gates": gates,
            "passed": quality_passed,
        },
        "posthoc_source_summary": {
            "overall": _summarize_source_rows(primary_rows),
            "by_declared_label": by_label,
            "by_archive": by_archive,
        },
        "decision": (
            "eligible_for_larger_blinded_annotation_only"
            if quality_passed
            else "annotation_quality_gate_failed"
        ),
        "origin_scoring_authorized": False,
        "limitations": [
            "The pilot is too small to select an AI-source threshold or estimate deployment error.",
            "Algorithm overmerge is an annotation-system error, not an image-source signal.",
            "A new source-isolated holdout is required after any relation-graph implementation.",
        ],
    }
    _write_json(report_path, report)
    return report, 0 if quality_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--posthoc-key", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, exit_code = audit(args.blind_root, args.posthoc_key, args.report)
    print(json.dumps({"status": report["status"], "decision": report["decision"]}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
