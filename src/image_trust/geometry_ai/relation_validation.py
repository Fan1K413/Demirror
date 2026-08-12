"""Shared semantic-closure validation for source-blind relation reviews."""

from __future__ import annotations

from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
    validate_annotation_against_packet,
)


def validate_annotation_semantic_closure(
    packet: GeometryRelationReviewPacket,
    annotation: GeometryRelationAnnotation,
) -> None:
    """Require every decided non-outlier member to be explained by a surface.

    The frozen pilot contract continues to check identifiers and references.
    This later graph-stage gate adds the semantic closure needed by downstream
    consumers: every referenced surface contributes an active member, and every
    active member occurs on at least one referenced surface.  A member may occur
    on more than one surface because a visible edge can bound two surfaces.
    """

    validate_annotation_against_packet(packet, annotation)
    surface_lines = {
        surface.surface_id: set(surface.line_ids) for surface in annotation.surfaces
    }
    proposal_by_id = {
        proposal.family_id: proposal for proposal in packet.family_proposals
    }
    for review in annotation.proposed_family_reviews:
        _require_unique(review.outlier_line_ids, "family review outlier_line_ids")
        if review.verdict in {"pending", "unassessable"}:
            continue
        _validate_surface_member_closure(
            subject=f"family {review.proposed_family_id}",
            member_line_ids=proposal_by_id[
                review.proposed_family_id
            ].member_line_ids,
            outlier_line_ids=review.outlier_line_ids,
            surface_ids=review.surface_ids,
            surface_lines=surface_lines,
        )
    for relation in annotation.additional_relations:
        _require_unique(relation.outlier_line_ids, "relation outlier_line_ids")
        _validate_surface_member_closure(
            subject=f"relation {relation.relation_id}",
            member_line_ids=relation.member_line_ids,
            outlier_line_ids=relation.outlier_line_ids,
            surface_ids=relation.surface_ids,
            surface_lines=surface_lines,
        )


def _validate_surface_member_closure(
    *,
    subject: str,
    member_line_ids: list[str],
    outlier_line_ids: list[str],
    surface_ids: list[str],
    surface_lines: dict[str, set[str]],
) -> None:
    active_members = set(member_line_ids) - set(outlier_line_ids)
    if not active_members:
        raise ValueError(f"{subject} has no non-outlier members")
    explained: set[str] = set()
    for surface_id in surface_ids:
        matched = active_members & surface_lines[surface_id]
        if not matched:
            raise ValueError(
                f"{subject} references surface {surface_id} without a member line"
            )
        explained.update(matched)
    missing = sorted(active_members - explained)
    if missing:
        raise ValueError(
            f"{subject} leaves member lines unexplained by its reviewed surfaces: "
            f"{missing}"
        )


def _require_unique(values: list[str], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
