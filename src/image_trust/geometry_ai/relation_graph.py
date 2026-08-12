"""Source-neutral semantic-surface relation graph and diagnostic exporter.

This module turns a finalized geometry relation review into an explicit graph.
It never receives an image-source label and does not produce an AI score.  The
first implementation is intentionally review-conditioned: automatic surface
inference remains a later, separately evaluated phase.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.geometry_ai.measurement_overlays import COLORS
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
    NormalizedBox,
    NormalizedPoint,
)
from image_trust.geometry_ai.relation_validation import (
    validate_annotation_semantic_closure,
)


GRAPH_SCHEMA_VERSION = "geometry-semantic-relation-graph-v1"
ARTIFACT_SCHEMA_VERSION = "geometry-semantic-relation-graph-artifacts-v1"

SurfaceKind = Literal[
    "facade",
    "roof",
    "road",
    "floor",
    "ceiling",
    "window_or_door_array",
    "fence_or_railing",
    "object_surface",
    "other",
    "uncertain",
]


class _FrozenGraphContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RelationGraphLine(_FrozenGraphContract):
    line_id: str
    start: NormalizedPoint
    end: NormalizedPoint
    length_normalized: float = Field(gt=0.0)
    angle_degrees: float = Field(ge=0.0, lt=180.0)
    cross_scale_stability: float = Field(ge=0.0, le=1.0)
    region_ids: list[str] = Field(default_factory=list)


class RelationGraphRegion(_FrozenGraphContract):
    region_id: str
    box: NormalizedBox
    line_ids: list[str] = Field(default_factory=list)
    status: Literal["usable", "insufficient_support"]


class RelationGraphSurface(_FrozenGraphContract):
    surface_id: str
    surface_kind: SurfaceKind
    polygon_normalized: list[NormalizedPoint] = Field(min_length=3)
    visibility: Literal["clear", "partial", "uncertain"]
    support_line_ids: list[str] = Field(default_factory=list)


class RelationGraphOriginalFamily(_FrozenGraphContract):
    family_id: str
    region_id: str
    kind: Literal["parallel", "finite_vp", "infinite_vp"]
    member_line_ids: list[str] = Field(min_length=2)
    review_verdict: Literal[
        "coherent_within_surface",
        "split_across_surfaces",
        "geometry_inconsistent_within_surface",
        "unassessable",
    ]


class RelationGraphConditionedFamily(_FrozenGraphContract):
    conditioned_family_id: str
    original_family_id: str
    surface_id: str
    kind: Literal["parallel", "finite_vp", "infinite_vp"]
    member_line_ids: list[str] = Field(min_length=1)
    usability: Literal["usable", "insufficient_members"]

    @model_validator(mode="after")
    def _usability_matches_members(self) -> "RelationGraphConditionedFamily":
        expected = "usable" if len(self.member_line_ids) >= 2 else "insufficient_members"
        if self.usability != expected:
            raise ValueError("conditioned-family usability does not match member count")
        return self


class LineOnSurfaceEdge(_FrozenGraphContract):
    line_id: str
    surface_id: str
    role: Literal["surface_support", "shared_boundary_candidate"]


class LinePairSameSurfaceEdge(_FrozenGraphContract):
    first_line_id: str
    second_line_id: str
    shared_surface_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_pair(self) -> "LinePairSameSurfaceEdge":
        if self.first_line_id >= self.second_line_id:
            raise ValueError("line pair IDs must be stored in ascending order")
        return self


class FamilyOnSurfaceEdge(_FrozenGraphContract):
    original_family_id: str
    surface_id: str
    conditioned_family_id: str
    member_line_ids: list[str] = Field(min_length=1)


class SurfaceRelationEdge(_FrozenGraphContract):
    relation_id: str
    relation_type: Literal[
        "parallel_family",
        "common_vanishing_direction",
        "repeated_spacing",
        "edge_continuation",
    ]
    scope: Literal["within_surface", "cross_surface"]
    surface_ids: list[str] = Field(min_length=1)
    member_line_ids: list[str] = Field(min_length=2)
    verdict: Literal["coherent", "inconsistent", "uncertain"]
    outlier_line_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _scope_matches_surfaces(self) -> "SurfaceRelationEdge":
        if self.scope == "within_surface" and len(self.surface_ids) != 1:
            raise ValueError("within-surface relation requires exactly one surface")
        if self.scope == "cross_surface" and len(self.surface_ids) < 2:
            raise ValueError("cross-surface relation requires at least two surfaces")
        return self


class FamilyMemberExclusion(_FrozenGraphContract):
    original_family_id: str
    line_id: str
    reason: Literal["explicit_outlier", "review_unassessable"]


class GeometryRelationGraph(_FrozenGraphContract):
    """Closed relationship graph; deliberately contains no source score."""

    schema_version: Literal[GRAPH_SCHEMA_VERSION] = GRAPH_SCHEMA_VERSION
    construction_mode: Literal["source_blind_review_annotation"] = (
        "source_blind_review_annotation"
    )
    source_label_visibility: Literal["forbidden"] = "forbidden"
    source_labels_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False
    annotation_semantics: Literal["relation_review_not_origin_ground_truth"] = (
        "relation_review_not_origin_ground_truth"
    )
    reviewer_id: str
    annotation_status: Literal["completed", "unassessable"]
    canonical_size: tuple[int, int]
    lines: list[RelationGraphLine] = Field(default_factory=list)
    regions: list[RelationGraphRegion] = Field(default_factory=list)
    surfaces: list[RelationGraphSurface] = Field(default_factory=list)
    original_families: list[RelationGraphOriginalFamily] = Field(default_factory=list)
    surface_conditioned_families: list[RelationGraphConditionedFamily] = Field(
        default_factory=list
    )
    line_on_surface: list[LineOnSurfaceEdge] = Field(default_factory=list)
    line_pair_same_surface: list[LinePairSameSurfaceEdge] = Field(default_factory=list)
    family_on_surface: list[FamilyOnSurfaceEdge] = Field(default_factory=list)
    surface_relations: list[SurfaceRelationEdge] = Field(default_factory=list)
    excluded_family_members: list[FamilyMemberExclusion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_are_closed(self) -> "GeometryRelationGraph":
        if self.canonical_size[0] <= 0 or self.canonical_size[1] <= 0:
            raise ValueError("relation graph canonical size must be positive")
        line_by_id = _index_unique(self.lines, "line_id", "line")
        region_by_id = _index_unique(self.regions, "region_id", "region")
        surface_by_id = _index_unique(self.surfaces, "surface_id", "surface")
        family_by_id = _index_unique(
            self.original_families, "family_id", "original family"
        )
        conditioned_by_id = _index_unique(
            self.surface_conditioned_families,
            "conditioned_family_id",
            "conditioned family",
        )

        line_ids = set(line_by_id)
        expected_regions_by_line: dict[str, set[str]] = {
            line_id: set() for line_id in line_ids
        }
        for region in self.regions:
            _require_graph_subset(
                region.line_ids, line_ids, f"region {region.region_id} lines"
            )
            for line_id in region.line_ids:
                expected_regions_by_line[line_id].add(region.region_id)
        for line in self.lines:
            _require_graph_subset(
                line.region_ids, set(region_by_id), f"line {line.line_id} regions"
            )
            if set(line.region_ids) != expected_regions_by_line[line.line_id]:
                raise ValueError("line region memberships do not match region nodes")
        for surface in self.surfaces:
            _require_graph_subset(
                surface.support_line_ids, line_ids, f"surface {surface.surface_id} lines"
            )
        for family in self.original_families:
            _require_graph_subset(
                family.member_line_ids, line_ids, f"family {family.family_id} lines"
            )
            if family.region_id != "global" and family.region_id not in region_by_id:
                raise ValueError(
                    f"family {family.family_id} references an unknown region"
                )

        for conditioned in self.surface_conditioned_families:
            original = family_by_id.get(conditioned.original_family_id)
            surface = surface_by_id.get(conditioned.surface_id)
            if original is None or surface is None:
                raise ValueError("conditioned family references an unknown node")
            if conditioned.kind != original.kind:
                raise ValueError("conditioned family kind differs from original family")
            if conditioned.conditioned_family_id != _conditioned_family_id(
                conditioned.original_family_id,
                conditioned.surface_id,
            ):
                raise ValueError("conditioned family ID is not deterministic")
            _require_graph_subset(
                conditioned.member_line_ids,
                set(original.member_line_ids),
                f"conditioned family {conditioned.conditioned_family_id} original members",
            )
            _require_graph_subset(
                conditioned.member_line_ids,
                set(surface.support_line_ids),
                f"conditioned family {conditioned.conditioned_family_id} surface members",
            )

        expected_line_edges = {
            (line_id, surface.surface_id)
            for surface in self.surfaces
            for line_id in surface.support_line_ids
        }
        actual_line_edges = {
            (edge.line_id, edge.surface_id): edge.role for edge in self.line_on_surface
        }
        if len(actual_line_edges) != len(self.line_on_surface):
            raise ValueError("line-on-surface edges must be unique")
        if set(actual_line_edges) != expected_line_edges:
            raise ValueError("line-on-surface edges do not close over surface memberships")
        membership_degree: dict[str, int] = {line_id: 0 for line_id in line_ids}
        for line_id, _ in expected_line_edges:
            membership_degree[line_id] += 1
        for (line_id, _), role in actual_line_edges.items():
            expected_role = (
                "shared_boundary_candidate"
                if membership_degree[line_id] > 1
                else "surface_support"
            )
            if role != expected_role:
                raise ValueError(
                    "line-on-surface role does not match membership degree"
                )

        expected_pairs = _expected_surface_pairs(self.surfaces)
        actual_pairs = {
            (edge.first_line_id, edge.second_line_id): tuple(edge.shared_surface_ids)
            for edge in self.line_pair_same_surface
        }
        if len(actual_pairs) != len(self.line_pair_same_surface):
            raise ValueError("same-surface line-pair edges must be unique")
        if actual_pairs != expected_pairs:
            raise ValueError("same-surface line-pair edges do not match surface memberships")

        expected_family_edges = {
            conditioned.conditioned_family_id: (
                conditioned.original_family_id,
                conditioned.surface_id,
                tuple(conditioned.member_line_ids),
            )
            for conditioned in self.surface_conditioned_families
        }
        actual_family_edges = {
            edge.conditioned_family_id: (
                edge.original_family_id,
                edge.surface_id,
                tuple(edge.member_line_ids),
            )
            for edge in self.family_on_surface
        }
        if len(actual_family_edges) != len(self.family_on_surface):
            raise ValueError("family-on-surface edges must be unique")
        if actual_family_edges != expected_family_edges:
            raise ValueError("family-on-surface edges do not match conditioned families")

        assigned_by_family: dict[str, set[str]] = {
            family_id: set() for family_id in family_by_id
        }
        for conditioned in conditioned_by_id.values():
            assigned_by_family[conditioned.original_family_id].update(
                conditioned.member_line_ids
            )
        excluded_by_family: dict[str, set[str]] = {
            family_id: set() for family_id in family_by_id
        }
        exclusion_keys: set[tuple[str, str]] = set()
        for exclusion in self.excluded_family_members:
            family = family_by_id.get(exclusion.original_family_id)
            if family is None or exclusion.line_id not in family.member_line_ids:
                raise ValueError("family-member exclusion references an unknown member")
            key = (exclusion.original_family_id, exclusion.line_id)
            if key in exclusion_keys:
                raise ValueError("family-member exclusions must be unique")
            exclusion_keys.add(key)
            excluded_by_family[exclusion.original_family_id].add(exclusion.line_id)
        for family_id, family in family_by_id.items():
            assigned = assigned_by_family[family_id]
            excluded = excluded_by_family[family_id]
            if assigned & excluded:
                raise ValueError("a family member cannot be both assigned and excluded")
            if assigned | excluded != set(family.member_line_ids):
                raise ValueError(f"family {family_id} member closure is incomplete")

        relation_ids: set[str] = set()
        for relation in self.surface_relations:
            if relation.relation_id in relation_ids:
                raise ValueError("surface relation IDs must be unique")
            relation_ids.add(relation.relation_id)
            _require_graph_subset(
                relation.surface_ids, set(surface_by_id), "surface relation surfaces"
            )
            _require_graph_subset(
                relation.member_line_ids, line_ids, "surface relation lines"
            )
            _require_graph_subset(
                relation.outlier_line_ids,
                set(relation.member_line_ids),
                "surface relation outliers",
            )
        return self


class RelationGraphArtifactManifest(_FrozenGraphContract):
    schema_version: Literal[ARTIFACT_SCHEMA_VERSION] = ARTIFACT_SCHEMA_VERSION
    graph_json: str
    surface_membership_overlay: str
    family_comparison_overlay: str
    line_count: int = Field(ge=0)
    region_count: int = Field(ge=0)
    surface_count: int = Field(ge=0)
    original_family_count: int = Field(ge=0)
    conditioned_family_count: int = Field(ge=0)
    excluded_member_count: int = Field(ge=0)
    source_labels_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False


def build_relation_graph(
    packet: GeometryRelationReviewPacket,
    annotation: GeometryRelationAnnotation,
) -> GeometryRelationGraph:
    """Build a closed, source-neutral graph from a finalized review."""

    if annotation.status == "pending" or any(
        review.verdict == "pending" for review in annotation.proposed_family_reviews
    ):
        raise ValueError("relation graph requires a finalized annotation")
    validate_annotation_semantic_closure(packet, annotation)

    region_ids_by_line: dict[str, list[str]] = {line.line_id: [] for line in packet.lines}
    for region in packet.region_proposals:
        for line_id in region.line_ids:
            if line_id in region_ids_by_line:
                region_ids_by_line[line_id].append(region.region_id)
    lines = [
        RelationGraphLine(
            line_id=line.line_id,
            start=line.start,
            end=line.end,
            length_normalized=math.hypot(
                line.end.x - line.start.x, line.end.y - line.start.y
            ),
            angle_degrees=(
                math.degrees(
                    math.atan2(
                        line.end.y - line.start.y,
                        line.end.x - line.start.x,
                    )
                )
                % 180.0
            ),
            cross_scale_stability=line.cross_scale_stability,
            region_ids=region_ids_by_line[line.line_id],
        )
        for line in packet.lines
    ]
    regions = [
        RelationGraphRegion(
            region_id=region.region_id,
            box=region.box,
            line_ids=region.line_ids,
            status=region.status,
        )
        for region in packet.region_proposals
    ]
    surfaces = [
        RelationGraphSurface(
            surface_id=surface.surface_id,
            surface_kind=surface.surface_kind,
            polygon_normalized=surface.polygon_normalized,
            visibility=surface.visibility,
            support_line_ids=surface.line_ids,
        )
        for surface in annotation.surfaces
    ]
    surface_lines = {
        surface.surface_id: set(surface.support_line_ids) for surface in surfaces
    }
    membership_count: dict[str, int] = {line.line_id: 0 for line in lines}
    for surface in surfaces:
        for line_id in surface.support_line_ids:
            membership_count[line_id] += 1
    line_edges = [
        LineOnSurfaceEdge(
            line_id=line_id,
            surface_id=surface.surface_id,
            role=(
                "shared_boundary_candidate"
                if membership_count[line_id] > 1
                else "surface_support"
            ),
        )
        for surface in surfaces
        for line_id in surface.support_line_ids
    ]
    pair_edges = [
        LinePairSameSurfaceEdge(
            first_line_id=first,
            second_line_id=second,
            shared_surface_ids=list(shared_surfaces),
        )
        for (first, second), shared_surfaces in _expected_surface_pairs(surfaces).items()
    ]

    review_by_id = {
        review.proposed_family_id: review
        for review in annotation.proposed_family_reviews
    }
    original_families: list[RelationGraphOriginalFamily] = []
    conditioned_families: list[RelationGraphConditionedFamily] = []
    family_edges: list[FamilyOnSurfaceEdge] = []
    exclusions: list[FamilyMemberExclusion] = []
    for proposal in packet.family_proposals:
        review = review_by_id[proposal.family_id]
        original_families.append(
            RelationGraphOriginalFamily(
                family_id=proposal.family_id,
                region_id=proposal.region_id,
                kind=proposal.kind,
                member_line_ids=proposal.member_line_ids,
                review_verdict=review.verdict,
            )
        )
        outliers = set(review.outlier_line_ids)
        for line_id in proposal.member_line_ids:
            if line_id in outliers:
                exclusions.append(
                    FamilyMemberExclusion(
                        original_family_id=proposal.family_id,
                        line_id=line_id,
                        reason="explicit_outlier",
                    )
                )
        if review.verdict == "unassessable":
            for line_id in proposal.member_line_ids:
                if line_id not in outliers:
                    exclusions.append(
                        FamilyMemberExclusion(
                            original_family_id=proposal.family_id,
                            line_id=line_id,
                            reason="review_unassessable",
                        )
                    )
            continue
        for surface_id in review.surface_ids:
            members = [
                line_id
                for line_id in proposal.member_line_ids
                if line_id not in outliers and line_id in surface_lines[surface_id]
            ]
            conditioned_id = _conditioned_family_id(proposal.family_id, surface_id)
            conditioned = RelationGraphConditionedFamily(
                conditioned_family_id=conditioned_id,
                original_family_id=proposal.family_id,
                surface_id=surface_id,
                kind=proposal.kind,
                member_line_ids=members,
                usability="usable" if len(members) >= 2 else "insufficient_members",
            )
            conditioned_families.append(conditioned)
            family_edges.append(
                FamilyOnSurfaceEdge(
                    original_family_id=proposal.family_id,
                    surface_id=surface_id,
                    conditioned_family_id=conditioned_id,
                    member_line_ids=members,
                )
            )

    relations = [
        SurfaceRelationEdge(
            relation_id=relation.relation_id,
            relation_type=relation.relation_type,
            scope=relation.scope,
            surface_ids=relation.surface_ids,
            member_line_ids=relation.member_line_ids,
            verdict=relation.verdict,
            outlier_line_ids=relation.outlier_line_ids,
        )
        for relation in annotation.additional_relations
    ]
    return GeometryRelationGraph(
        reviewer_id=packet.reviewer_id,
        annotation_status=annotation.status,
        canonical_size=packet.canonical_size,
        lines=lines,
        regions=regions,
        surfaces=surfaces,
        original_families=original_families,
        surface_conditioned_families=conditioned_families,
        line_on_surface=line_edges,
        line_pair_same_surface=pair_edges,
        family_on_surface=family_edges,
        surface_relations=relations,
        excluded_family_members=exclusions,
    )


def export_relation_graph_diagnostics(
    image: Image.Image,
    packet: GeometryRelationReviewPacket,
    annotation: GeometryRelationAnnotation,
    output_dir: Path,
) -> tuple[GeometryRelationGraph, RelationGraphArtifactManifest]:
    """Write a graph JSON and two diagnostic overlays atomically."""

    graph = build_relation_graph(packet, annotation)
    rgb = image.convert("RGB")
    if rgb.size != graph.canonical_size:
        raise ValueError(
            f"diagnostic image size {rgb.size} does not match graph "
            f"canonical size {graph.canonical_size}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_name = "geometry_relation_graph.json"
    surface_name = "surface_membership_overlay.png"
    comparison_name = "original_vs_surface_conditioned_families.png"
    manifest_name = "geometry_relation_graph_artifacts.json"

    _atomic_write_json(output_dir / graph_name, graph.model_dump(mode="json"))
    _atomic_write_image(
        output_dir / surface_name,
        render_surface_membership_overlay(rgb, graph),
    )
    _atomic_write_image(
        output_dir / comparison_name,
        render_family_comparison_overlay(rgb, graph),
    )
    manifest = RelationGraphArtifactManifest(
        graph_json=graph_name,
        surface_membership_overlay=surface_name,
        family_comparison_overlay=comparison_name,
        line_count=len(graph.lines),
        region_count=len(graph.regions),
        surface_count=len(graph.surfaces),
        original_family_count=len(graph.original_families),
        conditioned_family_count=len(graph.surface_conditioned_families),
        excluded_member_count=len(graph.excluded_family_members),
    )
    _atomic_write_json(output_dir / manifest_name, manifest.model_dump(mode="json"))
    return graph, manifest


def render_surface_membership_overlay(
    image: Image.Image,
    graph: GeometryRelationGraph,
) -> Image.Image:
    """Render reviewed surface polygons and their supporting line memberships."""

    canvas = image.convert("RGBA")
    translucent = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(translucent)
    line_by_id = {line.line_id: line for line in graph.lines}
    color_by_surface = {
        surface.surface_id: COLORS[index % len(COLORS)]
        for index, surface in enumerate(graph.surfaces)
    }
    for index, surface in enumerate(graph.surfaces):
        color = color_by_surface[surface.surface_id]
        polygon = [_pixel(point, canvas.size) for point in surface.polygon_normalized]
        fill_draw.polygon(polygon, fill=(*color, 42), outline=(*color, 220), width=3)
        if polygon:
            fill_draw.text(
                polygon[0],
                f"S{index + 1}",
                fill=(*color, 255),
                stroke_width=2,
                stroke_fill=(255, 255, 255, 240),
            )
    canvas = Image.alpha_composite(canvas, translucent)
    draw = ImageDraw.Draw(canvas)
    membership_by_line: dict[str, list[str]] = {}
    for edge in graph.line_on_surface:
        membership_by_line.setdefault(edge.line_id, []).append(edge.surface_id)
    for line_id, surface_ids in membership_by_line.items():
        line = line_by_id[line_id]
        points = (*_pixel(line.start, canvas.size), *_pixel(line.end, canvas.size))
        if len(surface_ids) > 1:
            draw.line(points, fill=(255, 255, 255, 255), width=6)
            draw.line(points, fill=(30, 30, 30, 255), width=3)
        else:
            draw.line(points, fill=(*color_by_surface[surface_ids[0]], 255), width=4)
    _banner(draw, canvas.width, "SURFACE MEMBERSHIP - RELATION REVIEW ONLY")
    return canvas.convert("RGB")


def render_family_comparison_overlay(
    image: Image.Image,
    graph: GeometryRelationGraph,
) -> Image.Image:
    """Render original direction families beside surface-conditioned subfamilies."""

    left = image.convert("RGB")
    right = image.convert("RGB")
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    line_by_id = {line.line_id: line for line in graph.lines}
    for index, family in enumerate(graph.original_families):
        color = COLORS[index % len(COLORS)]
        for line_id in family.member_line_ids:
            _draw_graph_line(left_draw, line_by_id[line_id], left.size, color, width=4)
    for index, family in enumerate(graph.surface_conditioned_families):
        color = COLORS[index % len(COLORS)]
        for line_id in family.member_line_ids:
            _draw_graph_line(right_draw, line_by_id[line_id], right.size, color, width=4)
    for exclusion in graph.excluded_family_members:
        color = (205, 55, 55) if exclusion.reason == "explicit_outlier" else (110, 110, 110)
        _draw_dashed_graph_line(
            right_draw,
            line_by_id[exclusion.line_id],
            right.size,
            color,
            width=3,
        )
    _banner(left_draw, left.width, "ORIGINAL DIRECTION FAMILIES")
    _banner(right_draw, right.width, "SURFACE-CONDITIONED SUBFAMILIES")
    combined = Image.new("RGB", (left.width * 2 + 2, left.height), "white")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + 2, 0))
    divider = ImageDraw.Draw(combined)
    divider.rectangle((left.width, 0, left.width + 1, left.height), fill=(20, 20, 20))
    return combined


def _conditioned_family_id(original_family_id: str, surface_id: str) -> str:
    payload = json.dumps(
        [original_family_id, surface_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"scf-{hashlib.sha256(payload).hexdigest()[:16]}"


def _expected_surface_pairs(
    surfaces: list[RelationGraphSurface],
) -> dict[tuple[str, str], tuple[str, ...]]:
    shared: dict[tuple[str, str], list[str]] = {}
    for surface in surfaces:
        for first, second in combinations(sorted(surface.support_line_ids), 2):
            shared.setdefault((first, second), []).append(surface.surface_id)
    return {
        pair: tuple(sorted(surface_ids)) for pair, surface_ids in sorted(shared.items())
    }


def _index_unique(items: list[BaseModel], attribute: str, label: str) -> dict[str, BaseModel]:
    indexed: dict[str, BaseModel] = {}
    for item in items:
        identifier = str(getattr(item, attribute))
        if identifier in indexed:
            raise ValueError(f"{label} IDs must be unique")
        indexed[identifier] = item
    return indexed


def _require_graph_subset(values: list[str], allowed: set[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{label} contain unknown values: {unknown}")


def _pixel(point: NormalizedPoint, size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    return (
        round(point.x * max(0, width - 1)),
        round(point.y * max(0, height - 1)),
    )


def _draw_graph_line(
    draw: ImageDraw.ImageDraw,
    line: RelationGraphLine,
    size: tuple[int, int],
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    draw.line((*_pixel(line.start, size), *_pixel(line.end, size)), fill=color, width=width)


def _draw_dashed_graph_line(
    draw: ImageDraw.ImageDraw,
    line: RelationGraphLine,
    size: tuple[int, int],
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    start = _pixel(line.start, size)
    end = _pixel(line.end, size)
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    if distance <= 0.0:
        return
    dash = 7.0
    step_count = max(1, math.ceil(distance / dash))
    for index in range(0, step_count, 2):
        first_t = index / step_count
        second_t = min(1.0, (index + 1) / step_count)
        segment = (
            round(start[0] + (end[0] - start[0]) * first_t),
            round(start[1] + (end[1] - start[1]) * first_t),
            round(start[0] + (end[0] - start[0]) * second_t),
            round(start[1] + (end[1] - start[1]) * second_t),
        )
        draw.line(segment, fill=color, width=width)


def _banner(draw: ImageDraw.ImageDraw, width: int, text: str) -> None:
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_image(path: Path, image: Image.Image) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)
