"""Source-blind contracts for semantic surface and geometry-relation review.

The geometry-v2 measurements deliberately stop at image-space regions and
line families.  This module defines the human-review layer needed to decide
whether those lines belong to the same visible surface or merely share a
similar image direction.  It contains no source label and produces no AI
score.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.geometry_ai.measurement_overlays import COLORS
from image_trust.geometry_ai.measurement_types import GeometryMeasurementV2Result


PACKET_SCHEMA_VERSION = "geometry-semantic-relation-review-packet-v1"
ANNOTATION_SCHEMA_VERSION = "geometry-semantic-relation-annotation-v1"
MAX_REVIEW_FAMILIES_PER_SCOPE = 4


class _FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NormalizedPoint(_FrozenContract):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class NormalizedBox(_FrozenContract):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _inside_image(self) -> "NormalizedBox":
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("normalized box must remain inside the image")
        return self


class ReviewAssets(_FrozenContract):
    image: str
    line_ids_overlay: str
    regions_overlay: str
    local_families_overlay: str
    global_families_overlay: str
    consistency_overlay: str
    repeat_spacing_overlay: str
    measurement: str


class ReviewLine(_FrozenContract):
    line_id: str
    start: NormalizedPoint
    end: NormalizedPoint
    cross_scale_stability: float = Field(ge=0.0, le=1.0)


class ReviewRegionProposal(_FrozenContract):
    region_id: str
    box: NormalizedBox
    line_ids: list[str] = Field(default_factory=list)
    status: Literal["usable", "insufficient_support"]


class ReviewFamilyProposal(_FrozenContract):
    family_id: str
    region_id: str
    kind: Literal["parallel", "finite_vp", "infinite_vp"]
    member_line_ids: list[str] = Field(min_length=2)
    overlay_asset: Literal["local_families_overlay", "global_families_overlay"]
    detail_overlay: str
    overlay_color_rgb: tuple[int, int, int]
    priority_reason: Literal["check_finding", "stable_control"]
    maximum_finding_severity: float = Field(ge=0.0, le=1.0)


class SemanticSurface(_FrozenContract):
    """One visible plane or object surface, independent of source identity."""

    surface_id: str
    surface_kind: Literal[
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
    polygon_normalized: list[NormalizedPoint] = Field(min_length=3)
    line_ids: list[str] = Field(default_factory=list)
    visibility: Literal["clear", "partial", "uncertain"]
    note: str = ""

    @model_validator(mode="after")
    def _unique_lines(self) -> "SemanticSurface":
        if len(self.line_ids) != len(set(self.line_ids)):
            raise ValueError("surface line_ids must be unique")
        return self


class ProposedFamilyReview(_FrozenContract):
    proposed_family_id: str
    verdict: Literal[
        "pending",
        "coherent_within_surface",
        "split_across_surfaces",
        "geometry_inconsistent_within_surface",
        "unassessable",
    ] = "pending"
    surface_ids: list[str] = Field(default_factory=list)
    outlier_line_ids: list[str] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _surface_count_matches_verdict(self) -> "ProposedFamilyReview":
        if len(self.surface_ids) != len(set(self.surface_ids)):
            raise ValueError("family review surface_ids must be unique")
        if self.verdict in {
            "coherent_within_surface",
            "geometry_inconsistent_within_surface",
        } and len(self.surface_ids) != 1:
            raise ValueError("within-surface verdicts require exactly one surface")
        if self.verdict == "split_across_surfaces" and len(self.surface_ids) < 2:
            raise ValueError("split_across_surfaces requires at least two surfaces")
        if self.verdict == "pending" and (self.surface_ids or self.outlier_line_ids):
            raise ValueError("pending family reviews must not contain decisions")
        return self


class AdditionalSurfaceRelation(_FrozenContract):
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
    note: str = ""

    @model_validator(mode="after")
    def _scope_matches_surfaces(self) -> "AdditionalSurfaceRelation":
        if len(self.surface_ids) != len(set(self.surface_ids)):
            raise ValueError("relation surface_ids must be unique")
        if len(self.member_line_ids) != len(set(self.member_line_ids)):
            raise ValueError("relation member_line_ids must be unique")
        if self.scope == "within_surface" and len(self.surface_ids) != 1:
            raise ValueError("within_surface relations require exactly one surface")
        if self.scope == "cross_surface" and len(self.surface_ids) < 2:
            raise ValueError("cross_surface relations require at least two surfaces")
        return self


class GeometryRelationAnnotation(_FrozenContract):
    schema_version: Literal[ANNOTATION_SCHEMA_VERSION] = ANNOTATION_SCHEMA_VERSION
    reviewer_id: str
    status: Literal["pending", "completed", "unassessable"] = "pending"
    assessability_reason: str = ""
    surfaces: list[SemanticSurface] = Field(default_factory=list)
    proposed_family_reviews: list[ProposedFamilyReview] = Field(default_factory=list)
    additional_relations: list[AdditionalSurfaceRelation] = Field(default_factory=list)
    review_note: str = ""

    @model_validator(mode="after")
    def _valid_completion_state(self) -> "GeometryRelationAnnotation":
        surface_ids = [surface.surface_id for surface in self.surfaces]
        relation_ids = [relation.relation_id for relation in self.additional_relations]
        reviewed_ids = [review.proposed_family_id for review in self.proposed_family_reviews]
        if len(surface_ids) != len(set(surface_ids)):
            raise ValueError("surface_id values must be unique")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation_id values must be unique")
        if len(reviewed_ids) != len(set(reviewed_ids)):
            raise ValueError("proposed_family_id values must be unique")
        if self.status == "completed":
            if not self.surfaces:
                raise ValueError("completed annotations require at least one semantic surface")
            if any(review.verdict == "pending" for review in self.proposed_family_reviews):
                raise ValueError("completed annotations cannot contain pending family reviews")
        if self.status == "unassessable" and not self.assessability_reason.strip():
            raise ValueError("unassessable annotations require a reason")
        return self


class GeometryRelationReviewPacket(_FrozenContract):
    schema_version: Literal[PACKET_SCHEMA_VERSION] = PACKET_SCHEMA_VERSION
    purpose: str
    reviewer_id: str
    source_label_visibility: Literal["forbidden"] = "forbidden"
    canonical_size: tuple[int, int]
    assets: ReviewAssets
    lines: list[ReviewLine] = Field(default_factory=list)
    region_proposals: list[ReviewRegionProposal] = Field(default_factory=list)
    family_proposals: list[ReviewFamilyProposal] = Field(default_factory=list)
    instructions: list[str] = Field(min_length=1)


def build_review_packet(
    reviewer_id: str,
    measurement: GeometryMeasurementV2Result,
) -> tuple[GeometryRelationReviewPacket, GeometryRelationAnnotation]:
    """Create an immutable source-blind packet and its editable template."""

    width, height = measurement.canonical_size
    if width <= 0 or height <= 0:
        width, height = 1, 1
    lines = [
        ReviewLine(
            line_id=line.line_id,
            start=NormalizedPoint(x=_unit(line.x1 / width), y=_unit(line.y1 / height)),
            end=NormalizedPoint(x=_unit(line.x2 / width), y=_unit(line.y2 / height)),
            cross_scale_stability=line.cross_scale_stability,
        )
        for line in measurement.merged_lines
    ]
    regions = [
        ReviewRegionProposal(
            region_id=region.region_id,
            box=NormalizedBox(
                x=_unit(region.canonical_box.x / width),
                y=_unit(region.canonical_box.y / height),
                width=min(1.0, region.canonical_box.width / width),
                height=min(1.0, region.canonical_box.height / height),
            ),
            line_ids=list(region.line_ids),
            status=region.status,
        )
        for region in measurement.regions
    ]
    stable_global, stable_local, severity_by_family = select_review_families(measurement)
    families: list[ReviewFamilyProposal] = []
    for asset_name, selected in (
        ("global_families_overlay", stable_global),
        ("local_families_overlay", stable_local),
    ):
        for index, family in enumerate(selected):
            families.append(
                ReviewFamilyProposal(
                    family_id=family.family_id,
                    region_id=family.region_id,
                    kind=family.kind,
                    member_line_ids=list(family.member_line_ids),
                    overlay_asset=asset_name,
                    detail_overlay=f"family_details/{family.family_id}.png",
                    overlay_color_rgb=COLORS[index % len(COLORS)],
                    priority_reason=(
                        "check_finding" if severity_by_family.get(family.family_id, 0.0) > 0.0 else "stable_control"
                    ),
                    maximum_finding_severity=severity_by_family.get(family.family_id, 0.0),
                )
            )
    packet = GeometryRelationReviewPacket(
        purpose=(
            "Blind review of which detected lines belong to the same visible surface and "
            "which geometric relations hold within that surface; not AI-origin annotation."
        ),
        reviewer_id=reviewer_id,
        canonical_size=measurement.canonical_size,
        assets=ReviewAssets(
            image="image.png",
            line_ids_overlay="line_ids_overlay.png",
            regions_overlay="regions_overlay.png",
            local_families_overlay="local_families_overlay.png",
            global_families_overlay="global_families_overlay.png",
            consistency_overlay="consistency_overlay.png",
            repeat_spacing_overlay="repeat_spacing_overlay.png",
            measurement="geometry_measurement_v2.json",
        ),
        lines=lines,
        region_proposals=regions,
        family_proposals=families,
        instructions=[
            "先看匿名原图，再用区域与线号叠图圈定同一可见屋面、立面、道路或物体表面。",
            "方向近似但属于不同物体或不同表面的线必须分开；不要仅凭角度并为一族。",
            "逐个复核程序提议的线族：同一表面内自洽、跨表面误并、同一表面内冲突或无法判断。",
            "弧线、反射、阴影、遮挡、鱼眼和压缩伪影不能单独作为几何冲突。",
            "不得推断或记录图片是否为实拍、AI 生成或来自哪一种生成器。",
        ],
    )
    annotation = GeometryRelationAnnotation(
        reviewer_id=reviewer_id,
        proposed_family_reviews=[
            ProposedFamilyReview(proposed_family_id=family.family_id)
            for family in families
        ],
    )
    return packet, annotation


def validate_annotation_against_packet(
    packet: GeometryRelationReviewPacket,
    annotation: GeometryRelationAnnotation,
) -> None:
    """Reject identity drift, missing proposals, and dangling graph references."""

    if annotation.reviewer_id != packet.reviewer_id:
        raise ValueError("annotation reviewer_id does not match packet")
    line_ids = {line.line_id for line in packet.lines}
    surface_ids = {surface.surface_id for surface in annotation.surfaces}
    proposal_by_id = {proposal.family_id: proposal for proposal in packet.family_proposals}
    review_by_id = {
        review.proposed_family_id: review for review in annotation.proposed_family_reviews
    }
    if set(review_by_id) != set(proposal_by_id):
        raise ValueError("annotation must review every proposed family exactly once")
    for surface in annotation.surfaces:
        _require_subset(surface.line_ids, line_ids, "surface line_ids")
    for review in annotation.proposed_family_reviews:
        proposal = proposal_by_id[review.proposed_family_id]
        _require_subset(review.surface_ids, surface_ids, "family review surface_ids")
        _require_subset(
            review.outlier_line_ids,
            set(proposal.member_line_ids),
            "family review outlier_line_ids",
        )
        assigned_lines = {
            line_id
            for surface in annotation.surfaces
            if surface.surface_id in review.surface_ids
            for line_id in surface.line_ids
        }
        if review.verdict not in {"pending", "unassessable"} and not (
            assigned_lines & set(proposal.member_line_ids)
        ):
            raise ValueError("reviewed family must share a line with its assigned surface")
    for relation in annotation.additional_relations:
        _require_subset(relation.surface_ids, surface_ids, "relation surface_ids")
        _require_subset(relation.member_line_ids, line_ids, "relation member_line_ids")
        _require_subset(
            relation.outlier_line_ids,
            set(relation.member_line_ids),
            "relation outlier_line_ids",
        )


def surface_pair_signature(annotation: GeometryRelationAnnotation) -> dict[tuple[str, str], bool]:
    """Return ID-invariant line co-assignment decisions for duplicate auditing."""

    memberships: dict[str, set[str]] = {}
    for surface in annotation.surfaces:
        for line_id in surface.line_ids:
            memberships.setdefault(line_id, set()).add(surface.surface_id)
    return {
        (first, second): bool(memberships[first] & memberships[second])
        for first, second in combinations(sorted(memberships), 2)
    }


def select_review_families(measurement: GeometryMeasurementV2Result):
    """Select at most four global and four local families without source data."""

    severity_by_family: dict[str, float] = {}
    for check in measurement.checks:
        for finding in check.findings:
            for family_id in finding.family_ids:
                severity_by_family[family_id] = max(
                    severity_by_family.get(family_id, 0.0),
                    finding.severity,
                )

    def ranked(global_scope: bool):
        candidates = [
            family
            for family in measurement.families
            if family.stable and (family.region_id == "global") == global_scope
        ]
        candidates.sort(
            key=lambda family: (
                -severity_by_family.get(family.family_id, 0.0),
                -len(family.member_line_ids),
                -family.bootstrap_stability,
                -family.weighted_inlier_ratio,
                family.family_id,
            )
        )
        return candidates[:MAX_REVIEW_FAMILIES_PER_SCOPE]

    return ranked(True), ranked(False), severity_by_family


def write_relation_review_overlays(
    image_path: Path,
    measurement: GeometryMeasurementV2Result,
    output_dir: Path,
) -> None:
    """Write line-ID and global-family overlays used only by blind review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
    line_by_id = {line.line_id: line for line in measurement.merged_lines}

    global_families, local_families, _ = select_review_families(measurement)
    reviewed_line_ids = {
        line_id
        for family in [*global_families, *local_families]
        for line_id in family.member_line_ids
    }
    line_image = rgb.copy()
    line_draw = ImageDraw.Draw(line_image)
    for line in measurement.merged_lines:
        if line.line_id not in reviewed_line_ids:
            continue
        color = (45, 105, 155) if line.cross_scale_stability >= 0.65 else (120, 120, 120)
        line_draw.line((line.x1, line.y1, line.x2, line.y2), fill=color, width=2)
        middle = ((line.x1 + line.x2) / 2.0, (line.y1 + line.y2) / 2.0)
        line_draw.text(middle, line.line_id, fill=(20, 20, 20), stroke_width=2, stroke_fill=(255, 255, 255))
    _banner(line_draw, line_image.width, "PROPOSED-FAMILY LINE IDS - USE DETAIL OVERLAYS WHEN CROWDED")
    line_image.save(output_dir / "line_ids_overlay.png")

    global_image = rgb.copy()
    global_draw = ImageDraw.Draw(global_image)
    for index, family in enumerate(global_families):
        color = COLORS[index % len(COLORS)]
        for line_id in family.member_line_ids:
            line = line_by_id.get(line_id)
            if line is not None:
                global_draw.line((line.x1, line.y1, line.x2, line.y2), fill=color, width=3)
    _banner(global_draw, global_image.width, "GLOBAL FAMILY PROPOSALS - MAY MERGE DIFFERENT SURFACES")
    global_image.save(output_dir / "global_families_overlay.png")

    local_image = rgb.copy()
    local_draw = ImageDraw.Draw(local_image)
    for index, family in enumerate(local_families):
        color = COLORS[index % len(COLORS)]
        for line_id in family.member_line_ids:
            line = line_by_id.get(line_id)
            if line is not None:
                local_draw.line((line.x1, line.y1, line.x2, line.y2), fill=color, width=3)
    _banner(local_draw, local_image.width, "SELECTED LOCAL FAMILY PROPOSALS - REGION SCOPED")
    local_image.save(output_dir / "local_families_overlay.png")

    detail_dir = output_dir / "family_details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for family in [*global_families, *local_families]:
        detail_image = rgb.copy()
        detail_draw = ImageDraw.Draw(detail_image)
        for line_id in family.member_line_ids:
            line = line_by_id.get(line_id)
            if line is None:
                continue
            detail_draw.line((line.x1, line.y1, line.x2, line.y2), fill=(35, 105, 165), width=4)
            middle = ((line.x1 + line.x2) / 2.0, (line.y1 + line.y2) / 2.0)
            detail_draw.text(
                middle,
                line.line_id,
                fill=(15, 15, 15),
                stroke_width=3,
                stroke_fill=(255, 255, 255),
            )
        _banner(
            detail_draw,
            detail_image.width,
            f"FAMILY {family.family_id} - REVIEW SURFACE MEMBERSHIP BEFORE GEOMETRY",
        )
        detail_image.save(detail_dir / f"{family.family_id}.png")


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _require_subset(values: list[str], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{name} contain unknown values: {unknown}")


def _banner(draw: ImageDraw.ImageDraw, width: int, text: str) -> None:
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
