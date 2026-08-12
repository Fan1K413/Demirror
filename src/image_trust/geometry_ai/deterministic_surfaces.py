"""Deterministic, source-neutral coarse-surface and family-partition baseline.

The baseline consumes only an anonymous RGB image and the geometry relation
review packet produced by geometry-v2.  It does not consume review annotations,
source labels, provenance, camera, watermark, or pixel-detector results.  Its
outputs are diagnostic relationship candidates and never AI-origin evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.geometry_ai.measurement_overlays import COLORS
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationReviewPacket,
    NormalizedBox,
    ReviewLine,
)


BASELINE_SCHEMA_VERSION = "geometry-deterministic-surface-baseline-v1"
BASELINE_ARTIFACT_SCHEMA_VERSION = (
    "geometry-deterministic-surface-baseline-artifacts-v1"
)


class _FrozenBaselineContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LineAffinityWeights(_FrozenBaselineContract):
    midpoint: float = Field(default=0.24, ge=0.0, le=1.0)
    endpoint: float = Field(default=0.24, ge=0.0, le=1.0)
    axis_relation: float = Field(default=0.14, ge=0.0, le=1.0)
    appearance: float = Field(default=0.18, ge=0.0, le=1.0)
    shared_region: float = Field(default=0.10, ge=0.0, le=1.0)
    cross_scale_stability: float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "LineAffinityWeights":
        total = sum(self.model_dump().values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError("line affinity weights must sum to one")
        return self


class DeterministicSurfaceConfig(_FrozenBaselineContract):
    appearance_grid_rows: int = Field(default=16, ge=2, le=64)
    appearance_grid_columns: int = Field(default=16, ge=2, le=64)
    appearance_neighbor_mode: Literal[4] = 4
    appearance_component_color_distance_max: float = Field(
        default=0.11, gt=0.0, le=1.0
    )
    appearance_component_minimum_cells: int = Field(default=2, ge=1)
    minimum_cross_scale_stability: float = Field(default=0.55, ge=0.0, le=1.0)
    line_side_sample_positions: tuple[float, ...] = (0.15, 0.35, 0.50, 0.65, 0.85)
    line_side_offset_fraction: float = Field(default=0.015625, gt=0.0, le=0.1)
    line_side_vote_minimum: int = Field(default=2, ge=1)
    maximum_appearance_memberships_per_line: int = Field(default=2, ge=1, le=4)
    line_midpoint_distance_max: float = Field(default=0.30, gt=0.0, le=1.5)
    line_endpoint_distance_max: float = Field(default=0.10, gt=0.0, le=1.5)
    connected_endpoint_distance_max: float = Field(default=0.045, gt=0.0, le=1.5)
    line_axis_deviation_degrees_max: float = Field(default=30.0, gt=0.0, le=45.0)
    line_appearance_distance_max: float = Field(default=0.20, gt=0.0, le=1.0)
    line_affinity_minimum: float = Field(default=0.58, ge=0.0, le=1.0)
    line_affinity_weights: LineAffinityWeights = Field(
        default_factory=LineAffinityWeights
    )
    minimum_surface_support_lines: int = Field(default=2, ge=2)
    minimum_usable_subfamily_members: int = Field(default=2, ge=2)
    diagnostic_line_link_limit: int = Field(default=4096, ge=0)
    maximum_packet_lines: int = Field(default=512, ge=2)

    @model_validator(mode="after")
    def _positions_and_distances_are_consistent(self) -> "DeterministicSurfaceConfig":
        positions = self.line_side_sample_positions
        if not positions or any(value <= 0.0 or value >= 1.0 for value in positions):
            raise ValueError("line side sample positions must be strictly inside the line")
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("line side sample positions must be unique and ascending")
        if self.connected_endpoint_distance_max > self.line_endpoint_distance_max:
            raise ValueError("connected endpoint distance cannot exceed endpoint distance")
        if self.line_side_vote_minimum > 2 * len(positions):
            raise ValueError("line side vote minimum exceeds the available samples")
        return self


DEFAULT_DETERMINISTIC_SURFACE_CONFIG = DeterministicSurfaceConfig()


class AppearanceComponent(_FrozenBaselineContract):
    component_id: str
    cell_ids: list[str] = Field(min_length=1)
    coarse_box: NormalizedBox
    mean_lab: tuple[float, float, float]
    area_fraction: float = Field(gt=0.0, le=1.0)


class BaselineLineAssignment(_FrozenBaselineContract):
    line_id: str
    stability_eligible: bool
    appearance_component_ids: list[str] = Field(default_factory=list)
    surface_candidate_ids: list[str] = Field(default_factory=list)
    exclusion_reason: Literal[
        "low_cross_scale_stability",
        "no_supported_appearance_component",
        "no_connected_surface_candidate",
    ] | None = None


class BaselineLineLink(_FrozenBaselineContract):
    appearance_component_id: str
    first_line_id: str
    second_line_id: str
    affinity: float = Field(ge=0.0, le=1.0)
    midpoint_distance: float = Field(ge=0.0)
    endpoint_distance: float = Field(ge=0.0)
    axis_deviation_degrees: float = Field(ge=0.0, le=45.0)
    appearance_distance: float = Field(ge=0.0, le=1.0)
    shared_region: bool

    @model_validator(mode="after")
    def _canonical_pair(self) -> "BaselineLineLink":
        if self.first_line_id >= self.second_line_id:
            raise ValueError("baseline line links must use ascending line IDs")
        return self


class DeterministicSurfaceCandidate(_FrozenBaselineContract):
    surface_id: str
    surface_kind: Literal["coarse_appearance_geometry"] = (
        "coarse_appearance_geometry"
    )
    appearance_component_id: str
    coarse_box: NormalizedBox
    member_line_ids: list[str] = Field(min_length=2)
    region_ids: list[str] = Field(default_factory=list)
    support_link_count: int = Field(ge=1)
    affinity_mean: float = Field(ge=0.0, le=1.0)
    affinity_p50: float = Field(ge=0.0, le=1.0)
    stability_mean: float = Field(ge=0.0, le=1.0)


class DeterministicSurfaceSubfamily(_FrozenBaselineContract):
    subfamily_id: str
    surface_id: str
    member_line_ids: list[str] = Field(min_length=1)
    usability: Literal["usable", "insufficient_members"]


class DeterministicFamilyPartition(_FrozenBaselineContract):
    original_family_id: str
    original_region_id: str
    kind: Literal["parallel", "finite_vp", "infinite_vp"]
    original_member_line_ids: list[str] = Field(min_length=2)
    surface_subfamilies: list[DeterministicSurfaceSubfamily] = Field(
        default_factory=list
    )
    unassigned_line_ids: list[str] = Field(default_factory=list)
    partition_status: Literal[
        "split_candidate",
        "single_surface_candidate",
        "insufficient_support",
    ]


class DeterministicSurfaceArtifacts(_FrozenBaselineContract):
    result_json: str | None = None
    surface_candidates_overlay: str | None = None
    family_partitions_overlay: str | None = None


class DeterministicSurfaceBaselineResult(_FrozenBaselineContract):
    schema_version: Literal[BASELINE_SCHEMA_VERSION] = BASELINE_SCHEMA_VERSION
    status: Literal["available", "not_applicable"]
    summary: str
    reviewer_id: str
    canonical_size: tuple[int, int]
    config: DeterministicSurfaceConfig = Field(
        default_factory=DeterministicSurfaceConfig
    )
    appearance_components: list[AppearanceComponent] = Field(default_factory=list)
    line_assignments: list[BaselineLineAssignment] = Field(default_factory=list)
    accepted_line_link_count: int = Field(default=0, ge=0)
    diagnostic_line_links: list[BaselineLineLink] = Field(default_factory=list)
    diagnostic_links_truncated: bool = False
    surface_candidates: list[DeterministicSurfaceCandidate] = Field(
        default_factory=list
    )
    family_partitions: list[DeterministicFamilyPartition] = Field(
        default_factory=list
    )
    artifacts: DeterministicSurfaceArtifacts = Field(
        default_factory=DeterministicSurfaceArtifacts
    )
    limitations: list[str] = Field(default_factory=list)
    source_label_visibility: Literal["forbidden"] = "forbidden"
    source_labels_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False
    web_integration_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _relationship_closure(self) -> "DeterministicSurfaceBaselineResult":
        if self.canonical_size[0] <= 0 or self.canonical_size[1] <= 0:
            raise ValueError("baseline canonical size must be positive")
        if self.status == "not_applicable":
            if any(
                (
                    self.appearance_components,
                    self.line_assignments,
                    self.diagnostic_line_links,
                    self.surface_candidates,
                    self.family_partitions,
                )
            ) or self.accepted_line_link_count:
                raise ValueError("not-applicable baseline must not publish partial results")
            return self

        component_by_id = _index_unique(
            self.appearance_components, "component_id", "appearance component"
        )
        line_by_id = _index_unique(self.line_assignments, "line_id", "line assignment")
        surface_by_id = _index_unique(
            self.surface_candidates, "surface_id", "surface candidate"
        )
        family_by_id = _index_unique(
            self.family_partitions, "original_family_id", "family partition"
        )
        component_ids = set(component_by_id)
        surface_ids = set(surface_by_id)
        line_ids = set(line_by_id)

        expected_surfaces_by_line: dict[str, set[str]] = {
            line_id: set() for line_id in line_ids
        }
        for surface in self.surface_candidates:
            if surface.appearance_component_id not in component_ids:
                raise ValueError("surface candidate references an unknown appearance component")
            _require_unique_subset(
                surface.member_line_ids,
                line_ids,
                f"surface {surface.surface_id} members",
            )
            for line_id in surface.member_line_ids:
                assignment = line_by_id[line_id]
                if surface.appearance_component_id not in assignment.appearance_component_ids:
                    raise ValueError(
                        "surface member does not reference its appearance component"
                    )
                expected_surfaces_by_line[line_id].add(surface.surface_id)
        for assignment in self.line_assignments:
            _require_unique_subset(
                assignment.appearance_component_ids,
                component_ids,
                f"line {assignment.line_id} appearance memberships",
            )
            if (
                len(assignment.appearance_component_ids)
                > self.config.maximum_appearance_memberships_per_line
            ):
                raise ValueError("line appearance memberships exceed the frozen limit")
            _require_unique_subset(
                assignment.surface_candidate_ids,
                surface_ids,
                f"line {assignment.line_id} surface memberships",
            )
            if set(assignment.surface_candidate_ids) != expected_surfaces_by_line[
                assignment.line_id
            ]:
                raise ValueError("line and surface candidate memberships disagree")
            expected_exclusion_reason = _assignment_exclusion_reason(assignment)
            if assignment.exclusion_reason != expected_exclusion_reason:
                raise ValueError("line exclusion reason is inconsistent")

        link_keys: set[tuple[str, str, str]] = set()
        for link in self.diagnostic_line_links:
            key = (
                link.appearance_component_id,
                link.first_line_id,
                link.second_line_id,
            )
            if key in link_keys:
                raise ValueError("diagnostic line links must be unique")
            link_keys.add(key)
            if link.appearance_component_id not in component_ids:
                raise ValueError("line link references an unknown appearance component")
            _require_unique_subset(
                [link.first_line_id, link.second_line_id],
                line_ids,
                "line link endpoints",
            )
        if len(self.diagnostic_line_links) > self.accepted_line_link_count:
            raise ValueError("diagnostic links exceed the accepted-link count")
        if self.diagnostic_links_truncated != (
            len(self.diagnostic_line_links) < self.accepted_line_link_count
        ):
            raise ValueError("diagnostic link truncation flag is inconsistent")

        for partition in self.family_partitions:
            original_members = set(partition.original_member_line_ids)
            _require_unique_subset(
                partition.original_member_line_ids,
                line_ids,
                f"family {partition.original_family_id} original members",
            )
            _require_unique_subset(
                partition.unassigned_line_ids,
                original_members,
                f"family {partition.original_family_id} unassigned members",
            )
            subfamily_ids: set[str] = set()
            assigned: set[str] = set()
            usable_count = 0
            for subfamily in partition.surface_subfamilies:
                if subfamily.subfamily_id in subfamily_ids:
                    raise ValueError("surface subfamily IDs must be unique")
                subfamily_ids.add(subfamily.subfamily_id)
                surface = surface_by_id.get(subfamily.surface_id)
                if surface is None:
                    raise ValueError("surface subfamily references an unknown surface")
                _require_unique_subset(
                    subfamily.member_line_ids,
                    original_members & set(surface.member_line_ids),
                    f"subfamily {subfamily.subfamily_id} members",
                )
                expected_subfamily_id = _subfamily_id(
                    partition.original_family_id,
                    subfamily.surface_id,
                )
                if subfamily.subfamily_id != expected_subfamily_id:
                    raise ValueError("surface subfamily ID is not deterministic")
                expected_usability = (
                    "usable"
                    if len(subfamily.member_line_ids)
                    >= self.config.minimum_usable_subfamily_members
                    else "insufficient_members"
                )
                if subfamily.usability != expected_usability:
                    raise ValueError("surface subfamily usability is inconsistent")
                if subfamily.usability == "usable":
                    usable_count += 1
                assigned.update(subfamily.member_line_ids)
            unassigned = set(partition.unassigned_line_ids)
            if assigned & unassigned or assigned | unassigned != original_members:
                raise ValueError(
                    f"family {partition.original_family_id} member closure is incomplete"
                )
            expected_status = _partition_status(usable_count)
            if partition.partition_status != expected_status:
                raise ValueError("family partition status is inconsistent")
        if len(family_by_id) != len(self.family_partitions):
            raise ValueError("family partition IDs must be unique")
        return self


class DeterministicSurfaceArtifactManifest(_FrozenBaselineContract):
    schema_version: Literal[BASELINE_ARTIFACT_SCHEMA_VERSION] = (
        BASELINE_ARTIFACT_SCHEMA_VERSION
    )
    result_json: str
    surface_candidates_overlay: str
    family_partitions_overlay: str
    line_count: int = Field(ge=0)
    appearance_component_count: int = Field(ge=0)
    surface_candidate_count: int = Field(ge=0)
    family_partition_count: int = Field(ge=0)
    split_candidate_count: int = Field(ge=0)
    accepted_line_link_count: int = Field(ge=0)
    source_labels_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False
    web_integration_authorized: Literal[False] = False


@dataclass(frozen=True)
class _AppearanceComponentInternal:
    model: AppearanceComponent
    cells: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class _LineFeature:
    line: ReviewLine
    midpoint: tuple[float, float]
    endpoints: tuple[tuple[float, float], tuple[float, float]]
    angle_degrees: float
    local_lab: tuple[float, float, float]
    region_ids: frozenset[str]
    appearance_component_ids: tuple[str, ...]
    stability_eligible: bool


def assess_deterministic_surface_baseline(
    image: Image.Image,
    packet: GeometryRelationReviewPacket,
    config: DeterministicSurfaceConfig = DEFAULT_DETERMINISTIC_SURFACE_CONFIG,
) -> DeterministicSurfaceBaselineResult:
    """Build deterministic coarse surfaces without consulting any annotation."""

    _validate_packet_geometry(packet)
    rgb_image = image.convert("RGB")
    if rgb_image.size != packet.canonical_size:
        raise ValueError(
            f"baseline image size {rgb_image.size} does not match packet "
            f"canonical size {packet.canonical_size}"
        )
    width, height = rgb_image.size
    if len(packet.lines) > config.maximum_packet_lines:
        return _not_applicable(
            packet,
            config,
            "packet exceeds the frozen line-count resource limit",
        )
    if width < config.appearance_grid_columns or height < config.appearance_grid_rows:
        return _not_applicable(
            packet,
            config,
            "image is smaller than the frozen appearance grid",
        )

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64) / 255.0
    components, cell_component = _appearance_components(lab, config)
    region_ids_by_line = _region_ids_by_line(packet)
    features = [
        _line_feature(
            line,
            lab,
            cell_component,
            region_ids_by_line[line.line_id],
            config,
        )
        for line in packet.lines
    ]
    feature_by_id = {feature.line.line_id: feature for feature in features}
    component_by_id = {component.model.component_id: component for component in components}
    surface_candidates, accepted_links = _surface_candidates(
        features,
        component_by_id,
        config,
    )
    surface_ids_by_line: dict[str, list[str]] = {
        feature.line.line_id: [] for feature in features
    }
    for surface in surface_candidates:
        for line_id in surface.member_line_ids:
            surface_ids_by_line[line_id].append(surface.surface_id)
    line_assignments = [
        BaselineLineAssignment(
            line_id=feature.line.line_id,
            stability_eligible=feature.stability_eligible,
            appearance_component_ids=list(feature.appearance_component_ids),
            surface_candidate_ids=surface_ids_by_line[feature.line.line_id],
            exclusion_reason=_line_exclusion_reason(
                feature,
                surface_ids_by_line[feature.line.line_id],
            ),
        )
        for feature in features
    ]
    partitions = _family_partitions(packet, surface_candidates, config)
    diagnostic_links = sorted(
        accepted_links,
        key=lambda link: (
            -link.affinity,
            link.appearance_component_id,
            link.first_line_id,
            link.second_line_id,
        ),
    )[: config.diagnostic_line_link_limit]
    return DeterministicSurfaceBaselineResult(
        status="available",
        summary=(
            "deterministic coarse appearance-geometry surfaces and family "
            "partitions completed"
        ),
        reviewer_id=packet.reviewer_id,
        canonical_size=packet.canonical_size,
        config=config,
        appearance_components=[component.model for component in components],
        line_assignments=line_assignments,
        accepted_line_link_count=len(accepted_links),
        diagnostic_line_links=diagnostic_links,
        diagnostic_links_truncated=len(diagnostic_links) < len(accepted_links),
        surface_candidates=surface_candidates,
        family_partitions=partitions,
        limitations=[
            "Grid appearance components are coarse regions, not semantic planes.",
            "Adjacent surfaces with similar appearance may merge; textured surfaces may split.",
            "The baseline measures grouping only and does not test projective consistency.",
            "No output from this baseline is eligible for AI-origin scoring.",
        ],
    )


def export_deterministic_surface_diagnostics(
    image: Image.Image,
    packet: GeometryRelationReviewPacket,
    output_dir: Path,
    config: DeterministicSurfaceConfig = DEFAULT_DETERMINISTIC_SURFACE_CONFIG,
) -> tuple[DeterministicSurfaceBaselineResult, DeterministicSurfaceArtifactManifest]:
    """Write the deterministic result and its review overlays atomically."""

    result = assess_deterministic_surface_baseline(image, packet, config)
    return write_deterministic_surface_diagnostics(
        image,
        packet,
        result,
        output_dir,
    )


def write_deterministic_surface_diagnostics(
    image: Image.Image,
    packet: GeometryRelationReviewPacket,
    result: DeterministicSurfaceBaselineResult,
    output_dir: Path,
) -> tuple[DeterministicSurfaceBaselineResult, DeterministicSurfaceArtifactManifest]:
    """Publish an already computed baseline result after identity validation."""

    _validate_result_against_packet(packet, result)
    rgb = image.convert("RGB")
    if rgb.size != packet.canonical_size:
        raise ValueError("diagnostic image size does not match the review packet")
    surface_overlay = render_deterministic_surface_candidates(rgb, packet, result)
    family_overlay = render_deterministic_family_partitions(rgb, packet, result)
    result_name = "deterministic_surface_baseline.json"
    surface_name = "deterministic_surface_candidates_overlay.png"
    family_name = "original_vs_deterministic_family_partitions.png"
    manifest_name = "deterministic_surface_baseline_artifacts.json"
    artifacts = DeterministicSurfaceArtifacts(
        result_json=result_name,
        surface_candidates_overlay=surface_name,
        family_partitions_overlay=family_name,
    )
    result = result.model_copy(update={"artifacts": artifacts})
    manifest = DeterministicSurfaceArtifactManifest(
        result_json=result_name,
        surface_candidates_overlay=surface_name,
        family_partitions_overlay=family_name,
        line_count=len(packet.lines),
        appearance_component_count=len(result.appearance_components),
        surface_candidate_count=len(result.surface_candidates),
        family_partition_count=len(result.family_partitions),
        split_candidate_count=sum(
            partition.partition_status == "split_candidate"
            for partition in result.family_partitions
        ),
        accepted_line_link_count=result.accepted_line_link_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / result_name, result.model_dump(mode="json"))
    _atomic_write_image(output_dir / surface_name, surface_overlay)
    _atomic_write_image(output_dir / family_name, family_overlay)
    _atomic_write_json(output_dir / manifest_name, manifest.model_dump(mode="json"))
    return result, manifest


def render_deterministic_surface_candidates(
    image: Image.Image,
    packet: GeometryRelationReviewPacket,
    result: DeterministicSurfaceBaselineResult,
) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_by_id = {line.line_id: line for line in packet.lines}
    for index, surface in enumerate(result.surface_candidates):
        color = COLORS[index % len(COLORS)]
        box = _pixel_box(surface.coarse_box, canvas.size)
        draw.rectangle(box, outline=color, width=2)
        draw.text(
            (box[0] + 3, box[1] + 3),
            f"C{index + 1}",
            fill=color,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
        for line_id in surface.member_line_ids:
            _draw_line(draw, line_by_id[line_id], canvas.size, color, width=3)
    _banner(
        draw,
        canvas.width,
        f"DETERMINISTIC COARSE SURFACES: {len(result.surface_candidates)} - DIAGNOSTIC ONLY",
    )
    return canvas


def render_deterministic_family_partitions(
    image: Image.Image,
    packet: GeometryRelationReviewPacket,
    result: DeterministicSurfaceBaselineResult,
) -> Image.Image:
    left = image.convert("RGB")
    right = image.convert("RGB")
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    line_by_id = {line.line_id: line for line in packet.lines}
    for index, family in enumerate(packet.family_proposals):
        color = COLORS[index % len(COLORS)]
        for line_id in family.member_line_ids:
            _draw_line(left_draw, line_by_id[line_id], left.size, color, width=4)
    subfamily_index = 0
    for partition in result.family_partitions:
        for subfamily in partition.surface_subfamilies:
            color = COLORS[subfamily_index % len(COLORS)]
            subfamily_index += 1
            for line_id in subfamily.member_line_ids:
                _draw_line(right_draw, line_by_id[line_id], right.size, color, width=4)
        for line_id in partition.unassigned_line_ids:
            _draw_dashed_line(
                right_draw,
                line_by_id[line_id],
                right.size,
                (115, 115, 115),
                width=2,
            )
    _banner(left_draw, left.width, "ORIGINAL DIRECTION FAMILIES")
    _banner(right_draw, right.width, "DETERMINISTIC SURFACE PARTITIONS")
    combined = Image.new("RGB", (left.width * 2 + 2, left.height), "white")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + 2, 0))
    ImageDraw.Draw(combined).rectangle(
        (left.width, 0, left.width + 1, left.height), fill=(20, 20, 20)
    )
    return combined


def _appearance_components(
    lab: np.ndarray,
    config: DeterministicSurfaceConfig,
) -> tuple[list[_AppearanceComponentInternal], dict[tuple[int, int], str]]:
    height, width = lab.shape[:2]
    rows = config.appearance_grid_rows
    columns = config.appearance_grid_columns
    means: dict[tuple[int, int], np.ndarray] = {}
    pixel_counts: dict[tuple[int, int], int] = {}
    for row in range(rows):
        y0, y1 = _grid_bounds(row, rows, height)
        for column in range(columns):
            x0, x1 = _grid_bounds(column, columns, width)
            patch = lab[y0:y1, x0:x1]
            means[(row, column)] = patch.mean(axis=(0, 1))
            pixel_counts[(row, column)] = int(patch.shape[0] * patch.shape[1])

    union = _UnionFind([_cell_key(row, column) for row in range(rows) for column in range(columns)])
    for row in range(rows):
        for column in range(columns):
            for other_row, other_column in ((row - 1, column), (row, column - 1)):
                if other_row < 0 or other_column < 0:
                    continue
                distance = _lab_distance(
                    means[(row, column)], means[(other_row, other_column)]
                )
                if distance <= config.appearance_component_color_distance_max:
                    union.union(
                        _cell_key(row, column),
                        _cell_key(other_row, other_column),
                    )
    cells_by_root: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in range(rows):
        for column in range(columns):
            cells_by_root[union.find(_cell_key(row, column))].append((row, column))

    components: list[_AppearanceComponentInternal] = []
    cell_component: dict[tuple[int, int], str] = {}
    total_pixels = width * height
    ordered_groups = sorted(cells_by_root.values(), key=lambda cells: cells[0])
    for cells in ordered_groups:
        if len(cells) < config.appearance_component_minimum_cells:
            continue
        cell_ids = [_cell_key(row, column) for row, column in cells]
        component_id = _stable_id("ac", cell_ids)
        pixel_count = sum(pixel_counts[cell] for cell in cells)
        weighted_mean = sum(
            means[cell] * pixel_counts[cell] for cell in cells
        ) / pixel_count
        min_row = min(row for row, _ in cells)
        max_row = max(row for row, _ in cells)
        min_column = min(column for _, column in cells)
        max_column = max(column for _, column in cells)
        model = AppearanceComponent(
            component_id=component_id,
            cell_ids=cell_ids,
            coarse_box=NormalizedBox(
                x=min_column / columns,
                y=min_row / rows,
                width=(max_column - min_column + 1) / columns,
                height=(max_row - min_row + 1) / rows,
            ),
            mean_lab=tuple(_rounded(value) for value in weighted_mean),
            area_fraction=_rounded(pixel_count / total_pixels),
        )
        components.append(
            _AppearanceComponentInternal(model=model, cells=frozenset(cells))
        )
        for cell in cells:
            cell_component[cell] = component_id
    return components, cell_component


def _line_feature(
    line: ReviewLine,
    lab: np.ndarray,
    cell_component: dict[tuple[int, int], str],
    region_ids: frozenset[str],
    config: DeterministicSurfaceConfig,
) -> _LineFeature:
    height, width = lab.shape[:2]
    start = (line.start.x, line.start.y)
    end = (line.end.x, line.end.y)
    dx_pixels = (end[0] - start[0]) * max(1, width - 1)
    dy_pixels = (end[1] - start[1]) * max(1, height - 1)
    pixel_length = math.hypot(dx_pixels, dy_pixels)
    stability_eligible = (
        line.cross_scale_stability >= config.minimum_cross_scale_stability
        and pixel_length > 0.0
    )
    votes: Counter[str] = Counter()
    lab_samples: list[np.ndarray] = []
    if pixel_length > 0.0:
        normal_x = -dy_pixels / pixel_length
        normal_y = dx_pixels / pixel_length
        offset = max(1, round(min(width, height) * config.line_side_offset_fraction))
        for position in config.line_side_sample_positions:
            center_x = (start[0] + (end[0] - start[0]) * position) * (width - 1)
            center_y = (start[1] + (end[1] - start[1]) * position) * (height - 1)
            for sign in (-1.0, 1.0):
                sample_x = int(round(center_x + sign * normal_x * offset))
                sample_y = int(round(center_y + sign * normal_y * offset))
                sample_x = min(width - 1, max(0, sample_x))
                sample_y = min(height - 1, max(0, sample_y))
                lab_samples.append(_patch_mean(lab, sample_x, sample_y))
                row = min(
                    config.appearance_grid_rows - 1,
                    sample_y * config.appearance_grid_rows // height,
                )
                column = min(
                    config.appearance_grid_columns - 1,
                    sample_x * config.appearance_grid_columns // width,
                )
                component_id = cell_component.get((row, column))
                if component_id is not None:
                    votes[component_id] += 1
    memberships = tuple(
        component_id
        for component_id, count in sorted(
            votes.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= config.line_side_vote_minimum
    )[: config.maximum_appearance_memberships_per_line]
    local_lab = (
        np.mean(np.stack(lab_samples), axis=0)
        if lab_samples
        else np.zeros(3, dtype=np.float64)
    )
    return _LineFeature(
        line=line,
        midpoint=((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0),
        endpoints=(start, end),
        angle_degrees=(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0),
        local_lab=tuple(float(value) for value in local_lab),
        region_ids=region_ids,
        appearance_component_ids=memberships if stability_eligible else (),
        stability_eligible=stability_eligible,
    )


def _surface_candidates(
    features: list[_LineFeature],
    component_by_id: dict[str, _AppearanceComponentInternal],
    config: DeterministicSurfaceConfig,
) -> tuple[list[DeterministicSurfaceCandidate], list[BaselineLineLink]]:
    feature_by_id = {feature.line.line_id: feature for feature in features}
    order_by_id = {
        feature.line.line_id: index for index, feature in enumerate(features)
    }
    line_ids_by_component: dict[str, list[str]] = defaultdict(list)
    for feature in features:
        for component_id in feature.appearance_component_ids:
            line_ids_by_component[component_id].append(feature.line.line_id)

    surfaces: list[DeterministicSurfaceCandidate] = []
    all_links: list[BaselineLineLink] = []
    for component_id in sorted(line_ids_by_component):
        line_ids = sorted(
            set(line_ids_by_component[component_id]), key=order_by_id.__getitem__
        )
        union = _UnionFind(line_ids)
        component_links: list[BaselineLineLink] = []
        for first_id, second_id in combinations(sorted(line_ids), 2):
            link = _accepted_line_link(
                component_id,
                feature_by_id[first_id],
                feature_by_id[second_id],
                config,
            )
            if link is None:
                continue
            union.union(first_id, second_id)
            component_links.append(link)
            all_links.append(link)
        members_by_root: dict[str, list[str]] = defaultdict(list)
        for line_id in line_ids:
            members_by_root[union.find(line_id)].append(line_id)
        groups = sorted(
            members_by_root.values(),
            key=lambda members: min(order_by_id[line_id] for line_id in members),
        )
        for members in groups:
            if len(members) < config.minimum_surface_support_lines:
                continue
            member_set = set(members)
            support_links = [
                link
                for link in component_links
                if link.first_line_id in member_set and link.second_line_id in member_set
            ]
            if not support_links:
                continue
            ordered_members = sorted(members, key=order_by_id.__getitem__)
            affinities = [link.affinity for link in support_links]
            region_ids = sorted(
                {
                    region_id
                    for line_id in ordered_members
                    for region_id in feature_by_id[line_id].region_ids
                }
            )
            surface_id = _stable_id("dsf", [component_id, *sorted(ordered_members)])
            surfaces.append(
                DeterministicSurfaceCandidate(
                    surface_id=surface_id,
                    appearance_component_id=component_id,
                    coarse_box=_line_group_box(
                        [feature_by_id[line_id] for line_id in ordered_members]
                    ),
                    member_line_ids=ordered_members,
                    region_ids=region_ids,
                    support_link_count=len(support_links),
                    affinity_mean=_rounded(statistics.fmean(affinities)),
                    affinity_p50=_rounded(statistics.median(affinities)),
                    stability_mean=_rounded(
                        statistics.fmean(
                            feature_by_id[line_id].line.cross_scale_stability
                            for line_id in ordered_members
                        )
                    ),
                )
            )
    return surfaces, all_links


def _accepted_line_link(
    component_id: str,
    first: _LineFeature,
    second: _LineFeature,
    config: DeterministicSurfaceConfig,
) -> BaselineLineLink | None:
    midpoint_distance = math.dist(first.midpoint, second.midpoint)
    endpoint_distance = min(
        math.dist(first_endpoint, second_endpoint)
        for first_endpoint in first.endpoints
        for second_endpoint in second.endpoints
    )
    angle_difference = abs(first.angle_degrees - second.angle_degrees)
    angle_difference = min(angle_difference, 180.0 - angle_difference)
    axis_deviation = min(angle_difference, abs(90.0 - angle_difference))
    appearance_distance = _lab_distance(first.local_lab, second.local_lab)
    if (
        midpoint_distance > config.line_midpoint_distance_max
        and endpoint_distance > config.line_endpoint_distance_max
    ):
        return None
    if appearance_distance > config.line_appearance_distance_max:
        return None
    if (
        axis_deviation > config.line_axis_deviation_degrees_max
        and endpoint_distance > config.connected_endpoint_distance_max
    ):
        return None
    midpoint_score = _linear_score(
        midpoint_distance, config.line_midpoint_distance_max
    )
    endpoint_score = _linear_score(
        endpoint_distance, config.line_endpoint_distance_max
    )
    axis_score = _linear_score(
        axis_deviation, config.line_axis_deviation_degrees_max
    )
    appearance_score = _linear_score(
        appearance_distance, config.line_appearance_distance_max
    )
    shared_region = bool(first.region_ids & second.region_ids)
    weights = config.line_affinity_weights
    affinity = (
        weights.midpoint * midpoint_score
        + weights.endpoint * endpoint_score
        + weights.axis_relation * axis_score
        + weights.appearance * appearance_score
        + weights.shared_region * float(shared_region)
        + weights.cross_scale_stability
        * min(
            first.line.cross_scale_stability,
            second.line.cross_scale_stability,
        )
    )
    if affinity < config.line_affinity_minimum:
        return None
    first_id, second_id = sorted((first.line.line_id, second.line.line_id))
    return BaselineLineLink(
        appearance_component_id=component_id,
        first_line_id=first_id,
        second_line_id=second_id,
        affinity=_rounded(affinity),
        midpoint_distance=_rounded(midpoint_distance),
        endpoint_distance=_rounded(endpoint_distance),
        axis_deviation_degrees=_rounded(axis_deviation),
        appearance_distance=_rounded(appearance_distance),
        shared_region=shared_region,
    )


def _family_partitions(
    packet: GeometryRelationReviewPacket,
    surfaces: list[DeterministicSurfaceCandidate],
    config: DeterministicSurfaceConfig,
) -> list[DeterministicFamilyPartition]:
    surface_members = {
        surface.surface_id: set(surface.member_line_ids) for surface in surfaces
    }
    partitions: list[DeterministicFamilyPartition] = []
    for family in packet.family_proposals:
        subfamilies: list[DeterministicSurfaceSubfamily] = []
        assigned: set[str] = set()
        for surface in surfaces:
            members = [
                line_id
                for line_id in family.member_line_ids
                if line_id in surface_members[surface.surface_id]
            ]
            if not members:
                continue
            assigned.update(members)
            subfamilies.append(
                DeterministicSurfaceSubfamily(
                    subfamily_id=_subfamily_id(family.family_id, surface.surface_id),
                    surface_id=surface.surface_id,
                    member_line_ids=members,
                    usability=(
                        "usable"
                        if len(members) >= config.minimum_usable_subfamily_members
                        else "insufficient_members"
                    ),
                )
            )
        usable_count = sum(
            subfamily.usability == "usable" for subfamily in subfamilies
        )
        partitions.append(
            DeterministicFamilyPartition(
                original_family_id=family.family_id,
                original_region_id=family.region_id,
                kind=family.kind,
                original_member_line_ids=family.member_line_ids,
                surface_subfamilies=subfamilies,
                unassigned_line_ids=[
                    line_id
                    for line_id in family.member_line_ids
                    if line_id not in assigned
                ],
                partition_status=_partition_status(usable_count),
            )
        )
    return partitions


def _validate_packet_geometry(packet: GeometryRelationReviewPacket) -> None:
    width, height = packet.canonical_size
    if width <= 0 or height <= 0:
        raise ValueError("review packet canonical size must be positive")
    line_ids = [line.line_id for line in packet.lines]
    if len(line_ids) != len(set(line_ids)):
        raise ValueError("review packet line IDs must be unique")
    line_id_set = set(line_ids)
    for line in packet.lines:
        if line.start == line.end:
            raise ValueError(f"review packet line {line.line_id} has zero length")
    region_ids = [region.region_id for region in packet.region_proposals]
    if len(region_ids) != len(set(region_ids)):
        raise ValueError("review packet region IDs must be unique")
    region_id_set = set(region_ids)
    for region in packet.region_proposals:
        _require_unique_subset(
            region.line_ids,
            line_id_set,
            f"region {region.region_id} lines",
        )
    family_ids = [family.family_id for family in packet.family_proposals]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("review packet family IDs must be unique")
    for family in packet.family_proposals:
        _require_unique_subset(
            family.member_line_ids,
            line_id_set,
            f"family {family.family_id} members",
        )
        if family.region_id != "global" and family.region_id not in region_id_set:
            raise ValueError(f"family {family.family_id} references an unknown region")


def _validate_result_against_packet(
    packet: GeometryRelationReviewPacket,
    result: DeterministicSurfaceBaselineResult,
) -> None:
    if result.reviewer_id != packet.reviewer_id:
        raise ValueError("baseline result reviewer ID does not match the packet")
    if result.canonical_size != packet.canonical_size:
        raise ValueError("baseline result canonical size does not match the packet")
    if result.status != "available":
        return
    if [assignment.line_id for assignment in result.line_assignments] != [
        line.line_id for line in packet.lines
    ]:
        raise ValueError("baseline line assignments do not match packet line order")
    if [partition.original_family_id for partition in result.family_partitions] != [
        family.family_id for family in packet.family_proposals
    ]:
        raise ValueError("baseline family partitions do not match packet family order")


def _region_ids_by_line(
    packet: GeometryRelationReviewPacket,
) -> dict[str, frozenset[str]]:
    values: dict[str, set[str]] = {line.line_id: set() for line in packet.lines}
    for region in packet.region_proposals:
        for line_id in region.line_ids:
            values[line_id].add(region.region_id)
    return {line_id: frozenset(region_ids) for line_id, region_ids in values.items()}


def _line_exclusion_reason(
    feature: _LineFeature,
    surface_ids: list[str],
) -> Literal[
    "low_cross_scale_stability",
    "no_supported_appearance_component",
    "no_connected_surface_candidate",
] | None:
    if surface_ids:
        return None
    if not feature.stability_eligible:
        return "low_cross_scale_stability"
    if not feature.appearance_component_ids:
        return "no_supported_appearance_component"
    return "no_connected_surface_candidate"


def _assignment_exclusion_reason(
    assignment: BaselineLineAssignment,
) -> Literal[
    "low_cross_scale_stability",
    "no_supported_appearance_component",
    "no_connected_surface_candidate",
] | None:
    if assignment.surface_candidate_ids:
        return None
    if not assignment.stability_eligible:
        return "low_cross_scale_stability"
    if not assignment.appearance_component_ids:
        return "no_supported_appearance_component"
    return "no_connected_surface_candidate"


def _not_applicable(
    packet: GeometryRelationReviewPacket,
    config: DeterministicSurfaceConfig,
    summary: str,
) -> DeterministicSurfaceBaselineResult:
    return DeterministicSurfaceBaselineResult(
        status="not_applicable",
        summary=summary,
        reviewer_id=packet.reviewer_id,
        canonical_size=packet.canonical_size,
        config=config,
        limitations=[
            summary,
            "No partial relationship result or origin score was published.",
        ],
    )


def _line_group_box(features: list[_LineFeature]) -> NormalizedBox:
    x_values = [point[0] for feature in features for point in feature.endpoints]
    y_values = [point[1] for feature in features for point in feature.endpoints]
    padding = 0.025
    x0 = max(0.0, min(x_values) - padding)
    y0 = max(0.0, min(y_values) - padding)
    x1 = min(1.0, max(x_values) + padding)
    y1 = min(1.0, max(y_values) + padding)
    return NormalizedBox(
        x=x0,
        y=y0,
        width=max(1e-9, x1 - x0),
        height=max(1e-9, y1 - y0),
    )


def _partition_status(
    usable_count: int,
) -> Literal["split_candidate", "single_surface_candidate", "insufficient_support"]:
    if usable_count >= 2:
        return "split_candidate"
    if usable_count == 1:
        return "single_surface_candidate"
    return "insufficient_support"


def _subfamily_id(family_id: str, surface_id: str) -> str:
    return _stable_id("dsf-family", [family_id, surface_id])


def _stable_id(prefix: str, values: list[str]) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _cell_key(row: int, column: int) -> str:
    return f"c{row:02d}-{column:02d}"


def _grid_bounds(index: int, count: int, length: int) -> tuple[int, int]:
    return index * length // count, (index + 1) * length // count


def _patch_mean(lab: np.ndarray, x: int, y: int) -> np.ndarray:
    height, width = lab.shape[:2]
    return lab[
        max(0, y - 1) : min(height, y + 2),
        max(0, x - 1) : min(width, x + 2),
    ].mean(axis=(0, 1))


def _lab_distance(first, second) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)) / math.sqrt(3.0))


def _linear_score(value: float, maximum: float) -> float:
    return max(0.0, 1.0 - value / maximum)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _index_unique(items: list[BaseModel], attribute: str, label: str) -> dict[str, BaseModel]:
    indexed: dict[str, BaseModel] = {}
    for item in items:
        identifier = str(getattr(item, attribute))
        if identifier in indexed:
            raise ValueError(f"{label} IDs must be unique")
        indexed[identifier] = item
    return indexed


def _require_unique_subset(values: list[str], allowed: set[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{label} contain unknown values: {unknown}")


def _pixel_box(box: NormalizedBox, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return (
        round(box.x * (width - 1)),
        round(box.y * (height - 1)),
        round((box.x + box.width) * (width - 1)),
        round((box.y + box.height) * (height - 1)),
    )


def _draw_line(
    draw: ImageDraw.ImageDraw,
    line: ReviewLine,
    size: tuple[int, int],
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    image_width, image_height = size
    draw.line(
        (
            round(line.start.x * (image_width - 1)),
            round(line.start.y * (image_height - 1)),
            round(line.end.x * (image_width - 1)),
            round(line.end.y * (image_height - 1)),
        ),
        fill=color,
        width=width,
    )


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    line: ReviewLine,
    size: tuple[int, int],
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    image_width, image_height = size
    start = (
        round(line.start.x * (image_width - 1)),
        round(line.start.y * (image_height - 1)),
    )
    end = (
        round(line.end.x * (image_width - 1)),
        round(line.end.y * (image_height - 1)),
    )
    distance = math.dist(start, end)
    if distance <= 0.0:
        return
    step_count = max(1, math.ceil(distance / 7.0))
    for index in range(0, step_count, 2):
        first_t = index / step_count
        second_t = min(1.0, (index + 1) / step_count)
        draw.line(
            (
                round(start[0] + (end[0] - start[0]) * first_t),
                round(start[1] + (end[1] - start[1]) * first_t),
                round(start[0] + (end[0] - start[0]) * second_t),
                round(start[1] + (end[1] - start[1]) * second_t),
            ),
            fill=color,
            width=width,
        )


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


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        lower, higher = sorted((first_root, second_root))
        self._parent[higher] = lower
