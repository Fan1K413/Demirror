"""Offline G1-G4 replay conditioned on frozen deterministic coarse surfaces."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.geometry_ai.consistency_v2 import measure_consistency_checks
from image_trust.geometry_ai.deterministic_surfaces import (
    DeterministicSurfaceBaselineResult,
)
from image_trust.geometry_ai.measurement_types import (
    CanonicalBox,
    GeometryCheckV2,
    GeometryFamilyV2,
    GeometryMeasurementV2Result,
    MergedGeometryLineV2,
    StructureRegionV2,
)
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationReviewPacket,
)


REPLAY_AUTHORIZATION_SCHEMA_VERSION = (
    "geometry-surface-conditioned-replay-authorization-v1"
)
REPLAY_SCHEMA_VERSION = "geometry-surface-conditioned-g1-g4-v1"


class _FrozenReplayContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SurfaceConditionedReplayAuthorization(_FrozenReplayContract):
    schema_version: Literal[REPLAY_AUTHORIZATION_SCHEMA_VERSION] = (
        REPLAY_AUTHORIZATION_SCHEMA_VERSION
    )
    continuation_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_annotation_hash_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    g1_g4_measurement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_ids: list[str] = Field(min_length=1)
    comparison_passed: Literal[True] = True
    origin_scoring_authorized: Literal[False] = False
    web_integration_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _unique_reviewers(self) -> "SurfaceConditionedReplayAuthorization":
        if self.reviewer_ids != sorted(set(self.reviewer_ids)):
            raise ValueError("replay reviewer IDs must be unique and sorted")
        return self


class SurfaceConditionedReplayResult(_FrozenReplayContract):
    schema_version: Literal[REPLAY_SCHEMA_VERSION] = REPLAY_SCHEMA_VERSION
    status: Literal["available", "not_applicable"]
    summary: str
    reviewer_id: str
    canonical_size: tuple[int, int]
    authorization_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measurement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conditioned_regions: list[StructureRegionV2] = Field(default_factory=list)
    conditioned_families: list[GeometryFamilyV2] = Field(default_factory=list)
    checks: list[GeometryCheckV2] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_labels_used: Literal[False] = False
    human_annotations_used: Literal[False] = False
    origin_scoring_authorized: Literal[False] = False
    web_integration_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _check_contract(self) -> "SurfaceConditionedReplayResult":
        if self.canonical_size[0] <= 0 or self.canonical_size[1] <= 0:
            raise ValueError("surface-conditioned canonical size must be positive")
        if self.status == "not_applicable":
            if self.conditioned_regions or self.conditioned_families or self.checks:
                raise ValueError("not-applicable replay must not publish partial checks")
            return self
        if [check.check_id for check in self.checks] != ["G1", "G2", "G3", "G4"]:
            raise ValueError("surface-conditioned replay must contain exactly G1-G4")
        if any(check.origin_eligible for check in self.checks):
            raise ValueError("surface-conditioned checks cannot be origin eligible")
        region_ids = {region.region_id for region in self.conditioned_regions}
        line_ids = {
            line_id for region in self.conditioned_regions for line_id in region.line_ids
        }
        for family in self.conditioned_families:
            if family.region_id != "global" and family.region_id not in region_ids:
                raise ValueError("conditioned family references an unknown surface scope")
            if not set(family.member_line_ids) <= line_ids:
                raise ValueError("conditioned family references lines outside surface scopes")
        return self


def build_surface_replay_authorization(
    comparison_report: dict[str, object],
    *,
    comparison_report_sha256: str,
    g1_g4_measurement_sha256: str,
) -> SurfaceConditionedReplayAuthorization:
    """Validate the frozen comparison decision before constructing replay input."""

    if comparison_report.get("schema_version") != "geometry-surface-human-comparison-v1":
        raise ValueError("unexpected geometry surface comparison schema")
    if comparison_report.get("status") != "complete":
        raise ValueError("geometry surface comparison is not complete")
    if comparison_report.get("passed") is not True:
        raise ValueError("geometry surface comparison gates did not pass")
    if comparison_report.get("decision") != (
        "eligible_for_surface_conditioned_g1_g4_replay_only"
    ):
        raise ValueError("geometry surface comparison does not authorize replay")
    gates = comparison_report.get("gates")
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise ValueError("geometry surface comparison has an incomplete gate closure")
    for name in (
        "source_key_opened",
        "source_labels_used",
        "ai_assisted_annotations_used",
        "origin_scoring_authorized",
        "web_integration_authorized",
    ):
        if comparison_report.get(name) is not False:
            raise ValueError(f"comparison report violates source-neutral field {name}")
    reviewer_ids = comparison_report.get("replay_reviewer_ids")
    if not isinstance(reviewer_ids, list) or not reviewer_ids:
        raise ValueError("comparison report has no replay reviewer IDs")
    return SurfaceConditionedReplayAuthorization(
        continuation_protocol_sha256=str(
            comparison_report["continuation_protocol_sha256"]
        ),
        comparison_report_sha256=comparison_report_sha256,
        baseline_audit_sha256=str(comparison_report["baseline_audit_sha256"]),
        human_annotation_hash_closure_sha256=str(
            comparison_report["human_annotation_hash_closure_sha256"]
        ),
        quality_receipt_sha256=str(comparison_report["quality_receipt_sha256"]),
        g1_g4_measurement_sha256=g1_g4_measurement_sha256,
        reviewer_ids=sorted(str(value) for value in reviewer_ids),
    )


def assess_surface_conditioned_g1_g4(
    canonical_rgb: np.ndarray,
    measurement: GeometryMeasurementV2Result,
    packet: GeometryRelationReviewPacket,
    baseline: DeterministicSurfaceBaselineResult,
    authorization: SurfaceConditionedReplayAuthorization,
    *,
    check_callback: Callable[[list[GeometryCheckV2]], None] | None = None,
    check_started_callback: Callable[[str], None] | None = None,
) -> SurfaceConditionedReplayResult:
    """Replay frozen G1-G4 inside deterministic surface scopes only."""

    _validate_inputs(canonical_rgb, measurement, packet, baseline, authorization)
    common = {
        "reviewer_id": packet.reviewer_id,
        "canonical_size": packet.canonical_size,
        "authorization_report_sha256": authorization.comparison_report_sha256,
        "baseline_result_sha256": _canonical_model_hash(baseline),
        "measurement_sha256": _canonical_model_hash(measurement),
    }
    if measurement.status != "measurable" or not baseline.surface_candidates:
        return SurfaceConditionedReplayResult(
            status="not_applicable",
            summary="No authorized measurable deterministic surface scope is available.",
            limitations=[
                "surface_conditioned_replay_requires_measurable_geometry_and_surfaces",
                "surface_conditioned_replay_not_origin_evidence",
            ],
            **common,
        )

    line_by_id = {line.line_id: line for line in measurement.merged_lines}
    regions = _conditioned_regions(
        baseline,
        line_by_id,
        packet.canonical_size,
    )
    families = _conditioned_families(baseline, measurement, regions)
    measured = measure_consistency_checks(
        measurement.merged_lines,
        regions,
        families,
        canonical_rgb,
        check_callback=(
            (lambda checks: check_callback(_without_g5(checks)))
            if check_callback is not None
            else None
        ),
        check_started_callback=check_started_callback,
    )
    checks = [
        check.model_copy(
            update={
                "origin_eligible": False,
                "limitations": [
                    *check.limitations,
                    "surface_conditioned_deterministic_scope_not_origin_evidence",
                ],
            }
        )
        for check in _without_g5(measured)
    ]
    return SurfaceConditionedReplayResult(
        status="available",
        summary="Frozen G1-G4 replay completed inside deterministic surface scopes.",
        conditioned_regions=regions,
        conditioned_families=families,
        checks=checks,
        limitations=[
            "surface_scopes_are_deterministic_candidates_not_semantic_truth",
            "original_family_model_fields_are_reused_with_surface_limited_support",
            "surface_conditioned_replay_not_origin_evidence",
        ],
        **common,
    )


def _validate_inputs(
    canonical_rgb: np.ndarray,
    measurement: GeometryMeasurementV2Result,
    packet: GeometryRelationReviewPacket,
    baseline: DeterministicSurfaceBaselineResult,
    authorization: SurfaceConditionedReplayAuthorization,
) -> None:
    if packet.reviewer_id not in authorization.reviewer_ids:
        raise ValueError("reviewer ID is not authorized by the human comparison")
    if baseline.reviewer_id != packet.reviewer_id:
        raise ValueError("deterministic baseline reviewer ID differs from its packet")
    if baseline.canonical_size != packet.canonical_size:
        raise ValueError("deterministic baseline size differs from its packet")
    if measurement.canonical_size != packet.canonical_size:
        raise ValueError("geometry measurement size differs from its packet")
    if canonical_rgb.ndim != 3 or canonical_rgb.shape[2] != 3:
        raise ValueError("surface-conditioned replay requires an RGB image")
    height, width = canonical_rgb.shape[:2]
    if (width, height) != packet.canonical_size:
        raise ValueError("surface-conditioned image size differs from its packet")
    packet_line_ids = [line.line_id for line in packet.lines]
    measurement_line_ids = [line.line_id for line in measurement.merged_lines]
    baseline_line_ids = [line.line_id for line in baseline.line_assignments]
    if packet_line_ids != measurement_line_ids or packet_line_ids != baseline_line_ids:
        raise ValueError("packet, measurement and baseline line order must match")
    packet_family_ids = [family.family_id for family in packet.family_proposals]
    baseline_family_ids = [
        partition.original_family_id for partition in baseline.family_partitions
    ]
    if packet_family_ids != baseline_family_ids:
        raise ValueError("packet and baseline family order must match")
    measurement_family_list = [family.family_id for family in measurement.families]
    if len(measurement_family_list) != len(set(measurement_family_list)):
        raise ValueError("geometry measurement family IDs must be unique")
    measurement_family_ids = set(measurement_family_list)
    if not set(packet_family_ids) <= measurement_family_ids:
        raise ValueError("packet families are missing from geometry measurement")
    if baseline.source_labels_used or baseline.origin_scoring_authorized:
        raise ValueError("deterministic baseline unexpectedly authorizes source scoring")


def _conditioned_regions(
    baseline: DeterministicSurfaceBaselineResult,
    line_by_id: dict[str, MergedGeometryLineV2],
    canonical_size: tuple[int, int],
) -> list[StructureRegionV2]:
    width, height = canonical_size
    regions: list[StructureRegionV2] = []
    for surface in baseline.surface_candidates:
        lines = [line_by_id[line_id] for line_id in surface.member_line_ids]
        box = surface.coarse_box
        x0 = min(width - 1, max(0, int(math.floor(box.x * width))))
        y0 = min(height - 1, max(0, int(math.floor(box.y * height))))
        x1 = min(width, max(x0 + 1, int(math.ceil((box.x + box.width) * width))))
        y1 = min(height, max(y0 + 1, int(math.ceil((box.y + box.height) * height))))
        regions.append(
            StructureRegionV2(
                region_id=surface.surface_id,
                canonical_box=CanonicalBox(
                    x=x0,
                    y=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                ),
                cell_ids=[],
                line_ids=list(surface.member_line_ids),
                line_count=len(lines),
                normalized_line_support=float(
                    sum(
                        line.length_normalized * line.cross_scale_stability
                        for line in lines
                    )
                ),
                orientation_entropy=_orientation_entropy(lines),
                status="usable",
            )
        )
    return regions


def _conditioned_families(
    baseline: DeterministicSurfaceBaselineResult,
    measurement: GeometryMeasurementV2Result,
    regions: list[StructureRegionV2],
) -> list[GeometryFamilyV2]:
    original_by_id = {family.family_id: family for family in measurement.families}
    region_line_ids = {
        region.region_id: set(region.line_ids) for region in regions
    }
    conditioned: list[GeometryFamilyV2] = []
    for partition in baseline.family_partitions:
        original = original_by_id[partition.original_family_id]
        for subfamily in partition.surface_subfamilies:
            if not set(subfamily.member_line_ids) <= region_line_ids[subfamily.surface_id]:
                raise ValueError("surface subfamily is outside its deterministic scope")
            conditioned.append(
                GeometryFamilyV2(
                    family_id=_stable_id(
                        "scf",
                        [partition.original_family_id, subfamily.surface_id],
                    ),
                    region_id=(
                        "global"
                        if original.region_id == "global"
                        else subfamily.surface_id
                    ),
                    kind=original.kind,
                    member_line_ids=list(subfamily.member_line_ids),
                    direction_rad=original.direction_rad,
                    vanishing_point=original.vanishing_point,
                    weighted_inlier_ratio=original.weighted_inlier_ratio,
                    residual_p50_deg=original.residual_p50_deg,
                    residual_p90_deg=original.residual_p90_deg,
                    bootstrap_stability=original.bootstrap_stability,
                    stable=original.stable and subfamily.usability == "usable",
                )
            )
    return conditioned


def _orientation_entropy(lines: list[MergedGeometryLineV2]) -> float:
    if not lines:
        return 0.0
    values = np.asarray([line.angle_rad for line in lines], dtype=np.float64)
    weights = np.asarray(
        [line.length_normalized * line.cross_scale_stability for line in lines],
        dtype=np.float64,
    )
    histogram, _ = np.histogram(values, bins=12, range=(0.0, math.pi), weights=weights)
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.0
    probability = histogram[histogram > 0.0] / total
    return float(-np.sum(probability * np.log(probability)) / math.log(12.0))


def _without_g5(checks: list[GeometryCheckV2]) -> list[GeometryCheckV2]:
    return [check for check in checks if check.check_id in {"G1", "G2", "G3", "G4"}]


def _stable_id(prefix: str, values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16]}"


def _canonical_model_hash(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
