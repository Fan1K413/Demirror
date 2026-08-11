"""Versioned contracts for the explainable geometry-v2 measurement chain."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "geometry-measurement-v2"


class CanonicalBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class GeometryPointV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class GeometryLineV2(BaseModel):
    """One LSD line mapped to EXIF-normalized source coordinates."""

    model_config = ConfigDict(frozen=True)

    line_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    length_px: float = Field(gt=0.0)
    length_normalized: float = Field(gt=0.0)


class GeometryScaleV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    scale_id: str
    scope: Literal["global", "local_tile"]
    canonical_crop: CanonicalBox
    analysis_size: tuple[int, int]
    line_count: int = Field(ge=0)
    normalized_total_length: float = Field(ge=0.0)
    lines: list[GeometryLineV2] = Field(default_factory=list)


class MergedGeometryLineV2(BaseModel):
    """A canonical representative shared by one or more scale detections."""

    model_config = ConfigDict(frozen=True)

    line_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    length_px: float = Field(gt=0.0)
    length_normalized: float = Field(gt=0.0)
    angle_rad: float = Field(ge=0.0)
    source_line_ids: list[str] = Field(default_factory=list)
    source_scale_ids: list[str] = Field(default_factory=list)
    cross_scale_stability: float = Field(ge=0.0, le=1.0)


class GeometryGateV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    gate_id: str
    passed: bool
    observed: float
    threshold: float
    description: str


class StructureRegionV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    region_id: str
    canonical_box: CanonicalBox
    cell_ids: list[str]
    line_ids: list[str]
    line_count: int = Field(ge=0)
    normalized_line_support: float = Field(ge=0.0)
    orientation_entropy: float = Field(ge=0.0, le=1.0)
    status: Literal["usable", "insufficient_support"]


class GeometryFamilyV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    family_id: str
    region_id: str
    kind: Literal["parallel", "finite_vp", "infinite_vp"]
    member_line_ids: list[str]
    direction_rad: float | None = None
    vanishing_point: GeometryPointV2 | None = None
    weighted_inlier_ratio: float = Field(ge=0.0, le=1.0)
    residual_p50_deg: float = Field(ge=0.0)
    residual_p90_deg: float = Field(ge=0.0)
    bootstrap_stability: float = Field(ge=0.0, le=1.0)
    stable: bool


class GeometryFindingV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    finding_id: str
    check_id: Literal["G1", "G2", "G3", "G4", "G5"]
    region_ids: list[str] = Field(default_factory=list)
    family_ids: list[str] = Field(default_factory=list)
    line_ids: list[str] = Field(default_factory=list)
    severity: float = Field(ge=0.0, le=1.0)
    measured_value: float | None = None
    reference_value: float | None = None
    description: str


class GeometryCheckV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    check_id: Literal["G1", "G2", "G3", "G4", "G5"]
    title: str
    status: Literal["available", "not_applicable", "not_run", "failed"]
    anomaly_score: float | None = Field(default=None, ge=0.0, le=1.0)
    origin_eligible: bool = False
    measurements: dict[str, Any] = Field(default_factory=dict)
    findings: list[GeometryFindingV2] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class GeometryArtifactsV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    result_json: str | None = None
    regions_overlay: str | None = None
    families_overlay: str | None = None
    consistency_overlay: str | None = None
    repeat_spacing_overlay: str | None = None
    perspective_fields_result: str | None = None


class GeometryMeasurementV2Result(BaseModel):
    """Source-neutral geometry result; it deliberately has no AI probability."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    status: Literal["measurable", "not_applicable", "failed"]
    summary: str
    canonical_size: tuple[int, int] = (0, 0)
    applicability: float = Field(default=0.0, ge=0.0, le=1.0)
    gates: list[GeometryGateV2] = Field(default_factory=list)
    global_scale: GeometryScaleV2 | None = None
    local_scales: list[GeometryScaleV2] = Field(default_factory=list)
    merged_lines: list[MergedGeometryLineV2] = Field(default_factory=list)
    regions: list[StructureRegionV2] = Field(default_factory=list)
    families: list[GeometryFamilyV2] = Field(default_factory=list)
    checks: list[GeometryCheckV2] = Field(default_factory=list)
    artifacts: GeometryArtifactsV2 = Field(default_factory=GeometryArtifactsV2)
    limitations: list[str] = Field(default_factory=list)
