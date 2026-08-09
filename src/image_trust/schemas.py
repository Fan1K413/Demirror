"""Versioned data contracts for the P0 pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class Observation(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NOT_OBSERVED = "not_observed"


class Direction(str, Enum):
    SUPPORTS_AI = "supports_ai"
    SUPPORTS_CAMERA = "supports_camera"
    NEUTRAL = "neutral"
    CONFLICTING = "conflicting"


class Point(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class CoordinateTransformSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: str = "canonical-image-v1"
    encoded_size: tuple[int, int]
    canonical_size: tuple[int, int]
    analysis_size: tuple[int, int]
    exif_orientation: int = 1
    orientation_applied: bool = False
    encoded_to_canonical: str = "identity"
    scale_x: float
    scale_y: float


class InputSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    sha256: str
    detected_format: str
    original_filename: str
    filename_format_mismatch: bool
    file_size_bytes: int
    encoded_size: tuple[int, int]
    canonical_size: tuple[int, int]
    analysis_size: tuple[int, int]
    pixel_count: int
    color_mode: str
    validation_limits: dict[str, int | bool] = Field(default_factory=dict)
    coordinate_transform: CoordinateTransformSummary
    exif_summary: dict[str, Any] = Field(default_factory=dict)


class LineRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    line_id: str
    p1_analysis: Point
    p2_analysis: Point
    p1: Point
    p2: Point
    length_analysis: float
    length: float
    angle_rad: float
    quality: float = Field(ge=0.0, le=1.0)
    backend_features: dict[str, float | None] = Field(default_factory=dict)
    selected: bool


class VPFamily(BaseModel):
    model_config = ConfigDict(frozen=True)

    family_id: str
    vp_type: str
    vp_analysis: Point | None = None
    vp: Point | None = None
    direction_analysis: float | None = None
    member_line_ids: list[str]
    weighted_inlier_ratio: float = Field(ge=0.0, le=1.0)
    weighted_median_residual_deg: float
    spatial_support: float = Field(ge=0.0, le=1.0)
    bootstrap_stability: float = Field(ge=0.0, le=1.0)
    residual_quantiles_deg: dict[str, float]
    stable: bool
    scope: str = "global_vp"
    spatial_window_analysis: tuple[float, float, float, float] | None = None


class AnomalyCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    line_id: str
    anomaly_candidate_score: float = Field(ge=0.0, le=1.0)
    nearest_family_id: str | None = None
    residual_deg: float | None = None
    reason: str


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    module: str = "geometry.vanishing_points"
    run_status: RunStatus
    observation: Observation
    direction: Direction
    raw_score: float | None = None
    applicability: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    features: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    model_version: str = "geometry-p0"


class RunInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at_utc: str
    config_version: str
    config_digest: str | None = None
    deterministic_seed: int | None = None
    requested_backend: str
    resolved_backend: str | None = None
    fallback_reason: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    runtime_environment: dict[str, str] = Field(default_factory=dict)


class Diagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    timing_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "evidence-result-v1"
    run: RunInfo
    input: InputSummary | None = None
    evidence: Evidence
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)


class IngestConfig(BaseModel):
    max_file_bytes: int = Field(default=52_428_800, ge=1)
    max_pixels: int = Field(default=40_000_000, ge=1)
    min_side_px: int = Field(default=64, ge=1)
    allow_filename_format_mismatch: bool = True


class AnalysisConfig(BaseModel):
    max_long_side: int = Field(default=1280, ge=1024, le=1536)


class LineBackendConfig(BaseModel):
    name: str = "opencv_lsd"
    allow_fallback: bool = False
    deeplsd_weights: str | None = None
    deeplsd_max_side: int = Field(default=512, ge=128, le=1536)
    deeplsd_threads: int = Field(default=2, ge=1, le=4)
    deeplsd_timeout_seconds: float = Field(default=45.0, ge=5.0, le=120.0)
    opencv_refine: str = "std"
    min_length_px: float = Field(default=24.0, ge=0.0)
    min_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    max_lines: int = Field(default=750, ge=1)
    suppress_curve_fragments: bool = True
    curve_max_segment_length_px: float = Field(default=96.0, gt=0.0)
    curve_neighbor_gap_px: float = Field(default=12.0, gt=0.0)
    curve_neighbor_angle_deg: float = Field(default=28.0, gt=0.0, lt=90.0)
    curve_min_component_lines: int = Field(default=5, ge=3)
    curve_min_direction_span_deg: float = Field(default=30.0, gt=0.0, lt=180.0)
    curve_min_contiguous_direction_bins: int = Field(default=4, ge=2, le=18)


class ApplicabilityConfig(BaseModel):
    target_line_count: int = Field(default=40, ge=1)
    target_total_length: float = Field(default=12.0, gt=0.0)
    target_spatial_coverage: float = Field(default=0.25, gt=0.0, le=1.0)
    min_line_count: int = Field(default=8, ge=0)
    min_total_length: float = Field(default=1.5, ge=0.0)
    min_spatial_coverage: float = Field(default=0.04, ge=0.0, le=1.0)
    grid_size: int = Field(default=8, ge=1)
    direction_bins: int = Field(default=18, ge=1)
    special_imaging_cap: float = Field(default=0.20, ge=0.0, le=1.0)
    anomaly_min_applicability: float = Field(default=0.45, ge=0.0, le=1.0)


class VanishingPointConfig(BaseModel):
    max_families: int = Field(default=4, ge=1)
    max_hypotheses: int = Field(default=500, ge=1)
    pair_min_angle_deg: float = Field(default=5.0, gt=0.0, lt=90.0)
    inlier_angle_deg: float = Field(default=2.5, gt=0.0, lt=90.0)
    min_family_lines: int = Field(default=4, ge=2)
    min_family_weight: float = Field(default=80.0, ge=0.0)
    bootstrap_rounds: int = Field(default=8, ge=0)
    bootstrap_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    family_jaccard_merge: float = Field(default=0.70, ge=0.0, le=1.0)
    stable_family_min_bootstrap: float = Field(default=0.50, ge=0.0, le=1.0)
    max_parallel_families: int = Field(default=6, ge=1)
    parallel_inlier_angle_deg: float = Field(default=2.5, gt=0.0, lt=90.0)
    competing_family_max_extent_ratio: float = Field(default=0.60, gt=0.0, le=1.0)
    compact_component_min_lines: int = Field(default=2, ge=2)
    compact_component_max_extent_ratio: float = Field(default=0.18, gt=0.0, le=1.0)
    compact_component_link_distance_ratio: float = Field(default=0.10, gt=0.0, le=1.0)
    compact_component_max_link_distance_px: float = Field(default=12.0, gt=0.0)
    compact_component_min_line_length_ratio: float = Field(default=0.06, gt=0.0, le=1.0)
    compact_component_max_family_inlier_ratio: float = Field(default=0.18, gt=0.0, le=1.0)
    compact_component_max_family_weight_ratio: float = Field(default=0.65, gt=0.0, le=1.0)
    compact_component_nearby_distance_ratio: float = Field(default=0.10, gt=0.0, le=1.0)
    compact_component_max_nearby_distance_px: float = Field(default=20.0, gt=0.0)
    unassigned_candidate_min_length_ratio: float = Field(default=0.08, gt=0.0, le=1.0)
    local_family_grid_size: int = Field(default=3, ge=1)
    local_families_per_cell: int = Field(default=2, ge=1)
    max_local_families: int = Field(default=8, ge=1)
    local_min_family_weight: float = Field(default=40.0, ge=0.0)
    local_direction_families_per_cell: int = Field(default=5, ge=1)
    local_direction_inlier_angle_deg: float = Field(default=5.0, gt=0.0, lt=90.0)
    local_direction_component_gap_ratio: float = Field(default=0.08, gt=0.0, le=1.0)
    local_direction_component_max_gap_px: float = Field(default=36.0, gt=0.0)
    local_direction_include_global_members: bool = False


class OverlayConfig(BaseModel):
    line_width: int = Field(default=2, ge=1)
    draw_line_ids: bool = False


class P0Config(BaseModel):
    config_version: str
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    line_backend: LineBackendConfig = Field(default_factory=LineBackendConfig)
    applicability: ApplicabilityConfig = Field(default_factory=ApplicabilityConfig)
    vanishing_points: VanishingPointConfig = Field(default_factory=VanishingPointConfig)
    overlays: OverlayConfig = Field(default_factory=OverlayConfig)
