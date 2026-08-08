"""Versioned, source-neutral contracts for P1 camera measurements."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.schemas import Point


class CameraEstimateStatus(str, Enum):
    """Whether a requested camera estimate was actually obtained."""

    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class CameraModel(str, Enum):
    UNKNOWN = "unknown"
    PINHOLE = "pinhole"
    RADIAL = "radial"
    DIVISIONAL = "divisional"
    FISHEYE = "fisheye"
    PANORAMIC = "panoramic"


class IntrinsicKind(str, Enum):
    VFOV_DEG = "vfov_deg"
    FOCAL_PX = "focal_px"


class CoordinateSpace(str, Enum):
    INPUT = "input"
    CANONICAL = "canonical"


class CameraConsistencyObservation(str, Enum):
    """Measurement availability, deliberately not a source-direction label."""

    MEASURED = "measured"
    NOT_OBSERVED = "not_observed"


class Matrix3x3(BaseModel):
    """A homogeneous affine transform written in row-major order."""

    model_config = ConfigDict(frozen=True)

    values: tuple[float, float, float, float, float, float, float, float, float]

    @classmethod
    def translation(cls, x: float, y: float) -> "Matrix3x3":
        return cls(values=(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0))

    def map_point(self, point: Point) -> Point:
        a, b, c, d, e, f, g, h, i = self.values
        denominator = g * point.x + h * point.y + i
        if abs(denominator) < 1e-12:
            raise ValueError("Affine transform maps point to infinity.")
        return Point(
            x=(a * point.x + b * point.y + c) / denominator,
            y=(d * point.x + e * point.y + f) / denominator,
        )

    @property
    def isotropic_scale(self) -> float:
        """Return the scale for focal-length conversion when the transform is affine."""
        a, b, _, d, e, _, g, h, _ = self.values
        if abs(g) > 1e-12 or abs(h) > 1e-12:
            raise ValueError("Perspective transforms cannot map focal length directly.")
        scale_x = (a * a + d * d) ** 0.5
        scale_y = (b * b + e * e) ** 0.5
        if abs(scale_x - scale_y) > 1e-6:
            raise ValueError("Anisotropic transform cannot map focal length directly.")
        return (scale_x + scale_y) / 2.0


class CropSpec(BaseModel):
    """One unresized crop and its mandatory mapping back to canonical pixels."""

    model_config = ConfigDict(frozen=True)

    crop_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crop_to_canonical: Matrix3x3
    requested_center: Point
    clamped_to_image_bounds: bool = False


class HorizonLine(BaseModel):
    """Two finite points on a horizon line in the estimate coordinate space."""

    model_config = ConfigDict(frozen=True)

    p1: Point
    p2: Point


class FieldOfViewOrFocal(BaseModel):
    """Either vertical FOV in degrees or focal length in the current pixel frame."""

    model_config = ConfigDict(frozen=True)

    kind: IntrinsicKind
    value: float = Field(gt=0.0)
    reference: Literal["camera", "input_frame"] = "camera"


class CameraUncertainty(BaseModel):
    """Normalised backend uncertainty; it is not a source-confidence score."""

    model_config = ConfigDict(frozen=True)

    overall: float | None = Field(default=None, ge=0.0, le=1.0)
    roll_rad: float | None = Field(default=None, ge=0.0)
    pitch_rad: float | None = Field(default=None, ge=0.0)
    vfov_deg: float | None = Field(default=None, ge=0.0)
    focal_px: float | None = Field(default=None, ge=0.0)
    principal_point_px: float | None = Field(default=None, ge=0.0)
    horizon_px: float | None = Field(default=None, ge=0.0)


class CameraBackendProvenance(BaseModel):
    """Reproducibility record required for every P1 backend response."""

    model_config = ConfigDict(frozen=True)

    backend_id: str
    backend_version: str | None = None
    model_commit: str | None = None
    weights_sha256: str | None = None
    weights_license: str | None = None
    inference_device: str
    requested_inference_device: str | None = None
    elapsed_ms: float = Field(ge=0.0)


class CameraEstimate(BaseModel):
    """Uniform P1 camera-estimation contract.

    Angles are radians; principal point and horizon coordinates are pixels in
    ``coordinate_space``.  ``vfov_or_focal`` names its own unit explicitly.
    The required uniform fields are intentionally present even when a backend
    cannot run, so unavailable measurements never resemble zero-valued ones.
    """

    model_config = ConfigDict(frozen=True)

    status: CameraEstimateStatus
    camera_model: CameraModel = CameraModel.UNKNOWN
    roll: float | None = None
    pitch: float | None = None
    vfov_or_focal: FieldOfViewOrFocal | None = None
    principal_point: Point | None = None
    horizon: HorizonLine | None = None
    uncertainty: CameraUncertainty = Field(default_factory=CameraUncertainty)
    applicability: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    provenance: CameraBackendProvenance
    coordinate_space: CoordinateSpace = CoordinateSpace.INPUT

    @model_validator(mode="after")
    def measurements_match_status(self) -> "CameraEstimate":
        values_present = any(
            value is not None
            for value in (
                self.roll,
                self.pitch,
                self.vfov_or_focal,
                self.principal_point,
                self.horizon,
            )
        )
        if self.status is CameraEstimateStatus.OK and self.camera_model is CameraModel.UNKNOWN:
            raise ValueError("Successful estimates must declare a camera_model.")
        if self.status is not CameraEstimateStatus.OK and values_present:
            raise ValueError("Unavailable or failed estimates must not carry measurements.")
        return self


class CameraQualityGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_applicability: float = Field(default=0.5, ge=0.0, le=1.0)
    min_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    max_uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    min_qualified_crops: int = Field(default=3, ge=3)


class CameraCropProtocolConfig(BaseModel):
    """Frozen candidate crop protocol from the P1 blueprint."""

    model_config = ConfigDict(frozen=True)

    crop_count: Literal[4, 6, 8] = 6
    side_fraction_of_short_edge: float = Field(default=0.52, ge=0.45, le=0.70)
    target_overlap_fraction: float = Field(default=0.25, ge=0.15, le=0.35)


class CameraBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["perspective_fields", "geocalib"]
    module_name: str
    weights_path: str | None = None
    expected_weights_sha256: str | None = None
    model_commit: str | None = None
    weights_license: str | None = None
    inference_device: Literal["auto", "cpu", "cuda"] = "auto"
    geocalib_camera_model: Literal[
        "pinhole", "simple_radial", "radial", "simple_divisional"
    ] = "pinhole"
    perspective_fields_model_version: Literal[
        "Paramnet-360Cities-edina-uncentered",
    ] = "Paramnet-360Cities-edina-uncentered"


class P1CameraConfig(BaseModel):
    """Configuration isolated from P0's frozen geometry configuration."""

    model_config = ConfigDict(frozen=True)

    config_version: str
    camera_backend: CameraBackendConfig
    crop_protocol: CameraCropProtocolConfig = Field(default_factory=CameraCropProtocolConfig)
    quality_gate: CameraQualityGateConfig = Field(default_factory=CameraQualityGateConfig)


class CameraInputSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    sha256: str
    original_filename: str
    canonical_size: tuple[int, int]


class CameraCropResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    crop: CropSpec
    estimate: CameraEstimate
    qualified_for_e_cam: bool
    exclusion_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def gate_result_is_self_consistent(self) -> "CameraCropResult":
        if self.qualified_for_e_cam and self.exclusion_reasons:
            raise ValueError("Qualified crops must not include exclusion reasons.")
        if not self.qualified_for_e_cam and not self.exclusion_reasons:
            raise ValueError("Excluded crops must record at least one exclusion reason.")
        return self


class CameraConsistencyMeasurement(BaseModel):
    """Uncalibrated global--local camera-consistency measurement only."""

    model_config = ConfigDict(frozen=True)

    observation: CameraConsistencyObservation
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    qualified_crop_ids: list[str] = Field(default_factory=list)
    required_qualified_crops: int = Field(ge=3)
    component_means: dict[str, float] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def observation_matches_value(self) -> "CameraConsistencyMeasurement":
        if self.observation is CameraConsistencyObservation.MEASURED:
            if self.value is None:
                raise ValueError("Measured E_cam observations require a value.")
            if len(self.qualified_crop_ids) < self.required_qualified_crops:
                raise ValueError("Measured E_cam observations require enough qualified crops.")
        elif self.value is not None:
            raise ValueError("Unobserved E_cam measurements must not include a value.")
        return self


class NumericDistributionSummary(BaseModel):
    """Descriptive statistics for a measurement cohort, not a decision model."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None

    @model_validator(mode="after")
    def values_match_count(self) -> "NumericDistributionSummary":
        values = (self.minimum, self.maximum, self.mean, self.p50, self.p90)
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("Empty distributions must not include summary values.")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("Non-empty distributions require all summary values.")
        if self.count > 0:
            assert self.minimum is not None
            assert self.maximum is not None
            assert self.mean is not None
            assert self.p50 is not None
            assert self.p90 is not None
            tolerance = 1e-12
            if not (
                self.minimum - tolerance
                <= self.p50
                <= self.p90
                <= self.maximum + tolerance
            ):
                raise ValueError("Distribution quantiles must lie between minimum and maximum.")
            if not self.minimum - tolerance <= self.mean <= self.maximum + tolerance:
                raise ValueError("Distribution mean must lie between minimum and maximum.")
        return self


class CameraGateAudit(BaseModel):
    """Descriptive gate outcomes across a calibration cohort."""

    model_config = ConfigDict(frozen=True)

    estimate_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    qualified_count: int = Field(ge=0)
    exclusion_reason_counts: dict[str, int] = Field(default_factory=dict)
    metric_distributions: dict[str, NumericDistributionSummary] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def counts_are_self_consistent(self) -> "CameraGateAudit":
        if any(value < 0 for value in self.status_counts.values()):
            raise ValueError("Status counts must be non-negative.")
        if sum(self.status_counts.values()) != self.estimate_count:
            raise ValueError("Status counts must total estimate_count.")
        if not 0 <= self.qualified_count <= self.estimate_count:
            raise ValueError("qualified_count must lie within estimate_count.")
        if any(value < 0 for value in self.exclusion_reason_counts.values()):
            raise ValueError("Exclusion reason counts must be non-negative.")
        if any(
            distribution.count > self.estimate_count
            for distribution in self.metric_distributions.values()
        ):
            raise ValueError("Metric distribution counts cannot exceed estimate_count.")
        return self


class CameraCalibrationBackendIdentity(BaseModel):
    """Stable backend identity shared by every result in one cohort."""

    model_config = ConfigDict(frozen=True)

    backend_id: str
    backend_version: str | None = None
    model_commit: str | None = None
    weights_sha256: str | None = None
    weights_license: str | None = None
    inference_devices: list[str] = Field(default_factory=list)
    requested_inference_devices: list[str] = Field(default_factory=list)


class CameraCalibrationDecisionState(str, Enum):
    """Whether a cohort summary can be treated as a registered decision rule."""

    DESCRIPTIVE_ONLY = "descriptive_only"


class CameraCalibrationDecisionReadiness(BaseModel):
    """Explicitly prevent cohort summaries from becoming source decisions."""

    model_config = ConfigDict(frozen=True)

    state: CameraCalibrationDecisionState = CameraCalibrationDecisionState.DESCRIPTIVE_ONLY
    e_cam_decision_threshold: None = None
    requirements_before_registration: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class CameraCalibrationSummary(BaseModel):
    """Versioned, source-neutral summary for P1 gate calibration work."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "camera-calibration-summary-v1"
    created_at_utc: str
    cohort_name: str = Field(min_length=1)
    config_version: str
    config_digest: str
    requested_backend: str
    operational_quality_gate: CameraQualityGateConfig
    result_count: int = Field(ge=1)
    unique_image_count: int = Field(ge=1)
    result_filenames: list[str] = Field(default_factory=list)
    backend_identity: CameraCalibrationBackendIdentity
    full_image_gate: CameraGateAudit
    crop_gate: CameraGateAudit
    e_cam_observation_counts: dict[str, int] = Field(default_factory=dict)
    e_cam_value_distribution: NumericDistributionSummary
    decision_readiness: CameraCalibrationDecisionReadiness = Field(
        default_factory=CameraCalibrationDecisionReadiness
    )
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def counts_are_self_consistent(self) -> "CameraCalibrationSummary":
        if self.unique_image_count > self.result_count:
            raise ValueError("unique_image_count cannot exceed result_count.")
        if self.result_filenames and len(self.result_filenames) != self.result_count:
            raise ValueError("result_filenames must be empty or align with result_count.")
        if self.full_image_gate.estimate_count != self.result_count:
            raise ValueError("full_image_gate must contain one estimate per result.")
        if sum(self.e_cam_observation_counts.values()) != self.result_count:
            raise ValueError("E_cam observation counts must total result_count.")
        measured_count = self.e_cam_observation_counts.get(
            CameraConsistencyObservation.MEASURED.value,
            0,
        )
        if self.e_cam_value_distribution.count != measured_count:
            raise ValueError("E_cam value count must equal measured observation count.")
        return self


class CameraExperimentRunInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at_utc: str
    config_version: str
    config_digest: str
    requested_backend: str


class CameraExperimentResult(BaseModel):
    """P1 result with no source-direction or AI-probability fields."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "camera-consistency-v1"
    run: CameraExperimentRunInfo
    input: CameraInputSummary
    full_image: CameraEstimate
    crops: list[CameraCropResult]
    e_cam: CameraConsistencyMeasurement
    limitations: list[str] = Field(default_factory=list)
