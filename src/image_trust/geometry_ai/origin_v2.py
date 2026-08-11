"""Portable, interpretable origin candidate built on geometry-v2 measurements.

This module does not make geometry measurements and does not install a model.
It turns a completed :class:`GeometryMeasurementV2Result` into a fixed feature
vector and evaluates a JSON logistic-regression artifact.  Candidate artifacts
are blocked by default until the independent replacement gate marks them as
deployment eligible.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_trust.geometry_ai.measurement_types import GeometryMeasurementV2Result


FEATURE_SCHEMA_VERSION = "geometry-origin-features-v2"
MODEL_SCHEMA_VERSION = "geometry-origin-logistic-model-v2"

_CHECK_IDS = ("G1", "G2", "G3", "G4", "G5")
_CHECK_STATUSES = ("available", "not_applicable", "not_run", "failed")


def _feature_names() -> tuple[str, ...]:
    names = [
        "measurement_applicability",
        "gate_pass_ratio",
        "merged_line_count_log1p",
        "mean_cross_scale_stability",
        "stable_merged_line_ratio",
        "region_count_log1p",
        "usable_region_ratio",
        "family_count_log1p",
        "stable_family_ratio",
        "mean_family_bootstrap_stability",
        "mean_family_inlier_ratio",
        "mean_family_residual_p90_deg",
    ]
    for check_id in _CHECK_IDS:
        prefix = check_id.lower()
        names.extend(f"{prefix}_status_{status}" for status in _CHECK_STATUSES)
        names.extend(
            (
                f"{prefix}_anomaly_score",
                f"{prefix}_finding_count_log1p",
                f"{prefix}_max_finding_severity",
                f"{prefix}_mean_finding_severity",
            )
        )
    return tuple(names)


FEATURE_NAMES = _feature_names()


class GeometryOriginV2Model(BaseModel):
    """A standardized logistic regression with an explicit deployment lock."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_SCHEMA_VERSION
    model_version: str
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_names: list[str] = Field(min_length=1)
    standardizer_mean: list[float] = Field(min_length=1)
    standardizer_scale: list[float] = Field(min_length=1)
    coefficients: list[float] = Field(min_length=1)
    intercept: float
    decision_threshold: float = Field(ge=0.0, le=1.0)
    deployment_eligible: bool = False
    replacement_gate: dict[str, Any] = Field(default_factory=dict)
    training_protocol: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_dimensions_and_gate(self) -> "GeometryOriginV2Model":
        width = len(self.feature_names)
        if self.feature_names != list(FEATURE_NAMES):
            raise ValueError("feature_names must exactly match geometry-origin-features-v2")
        if any(len(values) != width for values in (self.standardizer_mean, self.standardizer_scale, self.coefficients)):
            raise ValueError("standardizer and coefficient arrays must match feature_names")
        if any(value <= 0.0 for value in self.standardizer_scale):
            raise ValueError("standardizer_scale must contain positive values")
        if self.deployment_eligible and self.replacement_gate.get("eligible") is not True:
            raise ValueError("deployment_eligible requires an embedded passing replacement gate")
        return self


class GeometryOriginV2Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "geometry-origin-prediction-v2"
    status: Literal["available", "candidate_only", "not_applicable", "unavailable"]
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_ai: bool | None = None
    decision_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str | None = None
    contributions: dict[str, float] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


def extract_geometry_origin_features(
    result: GeometryMeasurementV2Result | Mapping[str, Any],
) -> dict[str, float]:
    """Extract fixed G1--G5 availability, finding, region and family features."""

    measurement = (
        result
        if isinstance(result, GeometryMeasurementV2Result)
        else GeometryMeasurementV2Result.model_validate(dict(result))
    )
    merged_stability = [line.cross_scale_stability for line in measurement.merged_lines]
    usable_regions = [region for region in measurement.regions if region.status == "usable"]
    stable_families = [family for family in measurement.families if family.stable]
    features: dict[str, float] = {
        "measurement_applicability": float(measurement.applicability),
        "gate_pass_ratio": _ratio(sum(gate.passed for gate in measurement.gates), len(measurement.gates)),
        "merged_line_count_log1p": math.log1p(len(measurement.merged_lines)),
        "mean_cross_scale_stability": _mean(merged_stability),
        "stable_merged_line_ratio": _ratio(sum(value >= 0.60 for value in merged_stability), len(merged_stability)),
        "region_count_log1p": math.log1p(len(measurement.regions)),
        "usable_region_ratio": _ratio(len(usable_regions), len(measurement.regions)),
        "family_count_log1p": math.log1p(len(measurement.families)),
        "stable_family_ratio": _ratio(len(stable_families), len(measurement.families)),
        "mean_family_bootstrap_stability": _mean(
            [family.bootstrap_stability for family in measurement.families]
        ),
        "mean_family_inlier_ratio": _mean(
            [family.weighted_inlier_ratio for family in measurement.families]
        ),
        "mean_family_residual_p90_deg": _mean(
            [family.residual_p90_deg for family in measurement.families]
        ),
    }
    by_id = {check.check_id: check for check in measurement.checks}
    for check_id in _CHECK_IDS:
        prefix = check_id.lower()
        check = by_id.get(check_id)
        status = check.status if check is not None else "not_run"
        for possible in _CHECK_STATUSES:
            features[f"{prefix}_status_{possible}"] = float(status == possible)
        findings = check.findings if check is not None else []
        severities = [finding.severity for finding in findings]
        features[f"{prefix}_anomaly_score"] = float(check.anomaly_score or 0.0) if check else 0.0
        features[f"{prefix}_finding_count_log1p"] = math.log1p(len(findings))
        features[f"{prefix}_max_finding_severity"] = max(severities, default=0.0)
        features[f"{prefix}_mean_finding_severity"] = _mean(severities)
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("geometry origin feature order changed unexpectedly")
    return features


def predict_geometry_origin_v2(
    result: GeometryMeasurementV2Result | Mapping[str, Any],
    model: GeometryOriginV2Model | Path,
    *,
    allow_ineligible_candidate: bool = False,
) -> GeometryOriginV2Prediction:
    """Evaluate one model, refusing candidate-only artifacts by default."""

    artifact = _load_model(model)
    if artifact.feature_schema_version != FEATURE_SCHEMA_VERSION:
        return GeometryOriginV2Prediction(
            status="unavailable",
            model_version=artifact.model_version,
            limitations=["geometry_origin_v2_feature_schema_mismatch"],
        )
    if not artifact.deployment_eligible and not allow_ineligible_candidate:
        return GeometryOriginV2Prediction(
            status="candidate_only",
            decision_threshold=artifact.decision_threshold,
            model_version=artifact.model_version,
            limitations=["geometry_origin_v2_replacement_gate_not_passed"],
        )
    measurement = (
        result
        if isinstance(result, GeometryMeasurementV2Result)
        else GeometryMeasurementV2Result.model_validate(dict(result))
    )
    if measurement.status != "measurable":
        return GeometryOriginV2Prediction(
            status="not_applicable",
            decision_threshold=artifact.decision_threshold,
            model_version=artifact.model_version,
            limitations=["geometry_origin_v2_measurement_not_applicable"],
        )
    features = extract_geometry_origin_features(measurement)
    standardized = [
        (features[name] - mean) / scale
        for name, mean, scale in zip(
            artifact.feature_names,
            artifact.standardizer_mean,
            artifact.standardizer_scale,
        )
    ]
    terms = [coefficient * value for coefficient, value in zip(artifact.coefficients, standardized)]
    probability = _sigmoid(artifact.intercept + sum(terms))
    contributions = {
        name: float(term)
        for name, term in sorted(
            zip(artifact.feature_names, terms),
            key=lambda pair: (-abs(pair[1]), pair[0]),
        )
    }
    return GeometryOriginV2Prediction(
        status="available" if artifact.deployment_eligible else "candidate_only",
        probability=probability,
        predicted_ai=probability >= artifact.decision_threshold,
        decision_threshold=artifact.decision_threshold,
        model_version=artifact.model_version,
        contributions=contributions,
        limitations=[] if artifact.deployment_eligible else ["offline_candidate_evaluation_only"],
    )


def _load_model(model: GeometryOriginV2Model | Path) -> GeometryOriginV2Model:
    if isinstance(model, GeometryOriginV2Model):
        return model
    return GeometryOriginV2Model.model_validate_json(model.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
