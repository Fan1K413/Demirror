"""Versioned contracts for the geometry relationship classifier."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DenseLayerArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    weights: list[list[float]]
    bias: list[float]
    activation: Literal["relu", "identity"]

    @model_validator(mode="after")
    def _valid_dimensions(self) -> "DenseLayerArtifact":
        if not self.weights or not self.bias:
            raise ValueError("dense layer weights and bias must not be empty")
        width = len(self.bias)
        if any(len(row) != width for row in self.weights):
            raise ValueError("every dense layer row must match the bias width")
        return self


class GeometryRelationshipModel(BaseModel):
    """Portable NumPy MLP plus an independently fitted Platt calibrator."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "geometry-relationship-model-v1"
    model_version: str
    feature_schema_version: str
    feature_names: list[str] = Field(min_length=1)
    standardizer_mean: list[float] = Field(min_length=1)
    standardizer_scale: list[float] = Field(min_length=1)
    layers: list[DenseLayerArtifact] = Field(min_length=1)
    platt_coefficient: float
    platt_intercept: float
    ai_threshold: float = Field(ge=0.0, le=1.0)
    strong_ai_threshold: float = Field(ge=0.0, le=1.0)
    minimum_line_count: int = Field(default=8, ge=1)
    target_definition: str
    dataset_protocol: dict[str, object]
    evaluation: dict[str, object]
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_model(self) -> "GeometryRelationshipModel":
        width = len(self.feature_names)
        if len(self.standardizer_mean) != width or len(self.standardizer_scale) != width:
            raise ValueError("standardizer arrays must match feature_names")
        if any(value <= 0.0 for value in self.standardizer_scale):
            raise ValueError("standardizer_scale must contain positive values")
        if len(self.layers[0].weights) != width:
            raise ValueError("first dense layer input does not match feature schema")
        for first, second in zip(self.layers, self.layers[1:]):
            if len(first.bias) != len(second.weights):
                raise ValueError("adjacent dense layers have incompatible dimensions")
        if len(self.layers[-1].bias) != 1:
            raise ValueError("final dense layer must emit one logit")
        if self.strong_ai_threshold < self.ai_threshold:
            raise ValueError("strong_ai_threshold must not be below ai_threshold")
        return self


class GeometryRelationshipResult(BaseModel):
    """One calibrated geometry signal with an explicit applicability gate."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "geometry-relationship-result-v1"
    status: Literal["available", "not_applicable", "unavailable", "failed"]
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_band: Literal["high", "medium", "low", "unknown"] = "unknown"
    applicability: float = Field(default=0.0, ge=0.0, le=1.0)
    line_count: int = Field(default=0, ge=0)
    decision_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    strong_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str | None = None
    summary: str
    findings: list[str] = Field(default_factory=list)
    evaluation: dict[str, object] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
