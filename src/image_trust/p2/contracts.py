"""Explicit contracts for the narrow P2 experimental classifier.

P2 is deliberately separate from P0.  P0 remains a source-neutral geometry
measurement; P2 consumes a fixed P0 feature vector and a versioned local model
artifact to return a probability only for the registered benchmark population.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class P2ModelArtifact(BaseModel):
    """JSON-serializable logistic model and its calibration record."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "p2-model-artifact-v1"
    model_version: str
    feature_names: list[str] = Field(min_length=1)
    standardizer_mean: list[float] = Field(min_length=1)
    standardizer_scale: list[float] = Field(min_length=1)
    base_coefficients: list[float] = Field(min_length=1)
    base_intercept: float
    platt_coefficient: float
    platt_intercept: float
    target_definition: str
    calibration_dataset: dict[str, object]
    evaluation: dict[str, object]
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _matching_feature_dimensions(self) -> "P2ModelArtifact":
        expected = len(self.feature_names)
        sequences = {
            "standardizer_mean": self.standardizer_mean,
            "standardizer_scale": self.standardizer_scale,
            "base_coefficients": self.base_coefficients,
        }
        for name, values in sequences.items():
            if len(values) != expected:
                raise ValueError(f"{name} must match feature_names length")
        if any(scale <= 0.0 for scale in self.standardizer_scale):
            raise ValueError("standardizer_scale must contain positive values")
        return self


class P2ExperimentalConfidence(BaseModel):
    """A calibrated probability with strict scope and limitation disclosure."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "p2-experimental-ai-confidence-v1"
    status: Literal["available", "unavailable"]
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    target_definition: str
    model_version: str | None = None
    feature_schema_version: str = "p2-geometry-features-v1"
    evaluation: dict[str, object] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
