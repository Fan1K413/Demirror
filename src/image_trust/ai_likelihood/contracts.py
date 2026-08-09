"""Stable contracts for the P3 calibrated AI-likelihood module."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AiSignal(BaseModel):
    """One auditable signal; its value is never silently treated as a probability."""

    model_config = ConfigDict(frozen=True)

    name: Literal[
        "verified_c2pa",
        "fsd_forensic",
        "frequency_texture",
        "geometry",
        "dda_pixel_detector",
        "safe_pixel_detector",
        "forensic_clip_detector",
        "community_forensics_detector",
        "nonescape_mini_detector",
    ]
    status: Literal["available", "neutral", "not_run", "unavailable", "failed"]
    value: float | None = None
    interpretation: str
    details: dict[str, object] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class AiLikelihoodResult(BaseModel):
    """A scoped local AI-origin signal, not a general provenance verdict."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "ai-likelihood-result-v2"
    status: Literal["available", "unavailable"]
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_band: Literal["low", "medium", "high", "unknown"] = "unknown"
    reliability: float = Field(default=0.0, ge=0.0, le=1.0)
    reliability_label: Literal["high", "limited", "unavailable"] = "unavailable"
    calibration_prior: Literal["balanced_50_50_review_pool"] = "balanced_50_50_review_pool"
    target_definition: str
    model_version: str | None = None
    decision_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    signals: list[AiSignal] = Field(default_factory=list)
    evaluation: dict[str, object] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
