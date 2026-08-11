"""Stable contracts shared by provider-specific watermark adapters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RunStatus = Literal["ok", "unavailable", "not_applicable", "failed"]
Observation = Literal["positive", "negative", "not_observed"]
EvidenceClass = Literal[
    "verified_provider_ai",
    "known_open_ai_watermark",
    "unverified_identifier",
    "none",
]
Direction = Literal["supports_ai", "neutral"]
Strength = Literal["strong", "limited", "none"]


class WatermarkProviderSignal(BaseModel):
    """Sanitized signal metadata returned by an official provider API."""

    model_config = ConfigDict(frozen=True)

    signal_type: Literal["c2pa", "synthid"]
    outcome: Literal["detected", "not_detected"]
    validation_state: Literal["trusted", "valid", "invalid", "not_present"] | None = None
    model: str | None = None
    issuer: str | None = None
    generated_at: str | None = None


class WatermarkCoverage(BaseModel):
    """The scheme and input domain to which one observation applies."""

    model_config = ConfigDict(frozen=True)

    media: Literal["image"] = "image"
    ecosystem: list[str] = Field(default_factory=list)
    min_short_side: int | None = Field(default=None, ge=1)
    supported_formats: list[str] = Field(default_factory=list)


class WatermarkScore(BaseModel):
    """A detector diagnostic, never a general AI probability."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    threshold: float | None = None
    threshold_id: str | None = None


class WatermarkPayload(BaseModel):
    """Privacy-preserving description of a decoded watermark payload."""

    model_config = ConfigDict(frozen=True)

    present: bool = False
    payload_schema: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    bit_length: int | None = Field(default=None, ge=1)


class WatermarkAdapterResult(BaseModel):
    """One adapter observation with explicit applicability and semantics."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "watermark-adapter-result-v1"
    adapter_id: str = Field(min_length=1)
    scheme: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    run_status: RunStatus
    observation: Observation
    evidence_class: EvidenceClass = "none"
    direction: Direction = "neutral"
    strength: Strength = "none"
    decision_eligible: bool = False
    coverage: WatermarkCoverage
    score: WatermarkScore | None = None
    payload: WatermarkPayload = Field(default_factory=WatermarkPayload)
    provider: Literal["openai"] | None = None
    provider_signals: list[WatermarkProviderSignal] = Field(default_factory=list)
    network_access: Literal["none", "explicit_opt_in"] = "none"
    data_sent: bool = False
    limitations: list[str] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_semantics(self) -> "WatermarkAdapterResult":
        if self.network_access == "none" and self.data_sent:
            raise ValueError("offline adapter cannot report external data transfer")
        if self.provider_signals and self.provider is None:
            raise ValueError("provider signals require a named provider")
        if self.run_status == "ok" and self.observation == "not_observed":
            raise ValueError("successful adapter must report positive or negative")
        if self.run_status != "ok" and self.observation != "not_observed":
            raise ValueError("non-successful adapter cannot report positive or negative")
        if self.observation == "negative":
            if self.evidence_class != "none" or self.direction != "neutral" or self.strength != "none":
                raise ValueError("negative watermark observation must remain neutral")
        if self.observation == "positive" and self.evidence_class == "none":
            raise ValueError("positive watermark observation requires an evidence class")
        if self.evidence_class == "unverified_identifier":
            if self.direction != "neutral" or self.strength != "none" or self.decision_eligible:
                raise ValueError("unverified identifier cannot affect the origin decision")
        if self.evidence_class == "known_open_ai_watermark" and self.strength == "strong":
            raise ValueError("open watermark cannot be strong provenance evidence")
        if self.decision_eligible:
            if self.observation != "positive" or self.direction != "supports_ai":
                raise ValueError("decision-eligible watermark must be positive AI support")
            if self.strength == "none":
                raise ValueError("decision-eligible watermark requires a strength")
        return self


class ImplicitWatermarkAssessment(BaseModel):
    """Aggregate local watermark result kept separate from camera evidence."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "implicit-watermark-assessment-v1"
    status: Literal["not_configured", "completed", "partial", "unavailable"]
    adapters: list[WatermarkAdapterResult] = Field(default_factory=list)
    direction: Direction = "neutral"
    strength: Strength = "none"
    decision_eligible: bool = False
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_aggregate(self) -> "ImplicitWatermarkAssessment":
        eligible = [item for item in self.adapters if item.decision_eligible]
        if self.decision_eligible != bool(eligible):
            raise ValueError("aggregate decision eligibility must match adapter results")
        if not eligible and (self.direction != "neutral" or self.strength != "none"):
            raise ValueError("non-eligible aggregate must remain neutral")
        if eligible and self.direction != "supports_ai":
            raise ValueError("eligible aggregate must support AI")
        return self

    @classmethod
    def not_configured(cls) -> "ImplicitWatermarkAssessment":
        return cls(
            status="not_configured",
            limitations=["implicit_watermark_detector_not_configured"],
        )
