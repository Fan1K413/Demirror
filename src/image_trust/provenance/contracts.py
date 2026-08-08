"""Source-neutral contracts for local C2PA inspection."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class C2paRecordStatus(str, Enum):
    NOT_OBSERVED = "not_observed"
    PRESENT = "present"
    MALFORMED = "malformed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class C2paSignatureValidationStatus(str, Enum):
    NOT_OBSERVED = "not_observed"
    VALID = "valid"
    INVALID = "invalid"
    INDETERMINATE = "indeterminate"


class C2paTrustStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    INDETERMINATE = "indeterminate"


class P1C2paConfig(BaseModel):
    """Configuration for an offline-only C2PA reader."""

    model_config = ConfigDict(frozen=True)

    config_version: str
    module_name: str = "c2pa"
    expected_dependency_version: str = "0.32.6"
    trust_list_version: str = "not_configured"
    network_access: Literal["disabled"] = "disabled"


class C2paRecord(BaseModel):
    """Minimal audit record from a C2PA manifest store.

    The record says only what the local SDK observed and validated.  It never
    converts claims, missing manifests, or signature outcomes into a source or
    AI-generation conclusion.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = "c2pa-record-v1"
    config_version: str
    config_digest: str
    input_sha256: str | None = None
    original_filename: str
    status: C2paRecordStatus
    manifest_present: bool
    asset_media_type: str | None = None
    active_manifest_label: str | None = None
    assertion_labels: list[str] = Field(default_factory=list)
    declared_actions: list[str] = Field(default_factory=list)
    validation_state: str | None = None
    validation_status_codes: list[str] = Field(default_factory=list)
    signature_validation_status: C2paSignatureValidationStatus
    trust_status: C2paTrustStatus
    trust_list_version: str
    sdk_version: str | None = None
    network_access: Literal["disabled"] = "disabled"
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_manifest_availability(self) -> "C2paRecord":
        if self.status in {C2paRecordStatus.PRESENT, C2paRecordStatus.MALFORMED}:
            if not self.manifest_present:
                raise ValueError("Present or malformed C2PA records require a manifest.")
        elif self.manifest_present:
            raise ValueError("Unavailable C2PA records must not claim a manifest.")
        if self.status is C2paRecordStatus.NOT_OBSERVED:
            if self.signature_validation_status is not C2paSignatureValidationStatus.NOT_OBSERVED:
                raise ValueError("Missing manifests must have an unobserved signature status.")
        return self
