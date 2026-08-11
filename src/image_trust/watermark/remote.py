"""Local configuration for optional provider-hosted provenance checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


OPENAI_PROVENANCE_ENDPOINT = "https://api.openai.com/v1/content_provenance_checks"
OPENAI_PROVENANCE_DOCS = "https://developers.openai.com/api/docs/guides/content-provenance"
GOOGLE_SYNTHID_VERIFY_URL = "https://gemini.google.com/"
GOOGLE_SYNTHID_INFO_URL = "https://deepmind.google/models/synthid/"
GOOGLE_SYNTHID_MANUAL_ENABLED = "GOOGLE_SYNTHID_MANUAL_ENABLED"


@dataclass(frozen=True)
class RemoteVerificationSettings:
    """Secrets stay in process memory and are deliberately excluded from repr."""

    openai_api_key: str | None = field(default=None, repr=False)
    google_synthid_manual_enabled: bool = False

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key)


def load_remote_verification_settings(project_root: Path) -> RemoteVerificationSettings:
    """Read optional provider settings from the environment or ignored ``.env``."""

    api_key = _clean_secret(os.environ.get("OPENAI_API_KEY"))
    if api_key is None:
        api_key = _clean_secret(_read_env_value(project_root / ".env", "OPENAI_API_KEY"))
    google_toggle = os.environ.get(GOOGLE_SYNTHID_MANUAL_ENABLED)
    if google_toggle is None:
        google_toggle = _read_env_value(project_root / ".env", GOOGLE_SYNTHID_MANUAL_ENABLED)
    return RemoteVerificationSettings(
        openai_api_key=api_key,
        google_synthid_manual_enabled=_is_truthy(google_toggle),
    )


def remote_verification_capabilities(
    settings: RemoteVerificationSettings | None = None,
) -> dict[str, object]:
    """Return browser-safe capability metadata without secret material."""

    configured = settings or RemoteVerificationSettings()
    return {
        "openai": {
            "mode": "api",
            "configured": configured.openai_configured,
            "requires_explicit_opt_in": True,
            "uploads_selected_file": True,
            "docs_url": OPENAI_PROVENANCE_DOCS,
        },
        "google": {
            "mode": "manual_only",
            "configured": configured.google_synthid_manual_enabled,
            "requires_explicit_opt_in": True,
            "uploads_selected_file": True,
            "verification_url": GOOGLE_SYNTHID_VERIFY_URL,
            "info_url": GOOGLE_SYNTHID_INFO_URL,
            "reason": "google_synthid_image_detection_api_not_public",
        },
    }


def _read_env_value(path: Path, name: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        candidate = value.strip()
        if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
            candidate = candidate[1:-1]
        return candidate
    return None


def _clean_secret(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _is_truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})
