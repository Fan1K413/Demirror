"""P1 C2PA configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from image_trust.provenance.contracts import P1C2paConfig


def load_c2pa_config(path: Path) -> P1C2paConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return P1C2paConfig.model_validate(raw)
