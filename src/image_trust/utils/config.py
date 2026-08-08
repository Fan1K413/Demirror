"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from image_trust.schemas import P0Config


def load_config(path: Path) -> P0Config:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return P0Config.model_validate(raw)

