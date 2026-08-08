"""P1 camera configuration loading, intentionally separate from P0."""

from __future__ import annotations

from pathlib import Path

import yaml

from image_trust.camera.contracts import P1CameraConfig


def load_camera_config(path: Path) -> P1CameraConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return P1CameraConfig.model_validate(raw)
