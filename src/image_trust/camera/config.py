"""P1 camera configuration loading, intentionally separate from P0."""

from __future__ import annotations

from pathlib import Path

import yaml

from image_trust.camera.contracts import P1CameraConfig
from image_trust.runtime_paths import runtime_weights_root


def load_camera_config(path: Path) -> P1CameraConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    backend = raw.get("camera_backend")
    if isinstance(backend, dict) and isinstance(backend.get("weights_path"), str):
        configured_path = Path(backend["weights_path"])
        if not configured_path.is_absolute() and configured_path.parts[:1] == ("weights",):
            backend["weights_path"] = str(
                runtime_weights_root(path.resolve().parents[1]) / Path(*configured_path.parts[1:])
            )
    return P1CameraConfig.model_validate(raw)
