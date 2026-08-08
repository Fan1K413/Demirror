"""File-based P1 entry point with canonical EXIF orientation handling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from image_trust.camera.backends import CameraBackend, resolve_camera_backend
from image_trust.camera.contracts import CameraExperimentResult, CameraInputSummary, P1CameraConfig
from image_trust.camera.experiment import run_camera_consistency_experiment


def analyze_camera_image(
    input_path: Path,
    config: P1CameraConfig,
    output_dir: Path,
    *,
    backend: CameraBackend | None = None,
) -> CameraExperimentResult:
    """Read one local image, run P1, and write a source-neutral JSON result."""

    canonical_rgb = _load_canonical_rgb(input_path)
    height, width = canonical_rgb.shape[:2]
    result = run_camera_consistency_experiment(
        canonical_rgb=canonical_rgb,
        input_summary=CameraInputSummary(
            sha256=_sha256(input_path),
            original_filename=input_path.name,
            canonical_size=(width, height),
        ),
        config=config,
        backend=backend or resolve_camera_backend(config.camera_backend),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "camera_result.json", result.model_dump(mode="json"))
    return result


def _load_canonical_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        canonical = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(canonical).copy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
