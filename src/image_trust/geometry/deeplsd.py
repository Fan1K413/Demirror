"""Explicit DeepLSD placeholder; P0 never pretends it is available."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from image_trust.geometry.line_backend import BackendUnavailableError, LineExtraction
from image_trust.schemas import LineBackendConfig


class DeepLSDBackend:
    backend_id = "deeplsd"

    def __init__(self, config: LineBackendConfig) -> None:
        weight_path = Path(config.deeplsd_weights) if config.deeplsd_weights else None
        if weight_path is None:
            raise BackendUnavailableError(
                self.backend_id,
                "DeepLSD was explicitly requested but deeplsd_weights is not configured. "
                "Install the official DeepLSD dependency and set a local, licensed weight path.",
            )
        if not weight_path.is_file():
            raise BackendUnavailableError(
                self.backend_id,
                f"DeepLSD weights were not found at: {weight_path}",
            )
        try:
            import deeplsd  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                self.backend_id,
                "DeepLSD weights are configured but the deeplsd package is not installed. "
                "Follow the pinned upstream installation instructions before selecting this backend.",
            ) from exc
        raise BackendUnavailableError(
            self.backend_id,
            "DeepLSD dependency and weights were found, but its inference adapter has not "
            "yet been implemented. P0 does not substitute another backend silently.",
        )

    def extract(self, grayscale: np.ndarray) -> LineExtraction:
        raise AssertionError("DeepLSDBackend cannot be constructed until its adapter is implemented.")

