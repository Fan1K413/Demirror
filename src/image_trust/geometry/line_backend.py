"""Stable abstraction over line-segment extraction backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from image_trust.schemas import LineBackendConfig


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested backend cannot run."""

    def __init__(self, backend: str, message: str) -> None:
        super().__init__(message)
        self.backend = backend
        self.message = message


@dataclass(frozen=True)
class RawLine:
    p1: tuple[float, float]
    p2: tuple[float, float]
    quality: float
    backend_features: dict[str, float | None]


@dataclass(frozen=True)
class LineExtraction:
    backend_id: str
    backend_version: str
    lines: list[RawLine]
    warnings: list[str]


class LineBackend(Protocol):
    backend_id: str

    def extract(self, grayscale: np.ndarray) -> LineExtraction:
        """Extract raw analysis-coordinate line segments."""


def resolve_backend(config: LineBackendConfig) -> LineBackend:
    requested = config.name.lower()
    if requested == "opencv_lsd":
        from image_trust.geometry.opencv_lsd import OpenCVLSDBackend

        return OpenCVLSDBackend(config)
    if requested == "deeplsd":
        from image_trust.geometry.deeplsd import DeepLSDBackend

        return DeepLSDBackend(config)
    if requested == "auto":
        try:
            from image_trust.geometry.deeplsd import DeepLSDBackend

            return DeepLSDBackend(config)
        except BackendUnavailableError:
            if config.allow_fallback:
                from image_trust.geometry.opencv_lsd import OpenCVLSDBackend

                return OpenCVLSDBackend(config, fallback_reason="deeplsd_unavailable")
            raise
    raise BackendUnavailableError(
        requested,
        f"Unsupported line backend '{config.name}'. Use opencv_lsd, deeplsd, or auto.",
    )

