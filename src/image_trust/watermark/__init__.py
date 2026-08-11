"""Offline implicit-watermark adapters and auditable result contracts."""

from image_trust.watermark.contracts import (
    ImplicitWatermarkAssessment,
    WatermarkAdapterResult,
    WatermarkCoverage,
    WatermarkPayload,
    WatermarkScore,
)
from image_trust.watermark.suite import (
    assess_implicit_watermarks,
    build_offline_watermark_adapters,
)

__all__ = [
    "ImplicitWatermarkAssessment",
    "WatermarkAdapterResult",
    "WatermarkCoverage",
    "WatermarkPayload",
    "WatermarkScore",
    "assess_implicit_watermarks",
    "build_offline_watermark_adapters",
]
