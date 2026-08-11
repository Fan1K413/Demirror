"""Provider-independent orchestration for local watermark adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from image_trust.watermark.contracts import (
    ImplicitWatermarkAssessment,
    WatermarkAdapterResult,
    WatermarkCoverage,
)


WatermarkAdapter = Callable[[Path], WatermarkAdapterResult]


def build_offline_watermark_adapters() -> tuple[WatermarkAdapter, ...]:
    """Return the pinned, network-free adapters enabled by default."""

    from image_trust.watermark.sdxl import SdxlDwtDctAdapter
    from image_trust.watermark.trustmark import TrustMarkQAdapter

    return (SdxlDwtDctAdapter(), TrustMarkQAdapter())


def assess_implicit_watermarks(
    input_path: Path,
    adapters: Iterable[WatermarkAdapter] = (),
) -> ImplicitWatermarkAssessment:
    """Run configured local adapters without turning absence into camera evidence."""

    configured = tuple(adapters)
    if not configured:
        return ImplicitWatermarkAssessment.not_configured()

    results: list[WatermarkAdapterResult] = []
    for adapter in configured:
        try:
            results.append(adapter(input_path))
        except Exception as error:
            results.append(
                WatermarkAdapterResult(
                    adapter_id=getattr(adapter, "adapter_id", adapter.__class__.__name__),
                    scheme=getattr(adapter, "scheme", "unknown"),
                    detector_version=getattr(adapter, "detector_version", "unknown"),
                    run_status="failed",
                    observation="not_observed",
                    coverage=getattr(
                        adapter,
                        "coverage",
                        WatermarkCoverage(ecosystem=["unknown"]),
                    ),
                    limitations=["watermark_adapter_unhandled_failure"],
                    errors=[{"code": type(error).__name__, "message": str(error)}],
                )
            )

    eligible = [result for result in results if result.decision_eligible]
    completed = [result for result in results if result.run_status == "ok"]
    if len(completed) == len(results):
        status = "completed"
    elif completed:
        status = "partial"
    else:
        status = "unavailable"
    strength = "none"
    if any(result.strength == "strong" for result in eligible):
        strength = "strong"
    elif eligible:
        strength = "limited"
    return ImplicitWatermarkAssessment(
        status=status,
        adapters=results,
        direction="supports_ai" if eligible else "neutral",
        strength=strength,
        decision_eligible=bool(eligible),
        limitations=sorted({item for result in results for item in result.limitations}),
    )
