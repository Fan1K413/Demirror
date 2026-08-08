"""Applicability gate for the P0 geometric measurement branch."""

from __future__ import annotations

from dataclasses import dataclass

from image_trust.geometry.metrics import LineMetrics
from image_trust.schemas import ApplicabilityConfig


@dataclass(frozen=True)
class ApplicabilityAssessment:
    score: float
    components: dict[str, float]
    low_information: bool
    special_imaging: bool
    limitations: list[str]


def assess_applicability(
    metrics: LineMetrics,
    config: ApplicabilityConfig,
    special_imaging_hints: list[str],
) -> ApplicabilityAssessment:
    components = {
        "line_count_component": min(
            1.0, metrics.line_count / max(config.target_line_count, 1)
        ),
        "total_length_component": min(
            1.0, metrics.total_length_normalized / max(config.target_total_length, 1e-9)
        ),
        "spatial_coverage_component": min(
            1.0, metrics.spatial_coverage / max(config.target_spatial_coverage, 1e-9)
        ),
    }
    score = (
        0.40 * components["line_count_component"]
        + 0.30 * components["total_length_component"]
        + 0.30 * components["spatial_coverage_component"]
    )
    low_information = (
        metrics.line_count < config.min_line_count
        or metrics.total_length_normalized < config.min_total_length
        or metrics.spatial_coverage < config.min_spatial_coverage
    )
    special_imaging = bool(special_imaging_hints)
    limitations = list(special_imaging_hints)
    if low_information:
        limitations.append("insufficient_long_line_coverage")
    if low_information or special_imaging:
        score = min(score, config.special_imaging_cap)
    if special_imaging:
        limitations.append("geometry_not_applicable_without_special_imaging_model")
    return ApplicabilityAssessment(
        score=float(max(0.0, min(score, 1.0))),
        components=components,
        low_information=low_information,
        special_imaging=special_imaging,
        limitations=sorted(set(limitations)),
    )
