from image_trust.geometry.applicability import assess_applicability
from image_trust.geometry.metrics import LineMetrics
from image_trust.schemas import ApplicabilityConfig


def _metrics(
    line_count: int = 40,
    total_length: float = 12.0,
    coverage: float = 0.25,
) -> LineMetrics:
    return LineMetrics(
        line_count=line_count,
        total_length_normalized=total_length,
        spatial_coverage=coverage,
        direction_entropy=0.5,
        spatial_entropy=0.5,
        occupied_cells=16,
    )


def test_applicability_reports_components_without_source_direction() -> None:
    assessment = assess_applicability(_metrics(), ApplicabilityConfig(), [])
    assert assessment.score == 1.0
    assert assessment.low_information is False
    assert assessment.special_imaging is False
    assert assessment.components == {
        "line_count_component": 1.0,
        "total_length_component": 1.0,
        "spatial_coverage_component": 1.0,
    }


def test_low_information_is_capped_and_limited() -> None:
    assessment = assess_applicability(
        _metrics(line_count=7, total_length=2.0, coverage=0.08),
        ApplicabilityConfig(),
        [],
    )
    assert assessment.low_information is True
    assert assessment.score == 0.20
    assert "insufficient_long_line_coverage" in assessment.limitations


def test_special_imaging_is_capped_and_distinguished() -> None:
    assessment = assess_applicability(
        _metrics(),
        ApplicabilityConfig(),
        ["panorama_metadata_detected"],
    )
    assert assessment.special_imaging is True
    assert assessment.score == 0.20
    assert "geometry_not_applicable_without_special_imaging_model" in assessment.limitations
