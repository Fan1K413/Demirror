from __future__ import annotations

from image_trust.geometry.vanishing_points import (
    fit_parallel_families,
    identify_parallel_anomaly_candidates,
)
from image_trust.schemas import (
    LineRecord,
    Point,
    VanishingPointConfig,
)
from image_trust.utils.coordinates import CoordinateTransform


def test_parallel_detector_flags_only_the_nearby_direction_outlier() -> None:
    lines = [
        _line(f"h{index}", 20, 20 + index * 25, 180, 20 + index * 25)
        for index in range(5)
    ]
    lines.extend(
        [
            _line("outlier", 25, 78, 175, 122),
            _line("unrelated", 20, 180, 180, 195),
        ]
    )
    config = VanishingPointConfig(
        min_family_lines=4,
        min_family_weight=20.0,
        bootstrap_rounds=4,
        parallel_inlier_angle_deg=2.5,
    )
    transform = CoordinateTransform(
        encoded_size=(200, 200),
        canonical_size=(200, 200),
        analysis_size=(200, 200),
        exif_orientation=1,
        orientation_applied=False,
    )
    families = fit_parallel_families(lines, (200, 200), transform, config, seed=7)

    candidates = identify_parallel_anomaly_candidates(
        lines,
        families,
        (200, 200),
        config,
        applicability=1.0,
        minimum_applicability=0.45,
    )

    assert [candidate.line_id for candidate in candidates] == ["outlier"]
    assert candidates[0].reason == "unassigned_line_deviates_from_nearby_parallel_family"
    assert candidates[0].residual_deg is not None
    assert candidates[0].residual_deg > config.parallel_inlier_angle_deg


def _line(line_id: str, x1: float, y1: float, x2: float, y2: float) -> LineRecord:
    return LineRecord(
        line_id=line_id,
        p1_analysis=Point(x=x1, y=y1),
        p2_analysis=Point(x=x2, y=y2),
        p1=Point(x=x1, y=y1),
        p2=Point(x=x2, y=y2),
        length_analysis=((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5,
        length=((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5,
        angle_rad=0.0,
        quality=1.0,
        selected=True,
    )
