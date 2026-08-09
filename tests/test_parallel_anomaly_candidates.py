from __future__ import annotations

import math

from image_trust.geometry.vanishing_points import (
    fit_parallel_families,
    fit_vanishing_families,
    identify_anomaly_candidates,
    identify_compact_component_conflict_candidates,
    identify_parallel_anomaly_candidates,
)
from image_trust.schemas import (
    LineRecord,
    Point,
    VanishingPointConfig,
    VPFamily,
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


def test_competing_parallel_family_must_be_spatially_compact() -> None:
    lines = [
        _line(f"dominant{index}", 10, 20 + index * 28, 190, 20 + index * 28)
        for index in range(6)
    ]
    # These diagonal lines share an orientation but occur in four distant
    # places.  They are a valid second scene direction, not evidence that one
    # local structural stroke bends.
    lines.extend(
        [
            _line("scattered1", 15, 15, 65, 35),
            _line("scattered2", 125, 25, 175, 45),
            _line("scattered3", 20, 150, 70, 170),
            _line("scattered4", 130, 155, 180, 175),
        ]
    )
    config = VanishingPointConfig(
        min_family_lines=4,
        min_family_weight=20.0,
        bootstrap_rounds=4,
        parallel_inlier_angle_deg=2.5,
        competing_family_max_extent_ratio=0.60,
    )
    transform = CoordinateTransform(
        encoded_size=(200, 200),
        canonical_size=(200, 200),
        analysis_size=(200, 200),
        exif_orientation=1,
        orientation_applied=False,
    )
    families = fit_parallel_families(lines, (200, 200), transform, config, seed=11)

    candidates = identify_parallel_anomaly_candidates(
        lines,
        families,
        (200, 200),
        config,
        applicability=1.0,
        minimum_applicability=0.45,
    )

    assert candidates == []


def test_parallel_family_already_explained_by_another_fit_is_not_repainted_as_anomaly() -> None:
    """A complete roof/rail family must remain a family, even near a dominant axis."""

    lines = [
        _line(f"dominant{index}", 10, 20 + index * 24, 190, 20 + index * 24)
        for index in range(6)
    ]
    roof = [
        _line(f"roof{index}", 115 + index * 5, 25, 165 + index * 5, 48)
        for index in range(4)
    ]
    lines.extend(roof)
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
    families = fit_parallel_families(lines, (200, 200), transform, config, seed=17)

    candidates = identify_parallel_anomaly_candidates(
        lines,
        families,
        (200, 200),
        config,
        applicability=1.0,
        minimum_applicability=0.45,
        explained_line_ids={line.line_id for line in roof},
    )

    assert candidates == []


def test_unassigned_fragment_is_not_a_geometric_contradiction() -> None:
    lines = [
        _line(f"base{index}", 10, 20 + index * 28, 190, 20 + index * 28)
        for index in range(5)
    ]
    lines.append(_line("fragment", 92, 105, 112, 112))
    config = VanishingPointConfig(
        min_family_lines=4,
        min_family_weight=20.0,
        bootstrap_rounds=4,
        unassigned_candidate_min_length_ratio=0.08,
    )
    transform = CoordinateTransform(
        encoded_size=(200, 200),
        canonical_size=(200, 200),
        analysis_size=(200, 200),
        exif_orientation=1,
        orientation_applied=False,
    )
    families = fit_vanishing_families(lines, (200, 200), transform, config, seed=13)

    candidates = identify_anomaly_candidates(
        lines,
        families,
        (200, 200),
        config,
        applicability=1.0,
        minimum_applicability=0.45,
    )

    assert candidates == []


def test_compact_component_can_be_reviewed_when_global_vp_fit_hides_it() -> None:
    """Two nearby conflicting edges must not be hidden by remote VP fragments."""

    compact = [
        _line("diag1", 130, 70, 150, 120),
        _line("diag2", 133, 70, 153, 120),
    ]
    remote = [
        _line("remote1", 10, 10, 30, 60),
        _line("remote2", 30, 150, 50, 200),
        _line("remote3", 20, 100, 40, 150),
    ]
    vertical = [
        _line("vertical1", 150, 20, 150, 180),
        _line("vertical2", 170, 20, 170, 180),
        _line("vertical3", 190, 20, 190, 180),
        _line("vertical4", 110, 20, 110, 180),
    ]
    skew_family = _family(
        "vp-small",
        "finite",
        [line.line_id for line in [*compact, *remote]],
        weighted_inlier_ratio=0.10,
    )
    dominant_parallel = _family(
        "parallel-dominant",
        "infinite",
        [line.line_id for line in vertical],
        weighted_inlier_ratio=0.50,
        direction_analysis=math.pi / 2.0,
    )
    candidates = identify_compact_component_conflict_candidates(
        [*compact, *remote, *vertical],
        [skew_family],
        [dominant_parallel],
        (200, 200),
        VanishingPointConfig(),
        applicability=1.0,
        minimum_applicability=0.45,
    )

    assert [candidate.line_id for candidate in candidates] == ["diag1", "diag2"]
    assert all(
        candidate.reason
        == "compact_global_component_conflicts_with_nearby_parallel_family"
        for candidate in candidates
    )


def test_compact_component_rule_requires_multiple_nearby_family_members() -> None:
    single = _line("diag", 130, 70, 150, 120)
    remote = [
        _line("remote1", 10, 10, 30, 60),
        _line("remote2", 30, 150, 50, 200),
        _line("remote3", 20, 100, 40, 150),
    ]
    vertical = [
        _line("vertical1", 150, 20, 150, 180),
        _line("vertical2", 170, 20, 170, 180),
        _line("vertical3", 190, 20, 190, 180),
        _line("vertical4", 110, 20, 110, 180),
    ]
    candidates = identify_compact_component_conflict_candidates(
        [single, *remote, *vertical],
        [
            _family(
                "vp-small",
                "finite",
                [single.line_id, *(line.line_id for line in remote)],
                weighted_inlier_ratio=0.10,
            )
        ],
        [
            _family(
                "parallel-dominant",
                "infinite",
                [line.line_id for line in vertical],
                weighted_inlier_ratio=0.50,
                direction_analysis=math.pi / 2.0,
            )
        ],
        (200, 200),
        VanishingPointConfig(),
        applicability=1.0,
        minimum_applicability=0.45,
    )

    assert candidates == []


def test_compact_component_rule_rejects_short_duplicate_edges() -> None:
    """Texture-scale duplicate edges are too short for structural review."""

    compact = [
        _line("short1", 130, 70, 136, 85),
        _line("short2", 132, 70, 138, 85),
    ]
    remote = [
        _line("remote1", 10, 10, 30, 60),
        _line("remote2", 30, 150, 50, 200),
        _line("remote3", 20, 100, 40, 150),
    ]
    vertical = [
        _line("vertical1", 150, 20, 150, 180),
        _line("vertical2", 170, 20, 170, 180),
        _line("vertical3", 190, 20, 190, 180),
        _line("vertical4", 110, 20, 110, 180),
    ]
    candidates = identify_compact_component_conflict_candidates(
        [*compact, *remote, *vertical],
        [
            _family(
                "vp-small",
                "finite",
                [*(line.line_id for line in compact), *(line.line_id for line in remote)],
                weighted_inlier_ratio=0.10,
            )
        ],
        [
            _family(
                "parallel-dominant",
                "infinite",
                [line.line_id for line in vertical],
                weighted_inlier_ratio=0.50,
                direction_analysis=math.pi / 2.0,
            )
        ],
        (200, 200),
        VanishingPointConfig(),
        applicability=1.0,
        minimum_applicability=0.45,
    )

    assert candidates == []


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


def _family(
    family_id: str,
    vp_type: str,
    member_line_ids: list[str],
    *,
    weighted_inlier_ratio: float,
    direction_analysis: float | None = None,
) -> VPFamily:
    return VPFamily(
        family_id=family_id,
        vp_type=vp_type,
        direction_analysis=direction_analysis,
        member_line_ids=member_line_ids,
        weighted_inlier_ratio=weighted_inlier_ratio,
        weighted_median_residual_deg=0.5,
        spatial_support=0.4,
        bootstrap_stability=0.9,
        residual_quantiles_deg={"p50": 0.5, "p90": 1.0, "p95": 1.5},
        stable=True,
    )
