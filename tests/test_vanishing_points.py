import math

from image_trust.geometry.vanishing_points import (
    fit_parallel_families,
    fit_local_parallel_families,
    fit_local_vanishing_families,
    fit_vanishing_families,
    identify_anomaly_candidates,
)
from image_trust.schemas import LineRecord, Point, VanishingPointConfig
from image_trust.utils.coordinates import CoordinateTransform


def _converging_lines() -> list[LineRecord]:
    vp_x, vp_y = 200.0, 100.0
    lines: list[LineRecord] = []
    for index, (x, y) in enumerate(
        [(30, 360), (90, 345), (150, 350), (250, 350), (320, 360), (370, 330)],
        start=1,
    ):
        midpoint = Point(x=(x + vp_x) / 2.0, y=(y + vp_y) / 2.0)
        p1 = Point(x=float(x), y=float(y))
        length = ((midpoint.x - p1.x) ** 2 + (midpoint.y - p1.y) ** 2) ** 0.5
        lines.append(
            LineRecord(
                line_id=f"l{index:06d}",
                p1_analysis=p1,
                p2_analysis=midpoint,
                p1=p1,
                p2=midpoint,
                length_analysis=length,
                length=length,
                angle_rad=0.0,
                quality=1.0,
                selected=True,
            )
        )
    return lines


def _rays_to_vp(
    prefix: str,
    vp: tuple[float, float],
    anchors: list[tuple[float, float]],
) -> list[LineRecord]:
    lines: list[LineRecord] = []
    for index, anchor in enumerate(anchors, start=1):
        midpoint = Point(x=(anchor[0] + vp[0]) / 2.0, y=(anchor[1] + vp[1]) / 2.0)
        start = Point(x=anchor[0], y=anchor[1])
        length = ((midpoint.x - start.x) ** 2 + (midpoint.y - start.y) ** 2) ** 0.5
        lines.append(
            LineRecord(
                line_id=f"{prefix}{index:04d}",
                p1_analysis=start,
                p2_analysis=midpoint,
                p1=start,
                p2=midpoint,
                length_analysis=length,
                length=length,
                angle_rad=0.0,
                quality=1.0,
                selected=True,
            )
        )
    return lines


def _parallel_lines(
    prefix: str,
    angle_deg: float,
    origins: list[tuple[float, float]],
) -> list[LineRecord]:
    angle = math.radians(angle_deg)
    delta = (120 * math.cos(angle), 120 * math.sin(angle))
    lines: list[LineRecord] = []
    for index, origin in enumerate(origins, start=1):
        p1 = Point(x=origin[0], y=origin[1])
        p2 = Point(x=origin[0] + delta[0], y=origin[1] + delta[1])
        lines.append(
            LineRecord(
                line_id=f"{prefix}{index:04d}",
                p1_analysis=p1,
                p2_analysis=p2,
                p1=p1,
                p2=p2,
                length_analysis=120,
                length=120,
                angle_rad=angle % math.pi,
                quality=1.0,
                selected=True,
            )
        )
    return lines


def test_ransac_finds_known_converging_family_deterministically() -> None:
    lines = _converging_lines()
    transform = CoordinateTransform(
        encoded_size=(400, 400),
        canonical_size=(400, 400),
        analysis_size=(400, 400),
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        max_hypotheses=100,
        min_family_lines=4,
        min_family_weight=1.0,
        bootstrap_rounds=4,
    )
    first = fit_vanishing_families(lines, (400, 400), transform, config, seed=123)
    second = fit_vanishing_families(lines, (400, 400), transform, config, seed=123)
    assert len(first) == 1
    assert first == second
    assert first[0].vp_type == "finite"
    assert first[0].weighted_inlier_ratio > 0.9
    assert first[0].vp_analysis is not None
    assert abs(first[0].vp_analysis.x - 200) < 1.0
    assert abs(first[0].vp_analysis.y - 100) < 1.0
    assert identify_anomaly_candidates(lines, first, (400, 400), config, 0.9, 0.45) == []


def test_ransac_separates_two_known_finite_families() -> None:
    lines = _rays_to_vp(
        "a",
        (120.0, 80.0),
        [(20.0, 350.0), (80.0, 360.0), (160.0, 360.0), (250.0, 340.0)],
    ) + _rays_to_vp(
        "b",
        (300.0, 90.0),
        [(180.0, 350.0), (260.0, 360.0), (340.0, 355.0), (390.0, 330.0)],
    )
    transform = CoordinateTransform(
        encoded_size=(400, 400),
        canonical_size=(400, 400),
        analysis_size=(400, 400),
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        max_families=2,
        max_hypotheses=300,
        min_family_lines=4,
        min_family_weight=1.0,
        bootstrap_rounds=4,
    )
    families = fit_vanishing_families(lines, (400, 400), transform, config, seed=123)
    assert len(families) == 2
    points = [(family.vp_analysis.x, family.vp_analysis.y) for family in families if family.vp_analysis]
    assert any(abs(x - 120) < 2 and abs(y - 80) < 2 for x, y in points)
    assert any(abs(x - 300) < 2 and abs(y - 90) < 2 for x, y in points)


def test_anomalies_require_a_stable_family() -> None:
    lines = _converging_lines()
    transform = CoordinateTransform(
        encoded_size=(400, 400),
        canonical_size=(400, 400),
        analysis_size=(400, 400),
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        max_hypotheses=100,
        min_family_lines=4,
        min_family_weight=1.0,
        bootstrap_rounds=0,
    )
    families = fit_vanishing_families(lines, (400, 400), transform, config, seed=123)
    assert families and families[0].stable is False
    assert identify_anomaly_candidates(lines, families, (400, 400), config, 0.9, 0.45) == []


def test_parallel_families_are_separate_from_global_vp_families() -> None:
    lines = _parallel_lines(
        "white",
        25,
        [(20, 30), (60, 70), (100, 110), (140, 150)],
    ) + _parallel_lines(
        "black",
        70,
        [(220, 30), (250, 70), (280, 110), (310, 150)],
    )
    transform = CoordinateTransform(
        encoded_size=(400, 400),
        canonical_size=(400, 400),
        analysis_size=(400, 400),
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        min_family_lines=4,
        min_family_weight=1.0,
        max_parallel_families=2,
        parallel_inlier_angle_deg=3.0,
        bootstrap_rounds=4,
    )
    families = fit_parallel_families(lines, (400, 400), transform, config, seed=321)
    assert len(families) == 2
    directions = sorted(family.direction_analysis for family in families)
    assert directions[0] is not None and directions[1] is not None
    assert abs(math.degrees(directions[0]) - 25) < 1.0
    assert abs(math.degrees(directions[1]) - 70) < 1.0
    assert {line_id[:5] for family in families for line_id in family.member_line_ids} == {
        "black",
        "white",
    }


def test_parallel_explained_line_is_not_shown_as_global_anomaly() -> None:
    base_lines = _converging_lines()
    transform = CoordinateTransform(
        encoded_size=(400, 400),
        canonical_size=(400, 400),
        analysis_size=(400, 400),
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        max_hypotheses=100,
        min_family_lines=4,
        min_family_weight=1.0,
        bootstrap_rounds=4,
    )
    families = fit_vanishing_families(base_lines, (400, 400), transform, config, seed=123)
    extra = _parallel_lines("parallel", 0, [(100, 50)])[0]
    candidates = identify_anomaly_candidates(
        [*base_lines, extra], families, (400, 400), config, 0.9, 0.45
    )
    assert any(candidate.line_id == extra.line_id for candidate in candidates)
    filtered = identify_anomaly_candidates(
        [*base_lines, extra],
        families,
        (400, 400),
        config,
        0.9,
        0.45,
        explained_line_ids={extra.line_id},
    )
    assert all(candidate.line_id != extra.line_id for candidate in filtered)


def test_local_vp_families_keep_separated_regions_distinct() -> None:
    lines = _rays_to_vp(
        "left",
        (80.0, 80.0),
        [(15.0, 200.0), (45.0, 210.0), (100.0, 220.0), (150.0, 230.0)],
    ) + _rays_to_vp(
        "right",
        (320.0, 80.0),
        [(240.0, 200.0), (280.0, 210.0), (335.0, 220.0), (375.0, 230.0)],
    )
    transform = CoordinateTransform(
        encoded_size=(400, 400),
        canonical_size=(400, 400),
        analysis_size=(400, 400),
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        max_hypotheses=200,
        min_family_lines=4,
        min_family_weight=1.0,
        local_family_grid_size=2,
        local_families_per_cell=1,
        max_local_families=2,
        local_min_family_weight=1.0,
        bootstrap_rounds=4,
    )
    families = fit_local_vanishing_families(lines, (400, 400), transform, config, seed=456)
    assert len(families) == 2
    assert all(family.scope == "local_vp" for family in families)
    assert {line_id[:4] for family in families for line_id in family.member_line_ids} == {
        "left",
        "righ",
    }
    assert len({family.spatial_window_analysis for family in families}) == 2


def test_local_direction_families_preserve_a_novel_parallel_roof_group() -> None:
    global_lines = _parallel_lines(
        "global",
        0,
        [(20, 50), (70, 65), (120, 80), (170, 95)],
    )
    roof_lines = _parallel_lines(
        "roof",
        72,
        [(240, 80), (265, 105), (290, 130), (315, 155)],
    )
    lines = [*global_lines, *roof_lines]
    transform = CoordinateTransform(
        encoded_size=(400, 400),
        canonical_size=(400, 400),
        analysis_size=(400, 400),
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        min_family_lines=4,
        min_family_weight=1.0,
        local_min_family_weight=1.0,
        local_family_grid_size=1,
        local_direction_families_per_cell=2,
        local_direction_inlier_angle_deg=3.0,
        max_local_families=2,
        bootstrap_rounds=4,
    )
    families = fit_local_parallel_families(
        lines,
        (400, 400),
        transform,
        config,
        seed=789,
        excluded_line_ids={line.line_id for line in global_lines},
    )
    assert len(families) == 1
    assert families[0].scope == "local_image_plane_parallel"
    assert abs(math.degrees(families[0].direction_analysis or 0.0) - 72) < 1.0
    assert set(families[0].member_line_ids) == {line.line_id for line in roof_lines}


def test_local_direction_family_joins_same_direction_across_cell_boundary() -> None:
    left = _parallel_lines("left", 45, [(20, 30), (45, 55), (70, 80), (95, 105)])
    right = _parallel_lines("right", 47, [(220, 30), (245, 55), (270, 80), (295, 105)])
    transform = CoordinateTransform(
        encoded_size=(400, 400),
        canonical_size=(400, 400),
        analysis_size=(400, 400),
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        min_family_lines=4,
        min_family_weight=1.0,
        local_min_family_weight=1.0,
        local_family_grid_size=2,
        local_direction_families_per_cell=1,
        local_direction_inlier_angle_deg=3.0,
        max_local_families=2,
        bootstrap_rounds=4,
    )
    families = fit_local_parallel_families(
        [*left, *right], (400, 400), transform, config, seed=987
    )
    assert len(families) == 1
    assert set(families[0].member_line_ids) == {
        *(line.line_id for line in left),
        *(line.line_id for line in right),
    }
