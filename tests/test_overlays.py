from pathlib import Path

import numpy as np
from PIL import Image

from image_trust.geometry.overlays import FAMILY_COLORS, write_overlays
from image_trust.schemas import LineRecord, OverlayConfig, Point, VPFamily


def _line(line_id: str, y: float) -> LineRecord:
    return LineRecord(
        line_id=line_id,
        p1_analysis=Point(x=30, y=y),
        p2_analysis=Point(x=170, y=y),
        p1=Point(x=30, y=y),
        p2=Point(x=170, y=y),
        length_analysis=140,
        length=140,
        angle_rad=0.0,
        quality=1.0,
        selected=True,
    )


def _family(family_id: str, member_line_id: str) -> VPFamily:
    return VPFamily(
        family_id=family_id,
        vp_type="infinite",
        direction_analysis=0.0,
        member_line_ids=[member_line_id],
        weighted_inlier_ratio=0.5,
        weighted_median_residual_deg=0.1,
        spatial_support=0.1,
        bootstrap_stability=1.0,
        residual_quantiles_deg={"p50": 0.1, "p90": 0.2, "p95": 0.2},
        stable=True,
    )


def test_stable_families_have_distinct_overlay_colors(tmp_path: Path) -> None:
    lines = [_line("l000001", 100), _line("l000002", 150), _line("l000003", 180)]
    write_overlays(
        np.full((220, 220, 3), 245, dtype=np.uint8),
        lines,
        [_family("vp001", "l000001"), _family("vp002", "l000002")],
        [],
        OverlayConfig(line_width=2),
        tmp_path / "lines.png",
        tmp_path / "anomalies.png",
    )
    with Image.open(tmp_path / "anomalies.png") as image:
        assert image.getpixel((100, 100)) == FAMILY_COLORS[0]
        assert image.getpixel((100, 150)) == FAMILY_COLORS[1]
        # Full line coverage belongs in lines_overlay; an unexplained line is
        # intentionally absent from the family/candidate review image.
        assert image.getpixel((100, 180)) == (245, 245, 245)
        assert FAMILY_COLORS[0] != FAMILY_COLORS[1]
