import math

from image_trust.geometry.metrics import compute_line_metrics
from image_trust.schemas import LineRecord, Point


def _line(line_id: str, p1: tuple[float, float], p2: tuple[float, float]) -> LineRecord:
    length = math.dist(p1, p2)
    return LineRecord(
        line_id=line_id,
        p1_analysis=Point(x=p1[0], y=p1[1]),
        p2_analysis=Point(x=p2[0], y=p2[1]),
        p1=Point(x=p1[0], y=p1[1]),
        p2=Point(x=p2[0], y=p2[1]),
        length_analysis=length,
        length=length,
        angle_rad=math.atan2(p2[1] - p1[1], p2[0] - p1[0]) % math.pi,
        quality=1.0,
        selected=True,
    )


def test_spatial_coverage_counts_every_grid_cell_crossed() -> None:
    metrics = compute_line_metrics([_line("l1", (0, 25), (99, 25))], (100, 100), 4, 8)
    assert metrics.occupied_cells == 4
    assert metrics.spatial_coverage == 0.25
    assert metrics.direction_entropy == 0.0


def test_direction_entropy_increases_for_multiple_axes() -> None:
    metrics = compute_line_metrics(
        [_line("l1", (0, 25), (99, 25)), _line("l2", (25, 0), (25, 99))],
        (100, 100),
        4,
        8,
    )
    assert metrics.direction_entropy > 0.0
    assert metrics.spatial_entropy > 0.0
