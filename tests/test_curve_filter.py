import math

from image_trust.geometry.curve_filter import suppress_curve_fragments
from image_trust.schemas import LineBackendConfig, LineRecord, Point


def _line(line_id: str, start: tuple[float, float], end: tuple[float, float]) -> LineRecord:
    length = math.dist(start, end)
    angle = math.atan2(end[1] - start[1], end[0] - start[0]) % math.pi
    return LineRecord(
        line_id=line_id,
        p1_analysis=Point(x=start[0], y=start[1]),
        p2_analysis=Point(x=end[0], y=end[1]),
        p1=Point(x=start[0], y=start[1]),
        p2=Point(x=end[0], y=end[1]),
        length_analysis=length,
        length=length,
        angle_rad=angle,
        quality=1.0,
        selected=True,
    )


def test_curve_chain_is_suppressed_but_isolated_road_edge_is_retained() -> None:
    center = (100.0, 100.0)
    radius = 70.0
    curve = []
    for index in range(7):
        first = math.radians(180 + index * 12)
        second = math.radians(180 + (index + 1) * 12)
        curve.append(
            _line(
                f"curve{index}",
                (center[0] + radius * math.cos(first), center[1] + radius * math.sin(first)),
                (center[0] + radius * math.cos(second), center[1] + radius * math.sin(second)),
            )
        )
    road_edge = _line("road", (300, 280), (390, 210))
    result = suppress_curve_fragments([*curve, road_edge], LineBackendConfig())
    assert set(result.suppressed_line_ids) == {line.line_id for line in curve}
    assert [line.line_id for line in result.lines] == ["road"]


def test_curve_filter_can_be_disabled() -> None:
    curve = [
        _line("a", (0, 0), (10, 2)),
        _line("b", (10, 2), (19, 6)),
        _line("c", (19, 6), (27, 12)),
        _line("d", (27, 12), (33, 20)),
        _line("e", (33, 20), (37, 29)),
    ]
    config = LineBackendConfig(suppress_curve_fragments=False)
    result = suppress_curve_fragments(curve, config)
    assert result.lines == curve
    assert result.suppressed_line_ids == []
