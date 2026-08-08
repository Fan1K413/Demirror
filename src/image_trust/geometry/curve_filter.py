"""Conservative suppression of short LSD fragments that trace a curve."""

from __future__ import annotations

import math
from dataclasses import dataclass

from image_trust.schemas import LineBackendConfig, LineRecord


@dataclass(frozen=True)
class CurveSuppressionResult:
    lines: list[LineRecord]
    suppressed_line_ids: list[str]


def suppress_curve_fragments(
    lines: list[LineRecord],
    config: LineBackendConfig,
) -> CurveSuppressionResult:
    """Remove chains of short, smoothly turning segments before line capping.

    LSD approximates an arch or circular ornament with several individually
    straight fragments.  A single short fragment is not enough to reject it:
    it must join a nearby tangent-continuous chain that covers several
    consecutive directions.  This keeps isolated road edges and lamp posts,
    while preventing curves from acting as vanishing-direction evidence.
    """
    if not config.suppress_curve_fragments:
        return CurveSuppressionResult(lines=lines, suppressed_line_ids=[])
    candidate_indices = [
        index
        for index, line in enumerate(lines)
        if line.length_analysis <= config.curve_max_segment_length_px
    ]
    if len(candidate_indices) < config.curve_min_component_lines:
        return CurveSuppressionResult(lines=lines, suppressed_line_ids=[])

    parent = list(range(len(candidate_indices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    gap = config.curve_neighbor_gap_px
    buckets: dict[tuple[int, int], list[int]] = {}
    for candidate_position, line_index in enumerate(candidate_indices):
        line = lines[line_index]
        for point in (line.p1_analysis, line.p2_analysis):
            cell_x = math.floor(point.x / gap)
            cell_y = math.floor(point.y / gap)
            for neighbor_x in range(cell_x - 1, cell_x + 2):
                for neighbor_y in range(cell_y - 1, cell_y + 2):
                    for previous_position in buckets.get((neighbor_x, neighbor_y), []):
                        previous = lines[candidate_indices[previous_position]]
                        if (
                            _axis_angle_deg(line, previous)
                            > config.curve_neighbor_angle_deg
                        ):
                            continue
                        if _endpoint_gap(line, previous) <= gap:
                            union(candidate_position, previous_position)
            buckets.setdefault((cell_x, cell_y), []).append(candidate_position)

    components: dict[int, list[int]] = {}
    for candidate_position, line_index in enumerate(candidate_indices):
        components.setdefault(find(candidate_position), []).append(line_index)
    suppressed_indices: set[int] = set()
    for component in components.values():
        if len(component) < config.curve_min_component_lines:
            continue
        directions = [lines[index].angle_rad for index in component]
        if _axial_direction_span_deg(directions) < config.curve_min_direction_span_deg:
            continue
        if (
            _longest_consecutive_direction_run(directions)
            < config.curve_min_contiguous_direction_bins
        ):
            continue
        suppressed_indices.update(component)
    return CurveSuppressionResult(
        lines=[line for index, line in enumerate(lines) if index not in suppressed_indices],
        suppressed_line_ids=[lines[index].line_id for index in sorted(suppressed_indices)],
    )


def _axis_angle_deg(first: LineRecord, second: LineRecord) -> float:
    difference = abs(math.degrees(first.angle_rad - second.angle_rad)) % 180.0
    return min(difference, 180.0 - difference)


def _endpoint_gap(first: LineRecord, second: LineRecord) -> float:
    return min(
        math.dist((a.x, a.y), (b.x, b.y))
        for a in (first.p1_analysis, first.p2_analysis)
        for b in (second.p1_analysis, second.p2_analysis)
    )


def _axial_direction_span_deg(directions_rad: list[float]) -> float:
    directions = sorted(math.degrees(direction) % 180.0 for direction in directions_rad)
    if len(directions) < 2:
        return 0.0
    gaps = [
        second - first for first, second in zip(directions, directions[1:])
    ] + [directions[0] + 180.0 - directions[-1]]
    return 180.0 - max(gaps)


def _longest_consecutive_direction_run(directions_rad: list[float]) -> int:
    bin_count = 18
    bins = sorted(
        {
            min(bin_count - 1, int((math.degrees(direction) % 180.0) / 10.0))
            for direction in directions_rad
        }
    )
    if not bins:
        return 0
    doubled = [*bins, *(value + bin_count for value in bins)]
    longest = 1
    current = 1
    for first, second in zip(doubled, doubled[1:]):
        current = current + 1 if second == first + 1 else 1
        longest = max(longest, current)
    return min(longest, len(bins))
