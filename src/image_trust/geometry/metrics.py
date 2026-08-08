"""Line-set metrics used for applicability and evidence coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from image_trust.schemas import LineRecord


@dataclass(frozen=True)
class LineMetrics:
    line_count: int
    total_length_normalized: float
    spatial_coverage: float
    direction_entropy: float
    spatial_entropy: float
    occupied_cells: int


def compute_line_metrics(
    lines: list[LineRecord],
    image_size: tuple[int, int],
    grid_size: int,
    direction_bins: int,
) -> LineMetrics:
    width, height = image_size
    diagonal = math.hypot(width, height)
    if not lines:
        return LineMetrics(0, 0.0, 0.0, 0.0, 0.0, 0)

    cell_weights = np.zeros((grid_size, grid_size), dtype=float)
    direction_weights = np.zeros(direction_bins, dtype=float)
    total_length = 0.0
    for line in lines:
        line_weight = line.length_analysis * max(line.quality, 0.05)
        total_length += line.length_analysis
        direction_index = min(
            direction_bins - 1,
            int(line.angle_rad / math.pi * direction_bins),
        )
        direction_weights[direction_index] += line_weight
        touched = _cells_touched_by_line(
            line.p1_analysis.x,
            line.p1_analysis.y,
            line.p2_analysis.x,
            line.p2_analysis.y,
            width,
            height,
            grid_size,
        )
        for row, col in touched:
            cell_weights[row, col] += line_weight / max(len(touched), 1)

    occupied = int(np.count_nonzero(cell_weights))
    return LineMetrics(
        line_count=len(lines),
        total_length_normalized=total_length / diagonal,
        spatial_coverage=occupied / (grid_size * grid_size),
        direction_entropy=_normalized_entropy(direction_weights),
        spatial_entropy=_normalized_entropy(cell_weights.ravel()),
        occupied_cells=occupied,
    )


def _cells_touched_by_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    grid_size: int,
) -> set[tuple[int, int]]:
    sample_count = max(2, int(math.ceil(max(abs(x2 - x1), abs(y2 - y1)))) + 1)
    xs = np.linspace(x1, x2, sample_count)
    ys = np.linspace(y1, y2, sample_count)
    cols = np.clip((xs / max(width, 1) * grid_size).astype(int), 0, grid_size - 1)
    rows = np.clip((ys / max(height, 1) * grid_size).astype(int), 0, grid_size - 1)
    return set(zip(rows.tolist(), cols.tolist()))


def _normalized_entropy(weights: np.ndarray) -> float:
    total = float(weights.sum())
    nonzero = weights[weights > 0]
    if total <= 0 or len(nonzero) <= 1:
        return 0.0
    probabilities = nonzero / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(weights))

