from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p0_x_aigd_edge_shape",
    REPOSITORY_ROOT / "scripts" / "evaluate_p0_x_aigd_edge_shape.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_polygon_and_line_overlap_marks_a_line_inside_buffer() -> None:
    size = (40, 30)
    polygon = np.asarray([[18, 10], [24, 10], [24, 16], [18, 16]], dtype=np.int32)
    target = module._dilate(module.polygon_mask([polygon], size), radius=2)
    line = {
        "p1_analysis": {"x": 4.0, "y": 9.0},
        "p2_analysis": {"x": 35.0, "y": 9.0},
    }
    pixels = module.line_mask([line], size)

    assert module._overlap_rate(pixels, target) > 0.0


def test_scaled_polygons_follow_canonical_resize() -> None:
    polygons = [[[10.0, 5.0], [20.0, 5.0], [20.0, 15.0]]]

    scaled = module._scaled_polygons(polygons, (40, 20), (20, 10))

    assert scaled[0].tolist() == [[5, 2], [10, 2], [10, 8]]
