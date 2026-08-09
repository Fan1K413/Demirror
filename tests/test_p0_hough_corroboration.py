from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "p0_hough_corroboration", REPOSITORY_ROOT / "scripts" / "screen_p0_hough_corroboration.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_line_support_requires_aligned_overlapping_visible_segments() -> None:
    candidate = np.asarray([10.0, 10.0, 100.0, 10.0])
    aligned_fragment = np.asarray([45.0, 11.0, 90.0, 11.0])
    crossing = np.asarray([55.0, 0.0, 55.0, 40.0])
    distant_parallel = np.asarray([10.0, 40.0, 100.0, 40.0])

    assert module._line_supports(candidate, aligned_fragment)
    assert not module._line_supports(candidate, crossing)
    assert not module._line_supports(candidate, distant_parallel)
