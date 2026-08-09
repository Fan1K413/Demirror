from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/download_bfree_weights.py")
    spec = importlib.util.spec_from_file_location("download_bfree_weights", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ranges_cover_the_download_without_gaps() -> None:
    module = _load_module()
    ranges = module.iter_ranges(10, 4)
    assert [(item.start, item.end) for item in ranges] == [(0, 3), (4, 7), (8, 9)]
    assert sum(item.size for item in ranges) == 10


def test_byte_range_filenames_are_stable() -> None:
    module = _load_module()
    assert module.ByteRange(0, 1_048_575).filename == "000000000000-000001048575.part"
