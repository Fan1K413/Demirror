from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "moge_geometry_shard", REPOSITORY_ROOT / "scripts" / "extract_moge_geometry_shard.py"
)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def _sample(index: int, label: int, split: str = "test") -> object:
    return worker.CacheSample(
        cache_index=index,
        path=Path(f"fixture-{index}.jpg"),
        archive="Recent_SDXL_Indoor",
        generator="sdxl",
        identifier=426 + index,
        label=label,
        scene="indoor",
        split=split,
        sha256="a" * 64,
    )


def test_select_samples_balances_labels_and_keeps_cache_order() -> None:
    samples = [_sample(0, 0), _sample(1, 0), _sample(2, 1), _sample(3, 1)]
    selected = worker.select_samples(
        samples, splits={"test"}, offset=0, limit=4, balanced=True
    )
    assert [sample.cache_index for sample in selected] == [0, 1, 2, 3]
    assert [sample.label for sample in selected].count(0) == 2
    assert [sample.label for sample in selected].count(1) == 2


def test_select_samples_never_leaks_unrequested_split() -> None:
    samples = [_sample(0, 0, "train"), _sample(1, 1, "test"), _sample(2, 0, "test")]
    selected = worker.select_samples(
        samples, splits={"test"}, offset=0, limit=2, balanced=False
    )
    assert [sample.cache_index for sample in selected] == [1, 2]
