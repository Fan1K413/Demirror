from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "combine_moge_geometry_shards",
    REPOSITORY_ROOT / "scripts" / "combine_moge_geometry_shards.py",
)
assert SPEC is not None and SPEC.loader is not None
combiner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = combiner
SPEC.loader.exec_module(combiner)


def _write_source(path: Path) -> list[dict[str, object]]:
    records = [
        {
            "archive": "Recent_SDXL_Indoor",
            "generator": "sdxl",
            "identifier": 426 + index,
            "label": index % 2,
            "scene": "indoor",
            "split": "test",
            "sha256": f"{index + 1:064x}",
        }
        for index in range(2)
    ]
    np.savez_compressed(path, records=np.asarray([json.dumps(record) for record in records]))
    return records


def _write_shard(path: Path, source_path: Path, records: list[dict[str, object]], indices: list[int]) -> None:
    digest = combiner._sha256(source_path)
    summary = {
        "schema_version": "moge-line-geometry-shard-v1",
        "status": "ok",
        "success_count": len(indices),
        "cache_sha256": digest,
        "feature_schema_version": "moge-line-geometry-v1",
        "feature_names": ["a", "b"],
        "model": {"checkpoint_sha256": "m"},
        "request": {"maximum_edge": 256},
    }
    payload = []
    for index in indices:
        source = records[index]
        payload.append(
            {
                "cache_index": index,
                "expected_sha256": source["sha256"],
                "input_sha256": source["sha256"],
                **{key: source[key] for key in ("archive", "generator", "identifier", "label", "scene", "split")},
            }
        )
    np.savez_compressed(
        path,
        features=np.asarray([[index, index + 0.5] for index in indices], dtype=np.float32),
        records=np.asarray([json.dumps(record) for record in payload]),
        summary=np.asarray(json.dumps(summary)),
    )


def test_combine_requires_complete_nonoverlapping_coverage(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    records = _write_source(source_path)
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    _write_shard(first_path, source_path, records, [0])
    _write_shard(second_path, source_path, records, [1])
    rows, serialized, metadata = combiner.combine(
        source_path,
        combiner.load_source_records(source_path),
        [combiner.load_shard(first_path), combiner.load_shard(second_path)],
    )
    assert rows.tolist() == [[0.0, 0.5], [1.0, 1.5]]
    assert len(serialized) == 2
    assert metadata["sample_count"] == 2


def test_combine_rejects_partial_coverage(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    records = _write_source(source_path)
    shard_path = tmp_path / "partial.npz"
    _write_shard(shard_path, source_path, records, [0])
    try:
        combiner.combine(
            source_path,
            combiner.load_source_records(source_path),
            [combiner.load_shard(shard_path)],
        )
    except ValueError as error:
        assert "incomplete_shard_coverage" in str(error)
    else:
        raise AssertionError("partial shards must never form a classifier cache")
