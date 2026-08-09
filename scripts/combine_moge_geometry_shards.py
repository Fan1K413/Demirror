"""Verify and combine complete MoGe geometry shards into a classifier cache.

The companion extractor intentionally writes small, short-lived shards.  This
combiner refuses partial coverage, overlapping rows, changed source images, or
mixed model/settings before it writes a cache that a classifier could consume.
It is therefore safe to run diagnostics on a few shards without accidentally
turning them into a purported cross-generator result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceRecord:
    cache_index: int
    payload: dict[str, Any]
    serialized: str


@dataclass(frozen=True)
class LoadedShard:
    path: Path
    summary: dict[str, Any]
    features: np.ndarray
    records: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_records(path: Path) -> list[SourceRecord]:
    with np.load(path, allow_pickle=False) as archive:
        if "records" not in archive.files:
            raise ValueError("source_cache_missing_records")
        raw_records = list(archive["records"])
    records: list[SourceRecord] = []
    for index, raw in enumerate(raw_records):
        serialized = str(raw)
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError(f"source_record_invalid_json_at_{index}") from error
        records.append(SourceRecord(index, payload, serialized))
    return records


def load_shard(path: Path) -> LoadedShard:
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "records", "summary"}
        if not required.issubset(archive.files):
            raise ValueError(f"shard_missing_keys:{path.name}")
        try:
            summary = json.loads(str(archive["summary"]))
            records = [json.loads(str(raw)) for raw in archive["records"]]
        except json.JSONDecodeError as error:
            raise ValueError(f"shard_invalid_json:{path.name}") from error
        features = np.asarray(archive["features"], dtype=np.float64)
    if summary.get("schema_version") != "moge-line-geometry-shard-v1":
        raise ValueError(f"shard_schema_mismatch:{path.name}")
    if summary.get("status") != "ok":
        raise ValueError(f"shard_not_successful:{path.name}")
    names = summary.get("feature_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"shard_feature_schema_missing:{path.name}")
    if features.ndim != 2 or features.shape != (len(records), len(names)):
        raise ValueError(f"shard_feature_record_mismatch:{path.name}")
    if not np.isfinite(features).all():
        raise ValueError(f"shard_nonfinite_features:{path.name}")
    if int(summary.get("success_count", -1)) != len(records):
        raise ValueError(f"shard_success_count_mismatch:{path.name}")
    return LoadedShard(path, summary, features, records)


def combine(
    source_cache: Path, source_records: list[SourceRecord], shards: list[LoadedShard]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Return complete feature rows in source-cache order or reject the run."""

    if not shards:
        raise ValueError("no_shards_supplied")
    source_sha256 = _sha256(source_cache)
    canonical = shards[0].summary
    required_match = ("cache_sha256", "feature_schema_version", "feature_names", "model", "request")
    for shard in shards:
        for key in required_match:
            if shard.summary.get(key) != canonical.get(key):
                raise ValueError(f"mixed_shard_{key}:{shard.path.name}")
        if shard.summary.get("cache_sha256") != source_sha256:
            raise ValueError(f"shard_source_cache_hash_mismatch:{shard.path.name}")

    by_index: dict[int, np.ndarray] = {}
    for shard in shards:
        for row, record in zip(shard.features, shard.records):
            try:
                index = int(record["cache_index"])
                source = source_records[index]
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ValueError(f"shard_invalid_cache_index:{shard.path.name}") from error
            if index in by_index:
                raise ValueError(f"duplicate_cache_index:{index}")
            expected_sha256 = str(source.payload.get("sha256", ""))
            if record.get("expected_sha256") != expected_sha256 or record.get("input_sha256") != expected_sha256:
                raise ValueError(f"source_image_hash_mismatch_at:{index}")
            for key in ("archive", "generator", "identifier", "label", "scene", "split"):
                if record.get(key) != source.payload.get(key):
                    raise ValueError(f"source_record_mismatch_{key}_at:{index}")
            by_index[index] = row

    missing = [record.cache_index for record in source_records if record.cache_index not in by_index]
    if missing:
        raise ValueError(f"incomplete_shard_coverage:{len(missing)}_rows_missing")
    rows = np.stack([by_index[index] for index in range(len(source_records))]).astype(np.float32)
    metadata = {
        "schema_version": "moge-line-geometry-cache-v1",
        "purpose": "Complete, hash-verified geometry-only candidate feature cache; not an AI-origin model.",
        "source_cache": str(source_cache),
        "source_cache_sha256": source_sha256,
        "feature_schema_version": canonical["feature_schema_version"],
        "feature_names": canonical["feature_names"],
        "model": canonical["model"],
        "request": canonical["request"],
        "shard_count": len(shards),
        "sample_count": len(source_records),
    }
    return rows, [record.serialized for record in source_records], metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    source_cache = args.source_cache.resolve()
    try:
        source_records = load_source_records(source_cache)
        shards = [load_shard(path.resolve()) for path in args.shards]
        features, records, metadata = combine(source_cache, source_records, shards)
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}:{error}"}))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        relations=features,
        records=np.asarray(records),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({"status": "ok", **metadata}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
