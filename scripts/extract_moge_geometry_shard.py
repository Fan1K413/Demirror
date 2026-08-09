"""Extract bounded MoGe line-geometry features from one cache shard.

This worker exists because the MoGe CPU model is intentionally kept out of the
web server.  Run it in short-lived shards, then combine the resulting ``.npz``
files in a separate, inexpensive evaluator.  It consumes no RGB values after
the model inference and never emits an AI-origin judgement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "outputs" / "deeplsd_geometry_v1" / "line_cache.npz"
DEFAULT_MOGE_ROOT = PROJECT_ROOT / "data" / "vendor" / "MoGe"
DEFAULT_UTILS3D_ROOT = PROJECT_ROOT / "data" / "vendor" / "utils3d"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "moge-2-vits-normal" / "model.pt"


@dataclass(frozen=True)
class CacheSample:
    cache_index: int
    path: Path
    archive: str
    generator: str
    identifier: int
    label: int
    scene: str
    split: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(cache_path: Path) -> list[CacheSample]:
    with np.load(cache_path, allow_pickle=False) as cache:
        if "records" not in cache.files:
            raise ValueError("cache_missing_records")
        raw_records = list(cache["records"])
    samples: list[CacheSample] = []
    for index, raw in enumerate(raw_records):
        try:
            row = json.loads(str(raw))
            samples.append(
                CacheSample(
                    cache_index=index,
                    path=PROJECT_ROOT / Path(str(row["path"])),
                    archive=str(row["archive"]),
                    generator=str(row["generator"]),
                    identifier=int(row["identifier"]),
                    label=int(row["label"]),
                    scene=str(row["scene"]),
                    split=str(row["split"]),
                    sha256=str(row["sha256"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"cache_record_invalid_at_{index}") from error
    return samples


def select_samples(
    samples: list[CacheSample],
    *,
    splits: set[str],
    offset: int,
    limit: int,
    balanced: bool,
) -> list[CacheSample]:
    """Choose a deterministic bounded shard without mixing unrequested splits."""

    if offset < 0 or limit < 1:
        raise ValueError("offset must be nonnegative and limit must be positive")
    selected = [sample for sample in samples if sample.split in splits]
    if not selected:
        raise ValueError("no_samples_match_requested_splits")
    if not balanced:
        return selected[offset : offset + limit]
    groups = {label: [sample for sample in selected if sample.label == label] for label in (0, 1)}
    if not groups[0] or not groups[1]:
        raise ValueError("balanced_shard_requires_both_labels")
    real_count = limit // 2
    generated_count = limit - real_count
    output = groups[0][offset : offset + real_count] + groups[1][offset : offset + generated_count]
    return sorted(output, key=lambda item: item.cache_index)


def _load_analysis_rgb(path: Path, maximum_edge: int) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, maximum_edge / max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if (resized_width, resized_height) != (width, height):
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    # PIL-backed arrays can be contiguous but read-only; make the ownership
    # explicit before passing the pixels to PyTorch.
    return np.array(rgb, dtype=np.uint8, copy=True, order="C")


def _record(sample: CacheSample, input_sha256: str) -> dict[str, Any]:
    return {
        "cache_index": sample.cache_index,
        "path": str(sample.path),
        "archive": sample.archive,
        "generator": sample.generator,
        "identifier": sample.identifier,
        "label": sample.label,
        "scene": sample.scene,
        "split": sample.split,
        "expected_sha256": sample.sha256,
        "input_sha256": input_sha256,
    }


def extract(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    cache_path = args.cache.resolve()
    checkpoint = args.checkpoint.resolve()
    moge_root = args.moge_root.resolve()
    utils3d_root = args.utils3d_root.resolve()
    if not cache_path.is_file():
        raise ValueError(f"cache_not_found:{cache_path}")
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint_not_found:{checkpoint}")
    if not (moge_root / "moge").is_dir() or not (utils3d_root / "utils3d").is_dir():
        raise ValueError("local_moge_or_utils3d_source_not_found")
    selected = select_samples(
        _records(cache_path),
        splits=set(args.splits),
        offset=args.offset,
        limit=args.limit,
        balanced=args.balanced,
    )
    if not selected:
        raise ValueError("requested_shard_is_empty")

    sys.path[:0] = [str(PROJECT_ROOT / "src"), str(moge_root), str(utils3d_root)]
    from image_trust.geometry_ai.features import extract_image_lines
    from image_trust.geometry_ai.moge_line_features import (
        MOGE_LINE_FEATURE_SCHEMA_VERSION,
        moge_line_feature_names,
        moge_line_geometry_features,
    )
    from moge.model.v2 import MoGeModel

    cv2.setNumThreads(1)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model = MoGeModel.from_pretrained(checkpoint).to("cpu").eval()
    load_elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample in selected:
        sample_started = time.perf_counter()
        try:
            if not sample.path.is_file():
                raise ValueError("sample_not_found")
            observed_sha256 = _sha256(sample.path)
            if observed_sha256 != sample.sha256:
                raise ValueError("sample_sha256_mismatch")
            rgb = _load_analysis_rgb(sample.path, args.maximum_edge)
            input_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="In CPU autocast.*", category=UserWarning)
                output = model.infer(
                    input_tensor,
                    num_tokens=args.num_tokens,
                    use_fp16=False,
                    apply_mask=True,
                )
            points = output.get("points")
            if points is None:
                raise RuntimeError("moge_output_missing_points")
            normal = output.get("normal")
            mask = output.get("mask")
            lines, line_size = extract_image_lines(sample.path)
            expected_size = (int(rgb.shape[1]), int(rgb.shape[0]))
            if line_size != expected_size:
                raise RuntimeError("line_and_moge_analysis_sizes_do_not_match")
            values = moge_line_geometry_features(
                lines,
                points.detach().float().cpu().numpy(),
                None if normal is None else normal.detach().float().cpu().numpy(),
                None if mask is None else mask.detach().cpu().numpy(),
            )
            rows.append(np.asarray(list(values.values()), dtype=np.float32))
            record = _record(sample, observed_sha256)
            record.update(
                {
                    "line_count": int(len(lines)),
                    "elapsed_ms": round((time.perf_counter() - sample_started) * 1000.0, 3),
                }
            )
            records.append(record)
        except (OSError, RuntimeError, ValueError) as error:
            failures.append(
                {
                    "cache_index": sample.cache_index,
                    "path": str(sample.path),
                    "error": f"{type(error).__name__}:{error}",
                }
            )

    summary = {
        "schema_version": "moge-line-geometry-shard-v1",
        "purpose": "Bounded offline extraction of geometry-only candidate features; not an AI-origin detector.",
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "feature_schema_version": MOGE_LINE_FEATURE_SCHEMA_VERSION,
        "feature_names": moge_line_feature_names(),
        "request": {
            "splits": sorted(set(args.splits)),
            "offset": args.offset,
            "limit": args.limit,
            "balanced": args.balanced,
            "maximum_edge": args.maximum_edge,
            "num_tokens": args.num_tokens,
            "cpu_threads": args.cpu_threads,
        },
        "model": {
            "source": "microsoft/MoGe MoGe-2 vits-normal",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "load_elapsed_ms": load_elapsed_ms,
        },
        "selected_count": len(selected),
        "success_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
    }
    return summary, np.stack(rows) if rows else np.empty((0, len(moge_line_feature_names())), dtype=np.float32), records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--moge-root", type=Path, default=DEFAULT_MOGE_ROOT)
    parser.add_argument("--utils3d-root", type=Path, default=DEFAULT_UTILS3D_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--maximum-edge", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.maximum_edge < 64:
        parser.error("--maximum-edge must be at least 64")
    if args.num_tokens < 1200:
        parser.error("--num-tokens must be at least 1200")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")
    try:
        summary, features, records = extract(args)
        summary["status"] = "ok"
    except (OSError, RuntimeError, ValueError) as error:
        summary = {
            "schema_version": "moge-line-geometry-shard-v1",
            "status": "failed",
            "error": f"{type(error).__name__}:{error}",
        }
        features = np.empty((0, 0), dtype=np.float32)
        records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        records=np.asarray([json.dumps(record, sort_keys=True) for record in records]),
        summary=np.asarray(json.dumps(summary, sort_keys=True)),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
