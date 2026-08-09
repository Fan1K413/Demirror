"""Run one bounded, local MoGe-2 CPU geometry probe.

This is deliberately a research probe, not a web-runtime integration.  It
loads a locally verified checkpoint in a short-lived process, caps image size
and token count, records process RSS, and writes only a compact JSON summary.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOGE_ROOT = PROJECT_ROOT / "data" / "vendor" / "MoGe"
DEFAULT_UTILS3D_ROOT = PROJECT_ROOT / "data" / "vendor" / "utils3d"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "moge-2-vits-normal" / "model.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        pass
    if os.name != "nt":
        return None
    try:
        from ctypes import wintypes

        class _ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
                ("private_usage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.WinDLL(
            "psapi", use_last_error=True
        ).GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        success = get_process_memory_info(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.working_set_size) if success else None
    except (AttributeError, OSError):
        return None


def _load_rgb(path: Path, maximum_edge: int) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        original_size = rgb.size
        scale = min(1.0, maximum_edge / max(original_size))
        if scale < 1.0:
            rgb = rgb.resize(
                (
                    max(1, round(original_size[0] * scale)),
                    max(1, round(original_size[1] * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        return np.asarray(rgb, dtype=np.uint8).copy(), original_size


def _finite_summary(values: torch.Tensor) -> dict[str, float | int]:
    array = values.detach().float().cpu().numpy()
    finite = np.isfinite(array)
    if not finite.any():
        return {"finite_count": 0}
    usable = array[finite]
    return {
        "finite_count": int(usable.size),
        "p05": float(np.quantile(usable, 0.05)),
        "p50": float(np.quantile(usable, 0.50)),
        "p95": float(np.quantile(usable, 0.95)),
    }


def probe(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    checkpoint = args.checkpoint.resolve()
    moge_root = args.moge_root.resolve()
    utils3d_root = args.utils3d_root.resolve()
    if not input_path.is_file():
        raise ValueError(f"input_not_found:{input_path}")
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint_not_found:{checkpoint}")
    if not (moge_root / "moge").is_dir() or not (utils3d_root / "utils3d").is_dir():
        raise ValueError("local_moge_or_utils3d_source_not_found")

    sys.path[:0] = [str(moge_root), str(utils3d_root)]
    from moge.model.v2 import MoGeModel

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    image, original_size = _load_rgb(input_path, args.maximum_edge)
    input_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
    rss_before_load = _rss_bytes()
    started = time.perf_counter()
    model = MoGeModel.from_pretrained(checkpoint).to("cpu").eval()
    rss_after_load = _rss_bytes()
    with warnings.catch_warnings():
        # MoGe's GPU-oriented autocast context is harmless on CPU, but PyTorch
        # emits a distracting warning because it falls back to float32.
        warnings.filterwarnings("ignore", message="In CPU autocast.*", category=UserWarning)
        output = model.infer(
            input_tensor,
            num_tokens=args.num_tokens,
            use_fp16=False,
            apply_mask=True,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    points = output.get("points")
    depth = output.get("depth")
    normal = output.get("normal")
    mask = output.get("mask")
    intrinsics = output.get("intrinsics")
    line_geometry: dict[str, Any] | None = None
    if args.include_line_features:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from image_trust.geometry_ai.features import extract_image_lines
        from image_trust.geometry_ai.moge_line_features import (
            MOGE_LINE_FEATURE_SCHEMA_VERSION,
            moge_line_geometry_features,
        )

        lines, line_image_size = extract_image_lines(input_path)
        if line_image_size != (int(image.shape[1]), int(image.shape[0])):
            raise ValueError("moge_and_line_analysis_sizes_do_not_match")
        line_geometry = {
            "schema_version": MOGE_LINE_FEATURE_SCHEMA_VERSION,
            "line_count": int(len(lines)),
            "features": dict(
                moge_line_geometry_features(
                    lines,
                    points.detach().float().cpu().numpy(),
                    None if normal is None else normal.detach().float().cpu().numpy(),
                    None if mask is None else mask.detach().cpu().numpy(),
                )
            ),
        }
    arrays_output = args.arrays_output.resolve() if args.arrays_output is not None else None
    if arrays_output is not None:
        arrays_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_output,
            points=None if points is None else points.detach().float().cpu().numpy(),
            depth=None if depth is None else depth.detach().float().cpu().numpy(),
            normal=None if normal is None else normal.detach().float().cpu().numpy(),
            mask=None if mask is None else mask.detach().cpu().numpy(),
            intrinsics=None if intrinsics is None else intrinsics.detach().float().cpu().numpy(),
        )
    return {
        "schema_version": "moge-geometry-probe-v1",
        "status": "ok",
        "purpose": "Single-image local CPU resource and output probe; not AI-origin evaluation.",
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "original_size": list(original_size),
            "probe_size": [int(image.shape[1]), int(image.shape[0])],
        },
        "model": {
            "source": "microsoft/MoGe MoGe-2 vits-normal",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "cpu_threads": args.cpu_threads,
            "num_tokens": args.num_tokens,
        },
        "runtime": {
            "elapsed_ms_including_load": elapsed_ms,
            "rss_before_load_bytes": rss_before_load,
            "rss_after_load_bytes": rss_after_load,
            "rss_after_inference_bytes": _rss_bytes(),
        },
        "outputs": {
            "arrays_output": str(arrays_output) if arrays_output is not None else None,
            "line_geometry": line_geometry,
            "point_map_shape": list(points.shape) if points is not None else None,
            "point_map": _finite_summary(points) if points is not None else None,
            "depth_shape": list(depth.shape) if depth is not None else None,
            "depth": _finite_summary(depth) if depth is not None else None,
            "normal_shape": list(normal.shape) if normal is not None else None,
            "normal": _finite_summary(normal) if normal is not None else None,
            "valid_mask_fraction": float(mask.float().mean().item()) if mask is not None else None,
            "intrinsics": intrinsics.detach().float().cpu().numpy().round(7).tolist() if intrinsics is not None else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arrays-output",
        type=Path,
        help="Optional .npz destination for geometry arrays; intended only for offline research.",
    )
    parser.add_argument(
        "--include-line-features",
        action="store_true",
        help="Summarize detected 2D lines in the predicted 3D geometry; offline research only.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--moge-root", type=Path, default=DEFAULT_MOGE_ROOT)
    parser.add_argument("--utils3d-root", type=Path, default=DEFAULT_UTILS3D_ROOT)
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
        result = probe(args)
    except (OSError, RuntimeError, ValueError) as error:
        result = {
            "schema_version": "moge-geometry-probe-v1",
            "status": "failed",
            "error": f"{type(error).__name__}:{error}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
