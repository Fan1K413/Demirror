"""Inspect the official LOTA source before any checkpoint is accepted.

This research-only utility intentionally stops *before* detector scoring.  It
locks the upstream source revision, checks the CPU adapter can construct the
published architecture without reaching the model zoo, and records the still
unresolved released-weight provenance.  It never imports the Demirror product
package or changes web/origin-scoring behaviour.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required registered source is missing: {path}")
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip().lower()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import registered source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if sys.platform == "darwin" else value * 1024
        except (ImportError, OSError, ValueError):
            return None

    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.PeakWorkingSetSize)


def _verify_upstream(protocol: Mapping[str, Any], repo_root: Path) -> Path:
    upstream = repo_root / str(protocol["upstream"]["relative_path"])
    if not (upstream / ".git").is_dir():
        raise FileNotFoundError(f"Registered upstream clone is missing: {upstream}")
    actual_revision = _git_head(upstream)
    expected_revision = str(protocol["upstream"]["revision"]).lower()
    if actual_revision != expected_revision:
        raise ValueError(
            f"Upstream revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    for relative, expected_hash in dict(protocol["upstream"]["source_files"]).items():
        _require_hash(upstream / relative, str(expected_hash))
    return upstream


def _fixed_smoke_image() -> Image.Image:
    """Return a deterministic, non-source image for an adapter-only smoke test."""
    height, width = 320, 448
    y, x = np.indices((height, width), dtype=np.uint16)
    rgb = np.stack(
        (
            (x * 3 + y) % 256,
            (x + y * 5) % 256,
            (x * 7 + y * 11) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _build_model_without_model_zoo(model_module: Any) -> Any:
    """Construct the published architecture without its unconditional download.

    At the registered revision, ``model.model(pretrain=False)`` still calls
    ``resnet50(pretrained=True)`` internally.  Temporarily guarding the symbol
    inside this one dynamically loaded module is narrower than modifying a
    source checkout or allowing a model-zoo side effect.  The guard is always
    restored before returning.
    """
    original_resnet50 = model_module.resnet50

    def _without_pretrained_weights(*_args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original_resnet50(**kwargs)

    model_module.resnet50 = _without_pretrained_weights
    try:
        return model_module.model()
    finally:
        model_module.resnet50 = original_resnet50


def _run_cpu_smoke(protocol: Mapping[str, Any], upstream: Path) -> dict[str, Any]:
    """Exercise only construction and one forward pass with random initial weights.

    ``model.model(pretrain=False)`` still requests ImageNet weights in the
    registered source.  The isolated, temporary guard in
    :func:`_build_model_without_model_zoo` is therefore required.  Released
    LOTA detector weights, when their provenance is closed, replace all model
    tensors.  No network call is allowed here.
    """
    import torch
    from torchvision import transforms

    torch.set_num_threads(int(protocol["cpu_smoke"]["cpu_threads"]))
    torch.set_num_interop_threads(1)
    seed = int(protocol["cpu_smoke"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_module = _load_module("demirror_lota_model_upstream", upstream / "model.py")
    bit_patch_module = _load_module("demirror_lota_bit_patch_upstream", upstream / "bit_patch.py")
    started = time.perf_counter()
    network = _build_model_without_model_zoo(model_module).to("cpu").eval()
    constructed_seconds = time.perf_counter() - started

    image = _fixed_smoke_image()
    torch.manual_seed(seed)
    patch = bit_patch_module.bit_patch(
        image,
        int(protocol["cpu_smoke"]["img_height"]),
        str(protocol["cpu_smoke"]["bit_mode"]),
        int(protocol["cpu_smoke"]["patch_size"]),
        str(protocol["cpu_smoke"]["patch_mode"]),
    )
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(Image.fromarray(patch)).unsqueeze(0)

    started = time.perf_counter()
    with torch.inference_mode():
        output = network(tensor)
    forward_seconds = time.perf_counter() - started
    value = float(output.squeeze().item())
    if output.shape != (1, 1) or not math.isfinite(value):
        raise ValueError(f"Unexpected CPU smoke output: shape={tuple(output.shape)}, value={value!r}")
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    expected_parameters = int(protocol["cpu_smoke"]["expected_parameter_count"])
    if parameter_count != expected_parameters:
        raise ValueError(
            f"LOTA parameter count mismatch: expected {expected_parameters}, got {parameter_count}"
        )
    return {
        "cpu_threads": int(protocol["cpu_smoke"]["cpu_threads"]),
        "random_seed": seed,
        "model_constructor": "model.model() with temporary module-local resnet50(pretrained=False) guard",
        "pretrained_model_zoo_download": False,
        "input": {
            "source": "deterministic synthetic gradient; no labelled or user image",
            "bit_mode": str(protocol["cpu_smoke"]["bit_mode"]),
            "patch_size": int(protocol["cpu_smoke"]["patch_size"]),
            "patch_mode": str(protocol["cpu_smoke"]["patch_mode"]),
            "output_shape": list(tensor.shape),
        },
        "model_output_shape": list(output.shape),
        "model_output_logit": value,
        "parameter_count": parameter_count,
        "construction_seconds": constructed_seconds,
        "forward_seconds": forward_seconds,
        "peak_working_set_bytes": _peak_working_set_bytes(),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    upstream = _verify_upstream(protocol, args.repo_root)
    smoke = _run_cpu_smoke(protocol, upstream)
    requirements = (upstream / "requirements.txt").read_text(encoding="utf-8")
    report = {
        "schema_version": "demirror-lota-cpu-preflight-audit-v1",
        "created_at": str(protocol["created_at"]),
        "purpose": "Source and CPU-adapter preflight only; no released detector checkpoint is loaded or scored.",
        "protocol_sha256": protocol_hash,
        "upstream": {
            "repository": protocol["upstream"]["repository"],
            "revision": _git_head(upstream),
            "repository_license": protocol["upstream"]["repository_license"],
            "source_files": dict(protocol["upstream"]["source_files"]),
            "official_requirements": requirements.splitlines(),
        },
        "stock_entrypoint_assessment": {
            "stock_test_uses_cuda": True,
            "stock_model_default_requests_imagenet_weights": True,
            "conclusion": "The stock evaluation entrypoint is not CPU-safe. The isolated adapter proves only construction with pretrain=False and a synthetic forward pass.",
        },
        "released_weight_gate": dict(protocol["released_weight_gate"]),
        "cpu_smoke": smoke,
        "status": "source_preflight_passed_checkpoint_screen_blocked",
        "deployment_eligible": False,
        "runtime_policy_changed": False,
        "next_required_step": "Obtain the authors' released checkpoint with explicit permission or documented terms, record its stable file hash and size in a new immutable protocol, then run source-isolated original/JPEG/WebP/resize/screenshot screening.",
    }
    _atomic_write_json(args.output, report)
    return report


def _path(value: str) -> Path:
    return Path(value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=_path, default=Path.cwd())
    parser.add_argument(
        "--protocol",
        type=_path,
        default=Path("research/records/2026-08-19/pixel/lota_cpu_preflight_protocol_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=_path,
        default=Path("research/records/2026-08-19/pixel/lota_cpu_preflight_audit_v1.json"),
    )
    args = parser.parse_args()
    report = preflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "parameter_count": report["cpu_smoke"]["parameter_count"],
                "peak_working_set_bytes": report["cpu_smoke"]["peak_working_set_bytes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
