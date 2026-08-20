"""Run a strict, CPU-only FerretNet research probe.

This script is deliberately outside the product package.  It verifies the
registered upstream revision, source files, checkpoint and image hashes before
executing a forward-only inference path in a short-lived subprocess.  It never
imports ``image_trust`` and cannot update web or origin-scoring policy.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


Z_95 = 1.959963984540054
DEFAULT_PROTOCOL = Path("research/records/2026-08-19/pixel/ferretnet_cpu_probe_protocol_v1.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
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
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected.lower()}, got {actual}")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _registered_upstream_root(protocol: Mapping[str, Any], repo_root: Path) -> Path:
    checkpoint_path = repo_root / str(protocol["checkpoint"]["relative_path"])
    # The checkpoint hierarchy is an official-source clone isolated below outputs/research.
    return checkpoint_path.parents[2]


def _verify_upstream(protocol: Mapping[str, Any], repo_root: Path) -> tuple[Path, Path]:
    upstream_root = _registered_upstream_root(protocol, repo_root)
    if not (upstream_root / ".git").exists():
        raise FileNotFoundError(f"Registered FerretNet source clone is missing: {upstream_root}")

    completed = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    actual_revision = completed.stdout.strip().lower()
    expected_revision = str(protocol["upstream"]["revision"]).lower()
    if actual_revision != expected_revision:
        raise ValueError(
            f"Upstream revision mismatch: expected {expected_revision}, got {actual_revision}"
        )

    for relative, digest in dict(protocol["upstream"]["source_files"]).items():
        _require_hash(upstream_root / relative, str(digest))

    checkpoint_path = repo_root / str(protocol["checkpoint"]["relative_path"])
    _require_hash(checkpoint_path, str(protocol["checkpoint"]["sha256"]))
    if checkpoint_path.stat().st_size != int(protocol["checkpoint"]["size_bytes"]):
        raise ValueError(f"Checkpoint byte-size mismatch: {checkpoint_path}")
    return upstream_root, checkpoint_path


def _normalise_state_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("FerretNet checkpoint must contain a model state dictionary")
    state = raw["model"]
    if not state:
        raise ValueError("FerretNet checkpoint state dictionary is empty")
    keys = [str(key) for key in state]
    has_prefix = [key.startswith("module.") for key in keys]
    if any(has_prefix) and not all(has_prefix):
        raise ValueError("FerretNet checkpoint mixes module.-prefixed and unprefixed keys")
    normalized = {
        str(key)[7:] if has_prefix[0] else str(key): value for key, value in state.items()
    }
    if len(normalized) != len(state):
        raise ValueError("FerretNet checkpoint key collision after prefix normalization")
    return normalized


def _load_model(protocol: Mapping[str, Any], upstream_root: Path, checkpoint_path: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    ferretnet = _load_module("demirror_ferretnet_upstream", upstream_root / "src/model/ferretnet.py")
    lpd = _load_module("demirror_ferretnet_lpd_upstream", upstream_root / "src/model/lpd.py")
    model = ferretnet.Ferret(
        in_channels=3,
        num_classes=1,
        dim=96,
        depths=[2, 2],
        lpd_func="median",
        window_size=3,
        lpd_dict=lpd.get_lpd_dict(),
    )
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # Older torch must fail rather than invoke unrestricted pickle loading.
        raise RuntimeError("PyTorch weights_only checkpoint loading is required for this research probe") from exc
    state = _normalise_state_dict(raw)
    expected = int(protocol["checkpoint"]["expected_state_tensor_count"])
    if len(state) != expected:
        raise ValueError(f"Checkpoint tensor count mismatch: expected {expected}, got {len(state)}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("Strict load returned incompatible FerretNet state")
    return model.to("cpu").eval(), {
        "strict": True,
        "checkpoint_state_tensor_count": len(state),
        "model_state_tensor_count": len(model.state_dict()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "missing_keys": [],
        "unexpected_keys": [],
    }


def _decode_variant(path: Path, profile: str) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    original_size = image.size
    if profile == "original_decode":
        return image
    if profile == "jpeg_reencode_quality=85":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    elif profile == "webp_reencode_quality=85":
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=85)
    elif profile == "resize_longest=1024_restore":
        longest = max(original_size)
        if longest > 1024:
            smaller = tuple(max(1, round(value * 1024 / longest)) for value in original_size)
            image = image.resize(smaller, Image.Resampling.LANCZOS)
            image = image.resize(original_size, Image.Resampling.LANCZOS)
        return image
    elif profile == "screenshot_raster_png_longest=1600":
        longest = max(original_size)
        if longest > 1600:
            size = tuple(max(1, round(value * 1600 / longest)) for value in original_size)
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    else:
        raise ValueError(f"Unsupported input profile: {profile}")
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        result = decoded.convert("RGB")
        result.load()
    return result


def _preprocess(image: Image.Image) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )(image).unsqueeze(0)


def _score(model: Any, image: Image.Image) -> float:
    import torch

    with torch.inference_mode():
        value = float(torch.sigmoid(model(_preprocess(image))).item())
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"FerretNet returned an invalid score: {value!r}")
    return value


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
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _wilson95(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    denominator = 1.0 + (Z_95 * Z_95) / total
    centre = proportion + (Z_95 * Z_95) / (2.0 * total)
    radius = Z_95 * math.sqrt(
        (proportion * (1.0 - proportion) + (Z_95 * Z_95) / (4.0 * total)) / total
    )
    return {
        "successes": successes,
        "total": total,
        "rate": proportion,
        "wilson_95_lower": max(0.0, (centre - radius) / denominator),
        "wilson_95_upper": min(1.0, (centre + radius) / denominator),
    }


def _records(manifest: Mapping[str, Any], image_root: Path, limit: int | None) -> Iterable[dict[str, str | int]]:
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Frozen manifest has no records")
    for manifest_index, entry in enumerate(raw_records[:limit]):
        if not isinstance(entry, dict):
            raise ValueError("Frozen manifest contains a non-object record")
        relative = entry.get("relative_path")
        digest = entry.get("asset_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("Frozen manifest record is missing relative_path or asset_sha256")
        path = image_root / relative
        _require_hash(path, digest)
        yield {
            "manifest_index": manifest_index,
            "relative_path": relative,
            "asset_sha256": digest,
            "path": str(path),
        }


def score(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    protocol = _read_json(args.protocol)
    _require_hash(args.protocol, _sha256(args.protocol))  # Force a readable, regular protocol file before work.
    _require_hash(
        repo_root / str(protocol["resource_probe"]["manifest"]),
        str(protocol["resource_probe"]["manifest_sha256"]),
    )
    upstream_root, checkpoint_path = _verify_upstream(protocol, repo_root)
    manifest = _read_json(repo_root / str(protocol["resource_probe"]["manifest"]))
    image_root = repo_root / str(protocol["resource_probe"]["image_root"])
    rows = list(_records(manifest, image_root, args.limit))
    if not rows:
        raise ValueError("No images selected for FerretNet probe")

    os.environ["OMP_NUM_THREADS"] = str(protocol["inference"]["cpu_threads"])
    os.environ["MKL_NUM_THREADS"] = str(protocol["inference"]["cpu_threads"])
    import torch

    torch.set_num_threads(int(protocol["inference"]["cpu_threads"]))
    model, load_audit = _load_model(protocol, upstream_root, checkpoint_path)
    timings: list[float] = []
    scored: list[dict[str, Any]] = []
    for row in rows:
        image = _decode_variant(Path(str(row["path"])), args.profile)
        start = time.perf_counter()
        value = _score(model, image)
        timings.append(time.perf_counter() - start)
        public_row: dict[str, Any] = {
            "manifest_index": int(row["manifest_index"]),
            "asset_sha256": row["asset_sha256"],
            "score": value,
        }
        if args.include_relative_path:
            public_row["relative_path"] = row["relative_path"]
        scored.append(public_row)
    peak = _peak_working_set_bytes()
    if peak is None:
        raise RuntimeError("Peak working-set observation unavailable; failing closed")
    return {
        "schema_version": "demirror-ferretnet-cpu-score-v1",
        "purpose": (
            "Forward scoring with source path retained only for an explicitly requested local diagnostic."
            if args.include_relative_path
            else "Forward scoring; rows intentionally omit labels, source classes, and source paths."
        ),
        "protocol_sha256": _sha256(args.protocol),
        "profile": args.profile,
        "model": {
            "upstream_revision": protocol["upstream"]["revision"],
            "checkpoint_sha256": protocol["checkpoint"]["sha256"],
            "checkpoint_size_bytes": protocol["checkpoint"]["size_bytes"],
            "load_audit": load_audit,
        },
        "resource_observation": {
            "cpu_threads": int(protocol["inference"]["cpu_threads"]),
            "batch_size": 1,
            "scored_count": len(scored),
            "p95_seconds_per_scored_view": _percentile(timings, 0.95),
            "elapsed_seconds_excluding_model_load": sum(timings),
            "observed_peak_working_set_bytes": peak,
        },
        "rows": scored,
    }


def _load_score_rows(path: Path) -> dict[str, float]:
    report = _read_json(path)
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid score report rows: {path}")
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid score row: {path}")
        digest, score_value = row.get("asset_sha256"), row.get("score")
        if not isinstance(digest, str) or not isinstance(score_value, (int, float)):
            raise ValueError(f"Invalid score row fields: {path}")
        result[digest] = float(score_value)
    return result


def _score_report(path: Path, protocol_sha256: str, expected_profile: str) -> tuple[dict[str, float], Mapping[str, Any]]:
    report = _read_json(path)
    if report.get("schema_version") != "demirror-ferretnet-cpu-score-v1":
        raise ValueError(f"Unexpected FerretNet score schema: {path}")
    if report.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Score report protocol mismatch: {path}")
    if report.get("profile") != expected_profile:
        raise ValueError(f"Score report profile mismatch: {path}")
    observation = report.get("resource_observation")
    if not isinstance(observation, dict):
        raise ValueError(f"Score report has no resource observation: {path}")
    return _load_score_rows(path), observation


def _tier1_baseline_hits(audit: Mapping[str, Any], profile_key: str) -> tuple[set[str], set[str]]:
    profile = audit["evaluation"]["profiles"][profile_key]["baseline_error_attribution"]
    generated_misses = {
        str(row["asset_sha256"]) for row in profile["generated_false_negatives"]
    }
    real_hits = {str(row["asset_sha256"]) for row in profile["real_false_positives"]}
    return generated_misses, real_hits


def _metric_rows(rows: Sequence[Mapping[str, Any]], hits: set[str], label: str) -> dict[str, float | int]:
    scoped = [row for row in rows if row["label"] == label]
    return _wilson95(sum(row["asset_sha256"] in hits for row in scoped), len(scoped))


def _group_metric_rows(rows: Sequence[Mapping[str, Any]], hits: set[str], label: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["label"] == label:
            groups.setdefault(str(row["model"]), []).append(row)
    return {
        name: _wilson95(sum(row["asset_sha256"] in hits for row in values), len(values))
        for name, values in sorted(groups.items())
    }


def audit_tier1(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_sha256 = _sha256(args.protocol)
    tier1 = protocol["labelled_elimination_tier1"]
    repo_root = args.repo_root.resolve()
    manifest_path = repo_root / str(tier1["manifest"])
    _require_hash(manifest_path, str(tier1["manifest_sha256"]))
    baseline_path = repo_root / str(tier1["registered_current_baseline_audit"])
    _require_hash(baseline_path, str(tier1["registered_current_baseline_audit_sha256"]))
    manifest = _read_json(manifest_path)
    raw_rows = manifest.get("records")
    if not isinstance(raw_rows, list) or len(raw_rows) != 160:
        raise ValueError("Tier-1 requires all 160 registered OpenFake records")
    rows: list[Mapping[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict) or not isinstance(row.get("asset_sha256"), str):
            raise ValueError("Invalid registered manifest row")
        if row.get("label") not in {"fake", "real"} or not isinstance(row.get("model"), str):
            raise ValueError("Tier-1 manifest requires frozen label and model fields")
        rows.append(row)

    score_paths = {
        "original_decode": args.original_score,
        "jpeg_reencode_quality=85": args.jpeg85_score,
    }
    expected = {str(row["asset_sha256"]) for row in rows}
    score_sets: dict[str, set[str]] = {}
    observations: dict[str, Mapping[str, Any]] = {}
    for profile, path in score_paths.items():
        scores, observation = _score_report(path, protocol_sha256, profile)
        if set(scores) != expected:
            raise ValueError(f"Score rows do not match frozen tier-1 manifest: {path}")
        score_sets[profile] = {
            asset_hash for asset_hash, score_value in scores.items()
            if score_value >= float(tier1["operation_point"])
        }
        observations[profile] = observation

    baseline = _read_json(baseline_path)
    evaluation: dict[str, Any] = {}
    gates = tier1["gates"]
    generator_slice_pass = True
    additional_generated_pass = True
    additional_real_pass = True
    for profile, baseline_profile in (("original_decode", "original"), ("jpeg_reencode_quality=85", "jpeg85")):
        generated_misses, baseline_real_hits = _tier1_baseline_hits(baseline, baseline_profile)
        candidate_hits = score_sets[profile]
        candidate_generated = _metric_rows(rows, candidate_hits, "fake")
        candidate_real = _metric_rows(rows, candidate_hits, "real")
        candidate_by_generator = _group_metric_rows(rows, candidate_hits, "fake")
        additional_generated = candidate_hits & generated_misses
        additional_real = candidate_hits - baseline_real_hits
        candidate_real_set = {str(row["asset_sha256"]) for row in rows if row["label"] == "real"}
        additional_real &= candidate_real_set
        generator_pass = all(
            float(result["rate"]) >= float(gates["each_generator_point_recall_minimum"])
            for result in candidate_by_generator.values()
        )
        generator_slice_pass = generator_slice_pass and generator_pass
        additional_generated_pass = additional_generated_pass and len(additional_generated) >= int(gates["additional_generated_hits_over_current_baseline_minimum"])
        additional_real_pass = additional_real_pass and len(additional_real) <= int(gates["additional_real_hits_over_current_baseline_maximum"])
        evaluation[profile] = {
            "candidate_generated": candidate_generated,
            "candidate_real": candidate_real,
            "candidate_by_generator": candidate_by_generator,
            "additional_generated_hits_over_current_baseline": len(additional_generated),
            "additional_real_hits_over_current_baseline": len(additional_real),
            "resource_observation": observations[profile],
        }

    agreement = sum(
        (asset_hash in score_sets["original_decode"])
        == (asset_hash in score_sets["jpeg_reencode_quality=85"])
        for asset_hash in expected
    ) / len(expected)
    maximum_peak = max(int(value["observed_peak_working_set_bytes"]) for value in observations.values())
    maximum_p95 = max(float(value["p95_seconds_per_scored_view"]) for value in observations.values())
    recall_lower_pass = all(
        float(evaluation[profile]["candidate_generated"]["wilson_95_lower"])
        >= float(gates["candidate_generated_recall_wilson_95_lower_minimum"])
        for profile in evaluation
    )
    real_upper_pass = all(
        float(evaluation[profile]["candidate_real"]["wilson_95_upper"])
        <= float(gates["candidate_real_false_positive_rate_wilson_95_upper_maximum"])
        for profile in evaluation
    )
    checks = {
        "candidate_generated_recall_wilson_lower": recall_lower_pass,
        "each_generator_point_recall": generator_slice_pass,
        "candidate_real_false_positive_rate_wilson_upper": real_upper_pass,
        "additional_generated_hits": additional_generated_pass,
        "additional_real_hits": additional_real_pass,
        "original_jpeg85_binary_agreement": agreement >= float(gates["original_jpeg85_binary_agreement_minimum"]),
        "peak_working_set": maximum_peak <= int(gates["maximum_peak_working_set_mib"]) * 1024 * 1024,
        "p95_seconds": maximum_p95 <= float(gates["maximum_p95_seconds_per_scored_view"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": "demirror-ferretnet-tier1-elimination-audit-v1",
        "purpose": "Pre-registered labelled elimination only; not product calibration or a deployment claim.",
        "protocol_sha256": protocol_sha256,
        "operation_point": tier1["operation_point"],
        "evaluation": evaluation,
        "cross_view_binary_agreement": agreement,
        "checks": checks,
        "screen_passed": passed,
        "deployment_eligible": False,
        "decision": "advance_to_five_variant_research_gate" if passed else "reject_before_five_variant_research_gate",
        "notes": [
            "This output uses the released 0.5 operating point and never selects a threshold from these data.",
            "The model checkpoint still lacks a separately explicit weight license; this audit cannot authorize product use."
        ],
    }


def probe(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    resource = protocol["resource_probe"]
    output_dir = args.output_dir.resolve()
    score_reports: dict[str, list[Path]] = {}
    for profile in resource["profiles"]:
        reports: list[Path] = []
        for run_index in range(1, int(resource["repeat_runs"]) + 1):
            output_path = output_dir / f"{profile.replace('=', '_').replace('/', '_')}-run{run_index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "score",
                "--repo-root", str(args.repo_root.resolve()),
                "--protocol", str(args.protocol.resolve()),
                "--profile", str(profile),
                "--limit", str(args.limit),
                "--output", str(output_path),
            ]
            subprocess.run(
                command,
                check=True,
                timeout=int(resource["subprocess_timeout_seconds"]),
            )
            reports.append(output_path)
        score_reports[str(profile)] = reports

    profile_checks: dict[str, Any] = {}
    maximum_peak = 0
    maximum_p95 = 0.0
    all_pass = True
    for profile, reports in score_reports.items():
        first, second = (_load_score_rows(path) for path in reports)
        if set(first) != set(second):
            raise ValueError(f"Probe runs selected different records for {profile}")
        largest_delta = max(abs(first[key] - second[key]) for key in first)
        observations = [_read_json(path)["resource_observation"] for path in reports]
        p95 = max(float(item["p95_seconds_per_scored_view"]) for item in observations)
        peak = max(int(item["observed_peak_working_set_bytes"]) for item in observations)
        maximum_peak = max(maximum_peak, peak)
        maximum_p95 = max(maximum_p95, p95)
        deterministic = largest_delta <= float(resource["score_absolute_difference_maximum"])
        profile_checks[profile] = {
            "run_reports": [str(path) for path in reports],
            "score_absolute_difference_maximum_observed": largest_delta,
            "deterministic": deterministic,
            "maximum_p95_seconds_per_scored_view": p95,
            "maximum_peak_working_set_bytes": peak,
        }
        all_pass = all_pass and deterministic

    memory_limit = int(resource["maximum_peak_working_set_mib"]) * 1024 * 1024
    checks = {
        "deterministic_forward": all_pass,
        "peak_working_set_within_limit": maximum_peak <= memory_limit,
        "p95_seconds_within_limit": maximum_p95 <= float(resource["maximum_p95_seconds_per_scored_view"]),
        "weight_license_allows_product": False,
    }
    result = {
        "schema_version": "demirror-ferretnet-cpu-probe-audit-v1",
        "protocol_sha256": _sha256(args.protocol),
        "selection": f"first {args.limit} frozen manifest records; labels omitted from all score reports",
        "profiles": profile_checks,
        "limits": {
            "maximum_peak_working_set_mib": resource["maximum_peak_working_set_mib"],
            "maximum_p95_seconds_per_scored_view": resource["maximum_p95_seconds_per_scored_view"],
            "score_absolute_difference_maximum": resource["score_absolute_difference_maximum"],
        },
        "checks": checks,
        "resource_probe_passed": checks["deterministic_forward"] and checks["peak_working_set_within_limit"] and checks["p95_seconds_within_limit"],
        "deployment_eligible": False,
        "next_step": "freeze_labelled_elimination_protocol" if all(checks[key] for key in checks if key != "weight_license_allows_product") else "stop_before_labelled_elimination_screen",
        "notes": [
            "The separately declared weight license remains unresolved; this does not prevent research measurement, but it blocks product integration.",
            "This is a resource probe only. It reports no labels, source families, accuracy, recall, false-positive rate, or proposed threshold."
        ],
    }
    _atomic_write_json(args.output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "probe"):
        command = subparsers.add_parser(name)
        command.add_argument("--repo-root", type=Path, default=Path("."))
        command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    score_parser = subparsers.choices["score"]
    score_parser.add_argument("--profile", required=True)
    score_parser.add_argument("--limit", type=int, default=12)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--include-relative-path", action="store_true")
    audit_parser = subparsers.add_parser("audit-tier1")
    audit_parser.add_argument("--repo-root", type=Path, default=Path("."))
    audit_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    audit_parser.add_argument("--original-score", type=Path, required=True)
    audit_parser.add_argument("--jpeg85-score", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    probe_parser = subparsers.choices["probe"]
    probe_parser.add_argument("--limit", type=int, default=12)
    probe_parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/ferretnet_cpu_probe_v1/runs"))
    probe_parser.add_argument("--output", type=Path, default=Path("outputs/research/ferretnet_cpu_probe_v1/ferretnet_cpu_probe_audit_v1.json"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if hasattr(args, "limit") and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.command == "score":
        _atomic_write_json(args.output, score(args))
    elif args.command == "probe":
        _atomic_write_json(args.output, probe(args))
    else:
        _atomic_write_json(args.output, audit_tier1(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
