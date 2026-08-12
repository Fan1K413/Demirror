"""Run an integrity-bound CPU screen for the SIDBench PatchCraft and NPR candidates.

This is an offline evaluation harness, not product integration.  It keeps the
released 0.5 operation point, verifies every registered input and upstream
artifact, runs one candidate per short-lived process, and emits a separate
summary whose gates cannot change the website or the origin score.
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
import random
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


PROFILES = ("original_decode", "jpeg_reencode_quality=85")
COMMUNITY_THRESHOLD = 0.5
NONESCAPE_THRESHOLD = 0.9260923266410828
Z_95 = 1.959963984540054


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream artifact identity, not security.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _require_hash(path: Path, expected: str, algorithm: str = "sha256") -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    actual = _sha256(path) if algorithm == "sha256" else _md5(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{algorithm.upper()} mismatch for {path}: expected {expected.lower()}, got {actual.lower()}"
        )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def _namespace_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _strict_npr_state_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("NPR checkpoint must contain a model state dictionary")
    state = raw["model"]
    if not state or any(not str(key).startswith("module.") for key in state):
        raise ValueError("Every NPR checkpoint tensor must have the registered module. prefix")
    normalized = {str(key)[7:]: value for key, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError("NPR checkpoint key collision after removing module. prefixes")
    return normalized


def _load_candidate_model(
    candidate: str, sidbench_root: Path, checkpoint: Path
) -> tuple[Any, dict[str, Any]]:
    import torch

    if candidate == "npr":
        _namespace_package("networks", sidbench_root / "networks")
        _load_module("networks.resnet_npr", sidbench_root / "networks" / "resnet_npr.py")
        module = _load_module("demirror_sidbench_npr", sidbench_root / "models" / "NPR.py")
        model = module.NPR()
        state = _strict_npr_state_dict(torch.load(checkpoint, map_location="cpu"))
        incompatible = model.model.load_state_dict(state, strict=True)
        adapter = "extract_model_remove_exact_module_prefix_strict_load"
    elif candidate == "patchcraft_rptc":
        _namespace_package("models", sidbench_root / "models")
        _load_module(
            "models.srm_filter_kernel", sidbench_root / "models" / "srm_filter_kernel.py"
        )
        module = _load_module("demirror_sidbench_rptc", sidbench_root / "models" / "RPTC.py")
        model = module.Net()
        model.load_weights(str(checkpoint))
        incompatible = None
        adapter = "upstream_strict_loader"
    else:
        raise ValueError(f"Unsupported candidate: {candidate}")
    state_count = len(model.model.state_dict()) if candidate == "npr" else len(model.state_dict())
    return model.to("cpu").eval(), {
        "adapter": adapter,
        "strict": True,
        "state_tensor_count": state_count,
        "missing_keys": [] if incompatible is None else list(incompatible.missing_keys),
        "unexpected_keys": [] if incompatible is None else list(incompatible.unexpected_keys),
        "complete": (
            incompatible is None
            or (not incompatible.missing_keys and not incompatible.unexpected_keys)
        ),
    }


def _decoded_variant(path: Path, profile: str) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    if profile == "original_decode":
        return image
    if profile != "jpeg_reencode_quality=85":
        raise ValueError(f"Unsupported input profile: {profile}")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        output = recompressed.convert("RGB")
        output.load()
    return output


def _npr_input(image: Image.Image) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )(image).unsqueeze(0)


def _edge_difference(image: Any) -> float:
    import torch

    r1, r2 = image[:, 0:-1, :], image[:, 1:, :]
    r3, r4 = image[:, :, 0:-1], image[:, :, 1:]
    r5, r6 = image[:, 0:-1, 0:-1], image[:, 1:, 1:]
    r7, r8 = image[:, 0:-1, 1:], image[:, 1:, 0:-1]
    return float(
        torch.sum(torch.abs(r1 - r2)).item()
        + torch.sum(torch.abs(r3 - r4)).item()
        + torch.sum(torch.abs(r5 - r6)).item()
        + torch.sum(torch.abs(r7 - r8)).item()
    )


def _patchcraft_input(image: Image.Image) -> Any:
    import torch
    from torchvision import transforms

    crop_size = 224
    patch_num = 3
    block_count = 2**patch_num
    patch_size = crop_size // block_count
    if min(image.size) < patch_size:
        image = transforms.Resize((patch_size, patch_size))(image)
    tensor = transforms.ToTensor()(image)
    random_crop = transforms.RandomCrop(patch_size)
    patches = [
        (cropped := random_crop(tensor), _edge_difference(cropped))
        for _ in range(block_count * block_count * 3)
    ]
    patches.sort(key=lambda item: item[1])
    template = torch.zeros(3, crop_size, crop_size)
    index = 0
    for row in range(block_count):
        for column in range(block_count):
            template[
                :, row * patch_size : (row + 1) * patch_size, column * patch_size : (column + 1) * patch_size
            ] = patches[index][0]
            index += 1
    poor = template.clone().unsqueeze(0)
    index = -1
    for row in range(block_count):
        for column in range(block_count):
            template[
                :, row * patch_size : (row + 1) * patch_size, column * patch_size : (column + 1) * patch_size
            ] = patches[index][0]
            index -= 1
    rich = template.clone().unsqueeze(0)
    return torch.cat((poor, rich), dim=0).unsqueeze(0)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _score(model: Any, candidate: str, image: Image.Image) -> float:
    import torch

    values = _npr_input(image) if candidate == "npr" else _patchcraft_input(image)
    with torch.inference_mode():
        prediction = model.predict(values)
    if not isinstance(prediction, list) or len(prediction) != 1:
        raise ValueError(f"Unexpected {candidate} prediction shape: {prediction!r}")
    score = float(prediction[0])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Non-finite or out-of-range {candidate} score: {score!r}")
    return score


def _records_from_manifest(repo_root: Path, cohort: str, registration: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_path = repo_root / str(registration["manifest"])
    _require_hash(manifest_path, str(registration["manifest_sha256"]))
    manifest = _read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != int(registration["count"]):
        raise ValueError(f"Unexpected record count in {manifest_path}")
    output: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record in {manifest_path}")
        label_text = str(record.get("label", "")).lower()
        if label_text not in {"fake", "real"}:
            raise ValueError(f"Unsupported label {label_text!r} in {manifest_path}")
        path = manifest_path.parent / str(record["relative_path"])
        _require_hash(path, str(record["asset_sha256"]))
        output.append(
            {
                "cohort": cohort,
                "row_idx": int(record["row_idx"]),
                "label": label_text,
                "source": str(record.get("model", "unknown")),
                "asset_sha256": str(record["asset_sha256"]),
                "path": path,
            }
        )
    if sum(row["label"] == "fake" for row in output) != int(registration["generated_count"]):
        raise ValueError(f"Generated count mismatch in {manifest_path}")
    if sum(row["label"] == "real" for row in output) != int(registration["real_count"]):
        raise ValueError(f"Real count mismatch in {manifest_path}")
    return output


def _verify_upstream(
    protocol: Mapping[str, Any], sidbench_root: Path, npr_upstream_root: Path, candidate: str
) -> None:
    for relative, expected in protocol["upstream"]["source_files"].items():
        _require_hash(sidbench_root / relative, str(expected))
    if candidate == "npr":
        for relative, expected in protocol["upstream"]["npr_source_files"].items():
            _require_hash(npr_upstream_root / relative, str(expected))


def score_candidate(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    registration = protocol["candidates"][args.candidate]
    checkpoint = args.repo_root / str(registration["checkpoint_relative_path"])
    _require_hash(checkpoint, str(registration["checkpoint_sha256"]))
    _require_hash(checkpoint, str(registration["checkpoint_md5"]), algorithm="md5")
    if checkpoint.stat().st_size != int(registration["checkpoint_size_bytes"]):
        raise ValueError(f"Checkpoint size mismatch: {checkpoint}")
    _verify_upstream(protocol, args.sidbench_root, args.npr_upstream_root, args.candidate)

    import torch

    torch.set_num_threads(int(protocol["execution"]["cpu_threads"]))
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model, checkpoint_load = _load_candidate_model(args.candidate, args.sidbench_root, checkpoint)
    model_load_seconds = time.perf_counter() - started
    cohorts = {
        name: _records_from_manifest(args.repo_root, name, protocol["data"]["cohorts"][name])
        for name in protocol["data"]["order"]
    }
    rows: list[dict[str, Any]] = []
    per_view_seconds: list[float] = []
    expected_total = (
        sum(len(records) for records in cohorts.values())
        * len(PROFILES)
        * len(registration["seeds"])
    )
    completed = 0
    scoring_started = time.perf_counter()
    for seed in registration["seeds"]:
        for profile in PROFILES:
            for cohort, records in cohorts.items():
                _set_seed(int(seed))
                for record in records:
                    image = _decoded_variant(record["path"], profile)
                    view_started = time.perf_counter()
                    score = _score(model, args.candidate, image)
                    elapsed = time.perf_counter() - view_started
                    per_view_seconds.append(elapsed)
                    completed += 1
                    rows.append(
                        {
                            "cohort": cohort,
                            "row_idx": record["row_idx"],
                            "label": record["label"],
                            "source": record["source"],
                            "asset_sha256": record["asset_sha256"],
                            "profile": profile,
                            "seed": int(seed),
                            "score": score,
                            "seconds": elapsed,
                        }
                    )
                    if completed == 1 or completed % 10 == 0 or completed == expected_total:
                        print(
                            f"candidate={args.candidate} scored={completed}/{expected_total} "
                            f"seed={seed} profile={profile} cohort={cohort} row={record['row_idx']}",
                            flush=True,
                        )
    peak = _peak_working_set_bytes()
    report = {
        "schema_version": "demirror-sidbench-candidate-score-v1",
        "purpose": "Frozen candidate scoring only; rows cannot modify product decisions.",
        "candidate": args.candidate,
        "protocol_sha256": protocol_hash,
        "operation_point": float(registration["operation_point"]),
        "seeds": [int(seed) for seed in registration["seeds"]],
        "primary_seed": int(registration["primary_seed"]),
        "profiles": list(PROFILES),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_load": checkpoint_load,
        "evaluator_sha256": _sha256(Path(__file__)),
        "upstream_revision": protocol["upstream"]["revision"],
        "resource": {
            "device": "cpu",
            "cpu_threads": int(protocol["execution"]["cpu_threads"]),
            "model_load_seconds": model_load_seconds,
            "scoring_seconds": time.perf_counter() - scoring_started,
            "scored_view_count": len(rows),
            "median_seconds_per_scored_view": statistics.median(per_view_seconds),
            "p95_seconds_per_scored_view": _percentile(per_view_seconds, 0.95),
            "maximum_seconds_per_scored_view": max(per_view_seconds),
            "peak_working_set_bytes": peak,
        },
        "rows": rows,
    }
    _atomic_write_json(args.output, report)
    return report


def _rows_for(
    report: Mapping[str, Any], *, seed: int, profile: str, label: str | None = None, cohort: str | None = None
) -> list[dict[str, Any]]:
    return [
        row
        for row in report["rows"]
        if int(row["seed"]) == seed
        and row["profile"] == profile
        and (label is None or row["label"] == label)
        and (cohort is None or row["cohort"] == cohort)
    ]


def _rate_for_rows(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, float | int]:
    return _wilson95(sum(float(row["score"]) >= threshold for row in rows), len(rows))


def _source_rates(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    return {source: _rate_for_rows(values, threshold) for source, values in sorted(grouped.items())}


def _binary_agreement(
    report: Mapping[str, Any], seed: int, threshold: float
) -> dict[str, float | int]:
    by_profile = {
        profile: {
            row["asset_sha256"]: float(row["score"]) >= threshold
            for row in _rows_for(report, seed=seed, profile=profile)
        }
        for profile in PROFILES
    }
    if set(by_profile[PROFILES[0]]) != set(by_profile[PROFILES[1]]):
        raise ValueError("Candidate profiles do not contain identical assets")
    total = len(by_profile[PROFILES[0]])
    agreed = sum(
        by_profile[PROFILES[0]][asset] == by_profile[PROFILES[1]][asset]
        for asset in by_profile[PROFILES[0]]
    )
    return {"agreed": agreed, "total": total, "rate": agreed / total}


def _seed_stability(report: Mapping[str, Any], threshold: float) -> dict[str, Any] | None:
    seeds = [int(seed) for seed in report["seeds"]]
    if len(seeds) == 1:
        return None
    results: dict[str, Any] = {}
    for profile in PROFILES:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in report["rows"]:
            if row["profile"] == profile:
                groups[str(row["asset_sha256"])].append(float(row["score"]))
        if not groups or any(len(values) != len(seeds) for values in groups.values()):
            raise ValueError(f"Incomplete cross-seed rows for {profile}")
        unanimous = sum(
            len({value >= threshold for value in values}) == 1 for values in groups.values()
        )
        ranges = [max(values) - min(values) for values in groups.values()]
        results[profile] = {
            "unanimous_binary_count": unanimous,
            "total": len(groups),
            "binary_agreement_rate": unanimous / len(groups),
            "median_score_range": statistics.median(ranges),
            "p95_score_range": _percentile(ranges, 0.95),
            "maximum_score_range": max(ranges),
        }
    return results


def _baseline_rows(
    repo_root: Path, protocol: Mapping[str, Any], profile: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    suffix = "original" if profile == "original_decode" else "jpeg85"
    reports = protocol["registered_baseline"]["reports"]
    output = []
    for detector in ("community", "nonescape"):
        registration = reports[f"{detector}_{suffix}"]
        path = repo_root / str(registration["path"])
        _require_hash(path, str(registration["sha256"]))
        rows = _read_json(path).get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Baseline report has no rows: {path}")
        output.append({str(row["asset_sha256"]): row for row in rows})
    if set(output[0]) != set(output[1]):
        raise ValueError(f"Baseline asset mismatch for {profile}")
    return output[0], output[1]


def _complementarity(
    repo_root: Path,
    protocol: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    profile: str,
    seed: int,
    threshold: float,
) -> dict[str, int]:
    community, nonescape = _baseline_rows(repo_root, protocol, profile)
    candidate = {
        str(row["asset_sha256"]): row
        for row in _rows_for(
            report,
            seed=seed,
            profile=profile,
            cohort="openfake_confirmation",
        )
    }
    if set(candidate) != set(community):
        raise ValueError(f"Candidate and baseline asset mismatch for {profile}")
    counts = {
        "baseline_generated_hits": 0,
        "candidate_generated_hits": 0,
        "candidate_additional_generated_hits": 0,
        "baseline_real_hits": 0,
        "candidate_real_hits": 0,
        "candidate_additional_real_hits": 0,
    }
    for asset, row in candidate.items():
        baseline_hit = (
            float(community[asset]["score"]) >= COMMUNITY_THRESHOLD
            or float(nonescape[asset]["score"]) >= NONESCAPE_THRESHOLD
        )
        candidate_hit = float(row["score"]) >= threshold
        prefix = "generated" if row["label"] == "fake" else "real"
        counts[f"baseline_{prefix}_hits"] += int(baseline_hit)
        counts[f"candidate_{prefix}_hits"] += int(candidate_hit)
        counts[f"candidate_additional_{prefix}_hits"] += int(candidate_hit and not baseline_hit)
    return counts


def _candidate_summary(
    repo_root: Path, protocol: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = str(report["candidate"])
    registration = protocol["candidates"][candidate]
    threshold = float(registration["operation_point"])
    seed = int(registration["primary_seed"])
    profile_summaries: dict[str, Any] = {}
    for profile in PROFILES:
        generated = _rows_for(
            report,
            seed=seed,
            profile=profile,
            label="fake",
            cohort="openfake_confirmation",
        )
        real = _rows_for(report, seed=seed, profile=profile, label="real")
        profile_summaries[profile] = {
            "openfake_generated": _rate_for_rows(generated, threshold),
            "openfake_generated_by_source": _source_rates(generated, threshold),
            "pooled_real": _rate_for_rows(real, threshold),
            "real_by_cohort": {
                cohort: _rate_for_rows(
                    _rows_for(report, seed=seed, profile=profile, label="real", cohort=cohort),
                    threshold,
                )
                for cohort in protocol["data"]["order"]
            },
            "complementarity": _complementarity(
                repo_root,
                protocol,
                report,
                profile=profile,
                seed=seed,
                threshold=threshold,
            ),
        }
    stability = _seed_stability(report, threshold)
    resource = report["resource"]
    gate = protocol["elimination_gate"]
    checks: dict[str, bool] = {}
    checks["model_load:strict_complete"] = bool(report.get("checkpoint_load", {}).get("strict")) and bool(
        report.get("checkpoint_load", {}).get("complete")
    )
    for profile, summary in profile_summaries.items():
        checks[f"{profile}:generated_recall_wilson_lower"] = (
            float(summary["openfake_generated"]["wilson_95_lower"])
            >= float(gate["openfake_generated_recall_wilson_95_lower_minimum"])
        )
        checks[f"{profile}:each_generator_recall"] = all(
            float(value["rate"]) >= float(gate["each_generator_point_recall_minimum"])
            for value in summary["openfake_generated_by_source"].values()
        )
        checks[f"{profile}:pooled_real_fpr_wilson_upper"] = (
            float(summary["pooled_real"]["wilson_95_upper"])
            <= float(gate["pooled_real_false_positive_rate_wilson_95_upper_maximum"])
        )
        checks[f"{profile}:additional_generated_hits"] = (
            int(summary["complementarity"]["candidate_additional_generated_hits"])
            >= int(gate["openfake_additional_generated_hits_over_registered_baseline_minimum"])
        )
    original = profile_summaries[PROFILES[0]]
    jpeg85 = profile_summaries[PROFILES[1]]
    checks["jpeg85:generated_recall_drop"] = (
        float(original["openfake_generated"]["rate"])
        - float(jpeg85["openfake_generated"]["rate"])
        <= float(gate["jpeg85_generated_recall_drop_maximum"])
    )
    checks["jpeg85:pooled_real_fpr_increase"] = (
        float(jpeg85["pooled_real"]["rate"])
        - float(original["pooled_real"]["rate"])
        <= float(gate["jpeg85_pooled_real_false_positive_rate_increase_maximum"])
    )
    agreement = _binary_agreement(report, seed, threshold)
    checks["original_jpeg85_binary_agreement"] = (
        float(agreement["rate"]) >= float(gate["original_jpeg85_binary_agreement_minimum"])
    )
    if stability is not None:
        checks["patchcraft_cross_seed_binary_agreement"] = all(
            float(values["binary_agreement_rate"])
            >= float(gate["patchcraft_cross_seed_binary_agreement_minimum"])
            for values in stability.values()
        )
        checks["patchcraft_cross_seed_score_range_p95"] = all(
            float(values["p95_score_range"])
            <= float(gate["patchcraft_cross_seed_score_range_p95_maximum"])
            for values in stability.values()
        )
    peak = resource.get("peak_working_set_bytes")
    checks["resource:peak_working_set"] = peak is not None and int(peak) <= int(
        protocol["execution"]["maximum_peak_working_set_mib"]
    ) * 1024 * 1024
    checks["resource:p95_seconds_per_view"] = float(resource["p95_seconds_per_scored_view"]) <= float(
        protocol["execution"]["maximum_p95_seconds_per_scored_view"]
    )
    screen_passed = all(checks.values())
    return {
        "candidate": candidate,
        "operation_point": threshold,
        "primary_seed": seed,
        "profile_metrics": profile_summaries,
        "original_jpeg85_binary_agreement": agreement,
        "cross_seed_stability": stability,
        "resource": resource,
        "gate_checks": checks,
        "screen_passed": screen_passed,
        "status": (
            "screen_passed_license_blocked"
            if screen_passed
            else "rejected_for_product_integration"
        ),
        "deployment_eligible": False,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    reports = {
        "patchcraft_rptc": _read_json(args.patchcraft_report),
        "npr": _read_json(args.npr_report),
    }
    for candidate, report in reports.items():
        if report.get("candidate") != candidate:
            raise ValueError(f"Candidate report mismatch: expected {candidate}")
        if report.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"Protocol hash mismatch in {candidate} report")
    evaluator_hashes = {str(report.get("evaluator_sha256")) for report in reports.values()}
    if evaluator_hashes != {_sha256(Path(__file__))}:
        raise ValueError("Candidate reports were not produced by this frozen evaluator")
    candidates = {
        candidate: _candidate_summary(args.repo_root, protocol, report)
        for candidate, report in reports.items()
    }
    summary = {
        "schema_version": "demirror-sidbench-candidate-screen-audit-v1",
        "created_at": "2026-08-12",
        "protocol_sha256": protocol_hash,
        "evaluator_sha256": _sha256(Path(__file__)),
        "raw_report_sha256": {
            "patchcraft_rptc": _sha256(args.patchcraft_report),
            "npr": _sha256(args.npr_report),
        },
        "deployment_eligible": False,
        "runtime_policy_changed": False,
        "candidates": candidates,
        "decision": {
            "product_integration": "not_allowed",
            "reasons": [
                "This is an elimination screen rather than a source-isolated deployment gate.",
                "SIDBench marks the checkpoint collection as mixed-upstream-licenses; candidate-specific weight rights remain unresolved.",
                "The website and registered origin score were not changed.",
            ],
            "candidate_next_step": {
                name: (
                    "resolve upstream weight license, then run a larger source-isolated blind gate"
                    if result["screen_passed"]
                    else "do not integrate; retain only the reproducible rejection record"
                )
                for name, result in candidates.items()
            },
        },
    }
    _atomic_write_json(args.output, summary)
    return summary


def _path(value: str) -> Path:
    return Path(value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score one candidate in an isolated process.")
    score.add_argument("--candidate", choices=("patchcraft_rptc", "npr"), required=True)
    score.add_argument("--repo-root", type=_path, default=Path.cwd())
    score.add_argument(
        "--protocol",
        type=_path,
        default=Path("research/records/2026-08-12/pixel/sidbench_patchcraft_npr_screen_protocol_v1.json"),
    )
    score.add_argument(
        "--sidbench-root",
        type=_path,
        default=Path("outputs/research/sidbench_candidate_v1"),
    )
    score.add_argument(
        "--npr-upstream-root",
        type=_path,
        default=Path("outputs/research/npr_upstream_candidate_v1"),
    )
    score.add_argument("--output", type=_path, required=True)

    summary = subparsers.add_parser("summarize", help="Apply the frozen gate to two completed reports.")
    summary.add_argument("--repo-root", type=_path, default=Path.cwd())
    summary.add_argument(
        "--protocol",
        type=_path,
        default=Path("research/records/2026-08-12/pixel/sidbench_patchcraft_npr_screen_protocol_v1.json"),
    )
    summary.add_argument("--patchcraft-report", type=_path, required=True)
    summary.add_argument("--npr-report", type=_path, required=True)
    summary.add_argument("--output", type=_path, required=True)

    args = parser.parse_args()
    if args.command == "score":
        report = score_candidate(args)
        print(json.dumps(report["resource"], sort_keys=True))
    else:
        report = summarize(args)
        print(
            json.dumps(
                {
                    candidate: {
                        "status": result["status"],
                        "screen_passed": result["screen_passed"],
                    }
                    for candidate, result in report["candidates"].items()
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    raise SystemExit(main())
