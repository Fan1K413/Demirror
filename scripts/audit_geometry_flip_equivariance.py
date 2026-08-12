"""Audit PerspectiveFields horizontal-flip equivariance across generators.

The source-neutral measurement is fixed in ``geometry_ai.equivariance``.  This
script fits a small diagnostic model on PixArt only, freezes its threshold, and
then evaluates it unchanged on SDXL.  Results never modify runtime scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_trust.geometry_ai.equivariance import (
    PARAMETER_KEYS,
    compare_horizontal_flip_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260812
MIB = 1024 * 1024
COORDINATE_SOURCE_PATHS = {
    "utils_py_sha256": ".venv/Lib/site-packages/perspective2d/utils/utils.py",
    "panocam_py_sha256": ".venv/Lib/site-packages/perspective2d/utils/panocam.py",
    "gravity_head_py_sha256": (
        ".venv/Lib/site-packages/perspective2d/modeling/"
        "persformer_heads/gravity_head.py"
    ),
    "latitude_head_py_sha256": (
        ".venv/Lib/site-packages/perspective2d/modeling/"
        "persformer_heads/latitude_head.py"
    ),
}


@dataclass(frozen=True)
class Sample:
    archive: str
    generator: str
    identifier: int
    label: int
    path: str
    scene: str
    sha256: str
    split: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _generator_and_scene(archive: str) -> tuple[str, str]:
    parts = archive.lower().split("_")
    if len(parts) != 3 or parts[0] != "recent":
        raise ValueError(f"unexpected Projective Geometry archive: {archive}")
    return parts[1], parts[2]


def _split_for(generator: str, identifier: int) -> str | None:
    if generator == "pixart" and 351 <= identifier <= 425:
        return "calibration"
    if generator == "sdxl" and 426 <= identifier <= 500:
        return "test"
    return None


def discover(root: Path) -> list[Sample]:
    """Discover the exact registered PixArt and SDXL cells."""

    rows: list[Sample] = []
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for path in sorted(root.glob("**/test/*/*.jpg")):
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        label_name = path.parent.name
        if label_name not in {"real", "gen"} or not path.stem.isdecimal():
            continue
        archive = path.parents[2].name
        generator, scene = _generator_and_scene(archive)
        identifier = int(path.stem)
        split = _split_for(generator, identifier)
        if split is None:
            continue
        digest = _sha256(path)
        if digest in seen_hashes:
            raise ValueError(f"duplicate source across registered cohorts: {path}")
        seen_hashes.add(digest)
        rows.append(
            Sample(
                archive=archive,
                generator=generator,
                identifier=identifier,
                label=int(label_name == "gen"),
                path=str(resolved),
                scene=scene,
                sha256=digest,
                split=split,
            )
        )
    rows.sort(key=lambda row: (row.split, row.scene, row.label, row.identifier))
    expected = {
        (split, scene, label): 75
        for split in ("calibration", "test")
        for scene in ("indoor", "outdoor")
        for label in (0, 1)
    }
    counts = Counter((row.split, row.scene, row.label) for row in rows)
    if counts != expected:
        raise ValueError(f"registered geometry cells do not match: {dict(counts)}")
    return rows


def _validate_inputs(
    protocol_path: Path,
    data_root: Path,
    raw_root: Path,
) -> tuple[dict[str, Any], list[Sample], Path, tuple[str, ...], int, int]:
    protocol = _load_object(protocol_path)
    if protocol.get("schema_version") != "demirror-geometry-flip-equivariance-protocol-v1":
        raise ValueError("unexpected equivariance protocol schema")
    if protocol.get("status") != "pre_registered_before_equivariance_measurement":
        raise ValueError("equivariance protocol was not pre-registered")
    upstream = protocol["upstream"]
    weights_path = (PROJECT_ROOT / upstream["checkpoint"]).resolve()
    if _sha256(weights_path) != upstream["checkpoint_sha256"]:
        raise ValueError("PerspectiveFields checkpoint hash mismatch")
    coordinate_evidence = upstream["coordinate_evidence"]
    if set(coordinate_evidence) != set(COORDINATE_SOURCE_PATHS):
        raise ValueError("PerspectiveFields coordinate-evidence map mismatch")
    for hash_key, relative_path in COORDINATE_SOURCE_PATHS.items():
        source_path = (PROJECT_ROOT / relative_path).resolve()
        if _sha256(source_path) != coordinate_evidence[hash_key]:
            raise ValueError(f"PerspectiveFields coordinate source hash mismatch: {hash_key}")
    for split in ("calibration", "test"):
        for archive_name, expected_hash in protocol["cohorts"][split][
            "archive_sha256"
        ].items():
            archive_path = (raw_root / archive_name).resolve()
            if _sha256(archive_path) != expected_hash:
                raise ValueError(f"archive hash mismatch: {archive_name}")
    samples = discover(data_root)
    features = tuple(str(value) for value in protocol["features"])
    if len(features) != len(set(features)) or not features:
        raise ValueError("protocol features must be unique and non-empty")
    resource = protocol["resource_policy"]
    return (
        protocol,
        samples,
        weights_path,
        features,
        int(resource["shard_size"]),
        int(resource["maximum_worker_working_set_mib"]) * MIB,
    )


def _load_perspective_fields(weights_path: Path, *, threads: int):
    import torch
    from perspective2d.perspectivefields import PerspectiveFields

    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)
    original_loader = torch.hub.load_state_dict_from_url

    def local_loader(*_args: Any, **_kwargs: Any):
        return torch.load(weights_path, map_location="cpu")

    torch.hub.load_state_dict_from_url = local_loader
    try:
        return PerspectiveFields("Paramnet-360Cities-edina-uncentered").to("cpu").eval()
    finally:
        torch.hub.load_state_dict_from_url = original_loader


def _prediction_values(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    latitude = (
        prediction["pred_latitude_original"].detach().float().cpu().numpy()
    )
    gravity = prediction["pred_gravity_original"].detach().float().cpu().numpy()
    parameters = {
        key: float(prediction[key].detach().float().cpu().reshape(-1)[0].item())
        for key in PARAMETER_KEYS
    }
    return latitude, gravity, parameters


def _extract_shard_worker(
    samples: list[Sample],
    weights_path: str,
    output_path: str,
    threads: int,
) -> None:
    import torch

    model = _load_perspective_fields(Path(weights_path), threads=threads)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for sample in samples:
            image = cv2.imread(sample.path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unable to decode image: {sample.path}")
            original = _prediction_values(model.inference(img_bgr=image))
            flipped = _prediction_values(
                model.inference(img_bgr=np.ascontiguousarray(image[:, ::-1]))
            )
            metrics = compare_horizontal_flip_predictions(
                original_latitude=original[0],
                original_gravity=original[1],
                original_parameters=original[2],
                flipped_latitude=flipped[0],
                flipped_gravity=flipped[1],
                flipped_parameters=flipped[2],
            )
            rows.append({"sample": asdict(sample), "metrics": metrics})
    _atomic_write_json(
        Path(output_path),
        {
            "schema_version": "demirror-geometry-flip-equivariance-shard-v1",
            "rows": rows,
        },
    )


def _valid_shard(path: Path, samples: list[Sample], features: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_object(path)
        if payload.get("schema_version") != "demirror-geometry-flip-equivariance-shard-v1":
            return False
        rows = payload["rows"]
        if [row["sample"] for row in rows] != [asdict(sample) for sample in samples]:
            return False
        for row in rows:
            metrics = row["metrics"]
            if tuple(metrics) != features:
                return False
            if not all(math.isfinite(float(metrics[key])) for key in features):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _process_working_set_bytes(pid: int) -> int | None:
    if os.name == "nt":
        try:
            import ctypes
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

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
            if not handle:
                return None
            try:
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                function = kernel32.K32GetProcessMemoryInfo
                function.argtypes = [
                    wintypes.HANDLE,
                    ctypes.POINTER(ProcessMemoryCounters),
                    wintypes.DWORD,
                ]
                function.restype = wintypes.BOOL
                ok = function(
                    handle,
                    ctypes.byref(counters),
                    ctypes.sizeof(counters),
                )
                return int(counters.WorkingSetSize) if ok else None
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        return resident_pages * page_size
    except (AttributeError, IndexError, OSError, ValueError):
        return None


def _run_isolated_shard(
    samples: list[Sample],
    *,
    weights_path: Path,
    output_path: Path,
    threads: int,
    maximum_bytes: int,
) -> int:
    context = mp.get_context("spawn")
    process = context.Process(
        target=_extract_shard_worker,
        args=(samples, str(weights_path), str(output_path), threads),
    )
    process.start()
    peak = 0
    exceeded = False
    while process.is_alive():
        sample = _process_working_set_bytes(process.pid)
        if sample is not None:
            peak = max(peak, sample)
            if sample > maximum_bytes:
                exceeded = True
                process.terminate()
                break
        process.join(timeout=0.25)
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        raise RuntimeError("PerspectiveFields shard did not terminate")
    if exceeded:
        output_path.unlink(missing_ok=True)
        raise MemoryError(
            f"PerspectiveFields worker exceeded {maximum_bytes / MIB:.0f} MiB"
        )
    if process.exitcode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"PerspectiveFields shard failed ({process.exitcode=})")
    return peak


def extract(
    samples: list[Sample],
    *,
    weights_path: Path,
    output_dir: Path,
    features: tuple[str, ...],
    shard_size: int,
    threads: int,
    maximum_bytes: int,
    max_new_shards: int | None,
) -> tuple[list[dict[str, Any]] | None, dict[str, int]]:
    parts = output_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    new_shards = 0
    peak = 0
    part_paths: list[Path] = []
    for shard_number, start in enumerate(range(0, len(samples), shard_size), start=1):
        stop = min(start + shard_size, len(samples))
        shard_samples = samples[start:stop]
        path = parts / f"part_{shard_number:03d}.json"
        part_paths.append(path)
        if _valid_shard(path, shard_samples, features):
            continue
        if max_new_shards is not None and new_shards >= max_new_shards:
            return None, {
                "completed_shards": sum(
                    _valid_shard(candidate, samples[index:index + shard_size], features)
                    for candidate, index in zip(part_paths, range(0, stop, shard_size))
                ),
                "total_shards": math.ceil(len(samples) / shard_size),
                "maximum_sampled_worker_working_set_bytes": peak,
            }
        peak = max(
            peak,
            _run_isolated_shard(
                shard_samples,
                weights_path=weights_path,
                output_path=path,
                threads=threads,
                maximum_bytes=maximum_bytes,
            ),
        )
        new_shards += 1
        print(
            f"equivariance {stop}/{len(samples)} "
            f"(shard {shard_number}/{math.ceil(len(samples) / shard_size)}, "
            f"peak {peak / MIB:.1f} MiB)",
            flush=True,
        )
    rows: list[dict[str, Any]] = []
    for shard_number, start in enumerate(range(0, len(samples), shard_size), start=1):
        stop = min(start + shard_size, len(samples))
        path = parts / f"part_{shard_number:03d}.json"
        if not _valid_shard(path, samples[start:stop], features):
            raise RuntimeError(f"invalid completed shard: {path}")
        rows.extend(_load_object(path)["rows"])
    return rows, {
        "completed_shards": len(part_paths),
        "total_shards": len(part_paths),
        "maximum_sampled_worker_working_set_bytes": peak,
    }


def _rates(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    decisions = np.asarray(decisions, dtype=bool)
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(decisions & positives))
    fp = int(np.sum(decisions & negatives))
    return {
        "true_positive_count": tp,
        "positive_count": int(np.sum(positives)),
        "true_positive_rate": tp / int(np.sum(positives)),
        "false_positive_count": fp,
        "negative_count": int(np.sum(negatives)),
        "false_positive_rate": fp / int(np.sum(negatives)),
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    maximum_false_positive_rate: float,
) -> tuple[float, dict[str, float | int]]:
    """Freeze the best calibration recall under the registered FPR constraint."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("threshold selection requires finite two-class scores")
    candidates = np.unique(probabilities)
    candidates = np.append(candidates, np.nextafter(np.max(candidates), np.inf))
    options: list[tuple[float, float, dict[str, float | int]]] = []
    for threshold in candidates:
        rates = _rates(labels, probabilities >= threshold)
        if float(rates["false_positive_rate"]) <= maximum_false_positive_rate:
            options.append((float(rates["true_positive_rate"]), float(threshold), rates))
    if not options:
        raise ValueError("no calibration threshold satisfies the FPR constraint")
    _recall, threshold, rates = max(options, key=lambda item: (item[0], item[1]))
    return threshold, rates


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    features: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    samples = [row["sample"] for row in rows]
    matrix = np.asarray(
        [[float(row["metrics"][feature]) for feature in features] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    calibration = np.asarray([sample["split"] == "calibration" for sample in samples])
    test = np.asarray([sample["split"] == "test" for sample in samples])
    scaler = StandardScaler().fit(matrix[calibration])
    estimator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
        solver="lbfgs",
    ).fit(scaler.transform(matrix[calibration]), labels[calibration])
    probabilities = estimator.predict_proba(scaler.transform(matrix))[:, 1]
    threshold, calibration_rates = select_threshold(
        labels[calibration],
        probabilities[calibration],
        maximum_false_positive_rate=0.25,
    )
    decisions = probabilities >= threshold

    def split_report(mask: np.ndarray) -> dict[str, Any]:
        value: dict[str, Any] = {
            "sample_count": int(np.sum(mask)),
            "roc_auc": float(roc_auc_score(labels[mask], probabilities[mask])),
            "frozen_threshold_rates": _rates(labels[mask], decisions[mask]),
        }
        by_scene: dict[str, Any] = {}
        for scene in ("indoor", "outdoor"):
            scene_mask = mask & np.asarray([sample["scene"] == scene for sample in samples])
            by_scene[scene] = {
                "sample_count": int(np.sum(scene_mask)),
                "roc_auc": float(roc_auc_score(labels[scene_mask], probabilities[scene_mask])),
                "frozen_threshold_rates": _rates(
                    labels[scene_mask], decisions[scene_mask]
                ),
            }
        value["by_scene"] = by_scene
        return value

    distributions: dict[str, Any] = {}
    single_feature_auc: dict[str, Any] = {}
    for split_name, split_mask in (("calibration", calibration), ("test", test)):
        distributions[split_name] = {}
        single_feature_auc[split_name] = {}
        for feature_index, feature in enumerate(features):
            distributions[split_name][feature] = {
                label_name: _distribution(
                    matrix[split_mask & (labels == label_value), feature_index]
                )
                for label_name, label_value in (("real", 0), ("generated", 1))
            }
            single_feature_auc[split_name][feature] = float(
                roc_auc_score(labels[split_mask], matrix[split_mask, feature_index])
            )

    calibration_report = split_report(calibration)
    calibration_report["selected_threshold"] = threshold
    calibration_report["selection_rates"] = calibration_rates
    test_report = split_report(test)
    requirements = protocol["screening_gate"]["requirements"]
    test_rates = test_report["frozen_threshold_rates"]
    checks = {
        "sdxl_overall_roc_auc": test_report["roc_auc"]
        >= requirements["sdxl_overall_roc_auc_minimum"],
        "sdxl_true_positive_rate_at_frozen_threshold": test_rates[
            "true_positive_rate"
        ]
        >= requirements["sdxl_true_positive_rate_at_frozen_threshold_minimum"],
        "sdxl_false_positive_rate_at_frozen_threshold": test_rates[
            "false_positive_rate"
        ]
        <= requirements["sdxl_false_positive_rate_at_frozen_threshold_maximum"],
        "sdxl_indoor_roc_auc": test_report["by_scene"]["indoor"]["roc_auc"]
        >= requirements["sdxl_indoor_roc_auc_minimum"],
        "sdxl_outdoor_roc_auc": test_report["by_scene"]["outdoor"]["roc_auc"]
        >= requirements["sdxl_outdoor_roc_auc_minimum"],
    }
    passed = all(checks.values())
    scored_rows = [
        {
            **row,
            "diagnostic_probability": float(probability),
            "frozen_threshold_decision": bool(decision),
        }
        for row, probability, decision in zip(rows, probabilities, decisions)
    ]
    model = {
        "schema_version": "demirror-geometry-flip-equivariance-diagnostic-model-v1",
        "deployment_eligible": False,
        "features": list(features),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "threshold": threshold,
        "fit_split": "PixArt calibration",
        "test_split": "SDXL held out from fitting and threshold selection",
    }
    report = {
        "schema_version": "demirror-geometry-flip-equivariance-audit-v1",
        "sample_count": len(rows),
        "calibration": calibration_report,
        "test": test_report,
        "feature_distributions": distributions,
        "single_feature_auc_fixed_higher_discrepancy_direction": single_feature_auc,
        "screening_gate": {
            "requirements": requirements,
            "checks": checks,
            "passed": passed,
        },
        "decision": (
            "eligible_for_new_independent_replication_not_integration"
            if passed
            else "rejected_for_current_origin_path"
        ),
        "limitations": [
            "The diagnostic model was fitted on PixArt and evaluated once on SDXL.",
            "PerspectiveFields consistency can reflect model domain shift rather than image-source physics.",
            "No runtime or origin-score change is authorized by this audit.",
        ],
    }
    return report, model, scored_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            PROJECT_ROOT
            / "research/records/2026-08-12/geometry/geometry_flip_equivariance_protocol_v1.json"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data/p3_aigc_v2/extracted",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data/p3_aigc_v2/raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/geometry_flip_equivariance_v1",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-new-shards", type=int)
    return parser.parse_args()


def _preserve_previous_peak(
    state_path: Path,
    *,
    protocol_sha256: str,
    resource: dict[str, int | float],
) -> None:
    """Keep a peak observed by an earlier resumable run of the same protocol."""

    if not state_path.is_file():
        return
    try:
        previous = _load_object(state_path)
        if (
            previous.get("schema_version")
            != "demirror-geometry-flip-equivariance-incomplete-v1"
            or previous.get("protocol_sha256") != protocol_sha256
        ):
            return
        previous_peak = int(
            previous["resource"]["maximum_sampled_worker_working_set_bytes"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    resource["maximum_sampled_worker_working_set_bytes"] = max(
        int(resource["maximum_sampled_worker_working_set_bytes"]),
        previous_peak,
    )


def main() -> int:
    args = _parse_args()
    if args.threads != 1:
        raise ValueError("the registered protocol requires exactly one torch thread")
    protocol, samples, weights, features, shard_size, maximum_bytes = _validate_inputs(
        args.protocol.resolve(),
        args.data_root.resolve(),
        args.raw_root.resolve(),
    )
    started = time.perf_counter()
    rows, resource = extract(
        samples,
        weights_path=weights,
        output_dir=args.output_dir.resolve(),
        features=features,
        shard_size=shard_size,
        threads=args.threads,
        maximum_bytes=maximum_bytes,
        max_new_shards=args.max_new_shards,
    )
    resource["registered_maximum_worker_working_set_bytes"] = maximum_bytes
    resource["elapsed_seconds_this_run"] = time.perf_counter() - started
    if rows is None:
        protocol_sha256 = _sha256(args.protocol)
        state_path = args.output_dir / "incomplete_state.json"
        _preserve_previous_peak(
            state_path,
            protocol_sha256=protocol_sha256,
            resource=resource,
        )
        _atomic_write_json(
            state_path,
            {
                "schema_version": "demirror-geometry-flip-equivariance-incomplete-v1",
                "protocol_sha256": protocol_sha256,
                "resource": resource,
                "decision": "incomplete_no_acceptance_decision",
            },
        )
        print(json.dumps(resource, indent=2), flush=True)
        return 2
    report, model, scored_rows = summarize(rows, protocol=protocol, features=features)
    report["protocol"] = str(args.protocol.resolve())
    report["protocol_sha256"] = _sha256(args.protocol)
    report["resource"] = resource
    _atomic_write_json(args.output_dir / "report.json", report)
    _atomic_write_json(args.output_dir / "diagnostic_model.json", model)
    _atomic_write_text(
        args.output_dir / "scores.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in scored_rows
        ),
    )
    (args.output_dir / "incomplete_state.json").unlink(missing_ok=True)
    print(json.dumps(report["screening_gate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
