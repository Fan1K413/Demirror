"""Audit frozen propagation views with the registered DDA detector.

The large DDA model is loaded once in a label-blind child process.  That child
scores one image at a time and atomically checkpoints every row.  The parent
samples the launcher and all descendants, terminating the complete process
tree if their summed working set crosses the registered 2 GiB stop.  Labels
are joined only after all raw scores have been written successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


PROFILES = (
    "original_decode",
    "jpeg_reencode_quality=85",
    "webp_reencode_quality=85",
    "resize_longest=1024_restore_png",
    "screenshot_raster_png_longest=1600",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing registered file: {path}")
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def _load_script(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load registered implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_views(
    repo_root: Path, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_registration = protocol["source_manifest"]
    source_path = repo_root / str(source_registration["path"])
    _require_hash(source_path, str(source_registration["sha256"]))
    source_records = _read_json(source_path).get("records")
    if not isinstance(source_records, list) or len(source_records) != int(
        source_registration["record_count"]
    ):
        raise ValueError("Registered source record count mismatch")

    variants_registration = protocol["variant_manifest"]
    variants_path = repo_root / str(variants_registration["path"])
    _require_hash(variants_path, str(variants_registration["sha256"]))
    variants = _read_json(variants_path).get("records")
    if not isinstance(variants, list) or len(variants) != int(
        variants_registration["record_count"]
    ):
        raise ValueError("Registered variant record count mismatch")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in variants:
        if not isinstance(record, dict):
            raise ValueError("Invalid variant record")
        key = (
            str(record.get("source_asset_sha256", "")).lower(),
            str(record.get("profile", "")),
        )
        if key in by_key:
            raise ValueError("Duplicate source/profile variant record")
        by_key[key] = record

    views: list[dict[str, Any]] = []
    for index, source in enumerate(source_records):
        if not isinstance(source, dict):
            raise ValueError("Invalid source record")
        asset = str(source.get("asset_sha256", "")).lower()
        original = (source_path.parent / str(source.get("relative_path", ""))).resolve()
        try:
            original.relative_to(source_path.parent.resolve())
        except ValueError as error:
            raise ValueError("Source path escapes manifest root") from error
        _require_hash(original, asset)
        views.append(
            {
                "source_manifest_index": index,
                "source_asset_sha256": asset,
                "profile": "original_decode",
                "path": original,
                "artifact_sha256": asset,
            }
        )
        for profile in PROFILES[1:]:
            variant = by_key.get((asset, profile))
            if variant is None:
                raise ValueError(f"Missing frozen variant for {asset} / {profile}")
            artifact = (variants_path.parent / str(variant.get("relative_path", ""))).resolve()
            try:
                artifact.relative_to(variants_path.parent.resolve())
            except ValueError as error:
                raise ValueError("Variant path escapes variant root") from error
            artifact_hash = str(variant.get("artifact_sha256", "")).lower()
            _require_hash(artifact, artifact_hash)
            views.append(
                {
                    "source_manifest_index": index,
                    "source_asset_sha256": asset,
                    "profile": profile,
                    "path": artifact,
                    "artifact_sha256": artifact_hash,
                }
            )
    if len(views) != len(source_records) * len(PROFILES):
        raise ValueError("Frozen propagation view count mismatch")
    return views, [dict(record) for record in source_records]


def _partial(protocol_hash: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "demirror-dda-propagation-score-partial-v1",
        "purpose": "Label-free resumable raw-score checkpoint; not an audit result.",
        "protocol_sha256": protocol_hash,
        "rows": rows,
    }


def _completed_rows(
    path: Path,
    protocol_hash: str,
    views: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    partial = _read_json(path)
    if partial.get("protocol_sha256") != protocol_hash:
        raise ValueError("Partial score checkpoint belongs to a different protocol")
    rows = partial.get("rows")
    if not isinstance(rows, list) or len(rows) > len(views):
        raise ValueError("Partial score checkpoint has an invalid row list")
    common = {
        "source_manifest_index",
        "source_asset_sha256",
        "profile",
        "artifact_sha256",
        "status",
    }
    completed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Partial score row is not a label-free registered row")
        status = row.get("status")
        expected_keys = common | ({"score"} if status == "available" else {"reason"})
        if status not in {"available", "unavailable"} or set(row) != expected_keys:
            raise ValueError("Partial score row is not a label-free registered row")
        view = views[index]
        for key in (
            "source_manifest_index",
            "source_asset_sha256",
            "profile",
            "artifact_sha256",
        ):
            if row[key] != view[key]:
                raise ValueError("Partial score rows are not an exact registered prefix")
        if status == "available":
            score = float(row["score"])
            if not 0.0 <= score <= 1.0:
                raise ValueError("Partial score is outside [0, 1]")
        completed.append(dict(row))
    return completed


def _model_input(path: Path, transform: Any) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    if min(image.size) < 336:
        raise ValueError("dda_input_too_small")
    return transform(image).unsqueeze(0)


def _score_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from torchvision import transforms

    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    implementation = protocol["implementation"]
    runtime_adapter = args.repo_root / str(implementation["runtime_adapter_path"])
    _require_hash(runtime_adapter, str(implementation["runtime_adapter_sha256"]))
    checkpoint_path = args.repo_root / str(protocol["model"]["checkpoint_path"])
    _require_hash(checkpoint_path, str(protocol["model"]["checkpoint_sha256"]))
    views, _ = _resolve_views(args.repo_root, protocol)

    partial_path = args.output_dir / "scores.partial.json"
    if partial_path.exists() and not args.resume:
        raise FileExistsError("Existing score checkpoint requires --resume")
    rows = _completed_rows(partial_path, protocol_hash, views) if args.resume else []

    threads = int(protocol["execution"]["cpu_threads"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from image_trust.ai_likelihood import dda

    model = dda._load_dda_model_low_memory(torch, nn, checkpoint_path)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    equivalence = protocol["implementation"]["equivalence_probe"]
    equivalent_view = next(
        view
        for view in views
        if view["source_asset_sha256"] == str(equivalence["source_asset_sha256"])
        and view["profile"] == str(equivalence["profile"])
    )
    with torch.inference_mode():
        batch_score = float(
            torch.sigmoid(model(_model_input(equivalent_view["path"], transform))).item()
        )
    registered_score = float(equivalence["registered_product_score"])
    difference = abs(batch_score - registered_score)
    maximum = float(equivalence["absolute_difference_maximum"])
    if difference > maximum:
        raise ValueError(f"Batch/product DDA score mismatch: {difference} exceeds {maximum}")
    equivalence_result = {
        "source_asset_sha256": equivalent_view["source_asset_sha256"],
        "profile": equivalent_view["profile"],
        "batch_score": batch_score,
        "registered_product_score": registered_score,
        "absolute_difference": difference,
        "maximum": maximum,
    }

    with torch.inference_mode():
        for view in views[len(rows) :]:
            try:
                score = float(torch.sigmoid(model(_model_input(view["path"], transform))).item())
            except ValueError as error:
                if str(error) != "dda_input_too_small":
                    raise
                row = {
                    "source_manifest_index": view["source_manifest_index"],
                    "source_asset_sha256": view["source_asset_sha256"],
                    "profile": view["profile"],
                    "artifact_sha256": view["artifact_sha256"],
                    "status": "unavailable",
                    "reason": "dda_input_too_small",
                }
            else:
                row = {
                    "source_manifest_index": view["source_manifest_index"],
                    "source_asset_sha256": view["source_asset_sha256"],
                    "profile": view["profile"],
                    "artifact_sha256": view["artifact_sha256"],
                    "status": "available",
                    "score": score,
                }
            rows.append(row)
            _atomic_write_json(partial_path, _partial(protocol_hash, rows))
            if len(rows) % 10 == 0 or len(rows) == len(views):
                print(f"dda_scored={len(rows)}/{len(views)}", flush=True)

    if len(rows) != len(views):
        raise ValueError("Incomplete propagation score rows")
    _atomic_write_json(
        args.worker_output,
        {
            "schema_version": "demirror-dda-propagation-worker-result-v1",
            "protocol_sha256": protocol_hash,
            "row_count": len(rows),
            "runtime_equivalence": equivalence_result,
        },
    )


def _rate(rows: list[Mapping[str, Any]], threshold: float) -> dict[str, int | float]:
    available = [row for row in rows if row.get("status") == "available"]
    hits = sum(float(row["score"]) >= threshold for row in available)
    return {
        "hits": hits,
        "available": len(available),
        "unavailable": len(rows) - len(available),
        "rate_over_available": hits / len(available) if available else 0.0,
    }


def _summarize(
    repo_root: Path,
    protocol: Mapping[str, Any],
    scores: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_asset = {str(record["asset_sha256"]).lower(): record for record in source_records}
    if len(source_by_asset) != len(source_records):
        raise ValueError("Duplicate source assets are not allowed")
    audit = _read_json(repo_root / str(protocol["model"]["audit_path"]))
    threshold = float(audit["high_confidence_threshold"])
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_profile[str(row["profile"])].append(row)
    profiles: dict[str, Any] = {}
    for profile in PROFILES:
        rows = by_profile[profile]
        generated = [
            row
            for row in rows
            if source_by_asset[str(row["source_asset_sha256"])]["label"] == "fake"
        ]
        real = [
            row
            for row in rows
            if source_by_asset[str(row["source_asset_sha256"])]["label"] == "real"
        ]
        profiles[profile] = {
            "generated_high_threshold": _rate(generated, threshold),
            "real_high_threshold": _rate(real, threshold),
            "product_high_eligible": profile != "webp_reencode_quality=85",
        }
    return {"high_confidence_threshold": threshold, "profile_metrics": profiles}


def _failure_report(
    protocol_hash: str,
    observation: Mapping[str, Any],
    partial_path: Path,
) -> dict[str, Any]:
    completed = 0
    checkpoint_hash: str | None = None
    if partial_path.is_file():
        partial = _read_json(partial_path)
        rows = partial.get("rows")
        completed = len(rows) if isinstance(rows, list) else 0
        checkpoint_hash = _sha256(partial_path)
    return {
        "schema_version": "demirror-dda-propagation-audit-failure-v1",
        "protocol_sha256": protocol_hash,
        "failure": {
            "reason": observation.get("stop_reason") or "worker_failed",
            "worker_returncode": observation.get("returncode"),
            "worker_warning_codes": observation.get("warning_codes", []),
            "worker_stderr_present_unclassified": observation.get(
                "stderr_present_unclassified", False
            ),
        },
        "resource": {
            "maximum_sampled_working_set_bytes": observation.get(
                "maximum_sampled_working_set_bytes", 0
            ),
            "maximum_sampled_process_count": observation.get(
                "maximum_sampled_process_count", 0
            ),
            "sampling_scope": "launcher_and_all_descendants",
        },
        "completed_label_free_score_rows": completed,
        "partial_checkpoint_sha256": checkpoint_hash,
        "result_interpretation_allowed": False,
        "deployment_eligible": False,
        "runtime_policy_changed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    implementation = protocol["implementation"]
    script_path = args.repo_root / str(implementation["audit_script_path"])
    monitor_path = args.repo_root / str(implementation["resource_monitor_path"])
    runtime_path = args.repo_root / str(implementation["runtime_adapter_path"])
    _require_hash(script_path, str(implementation["audit_script_sha256"]))
    _require_hash(monitor_path, str(implementation["resource_monitor_sha256"]))
    _require_hash(runtime_path, str(implementation["runtime_adapter_sha256"]))
    _require_hash(
        args.repo_root / str(protocol["model"]["checkpoint_path"]),
        str(protocol["model"]["checkpoint_sha256"]),
    )
    _require_hash(
        args.repo_root / str(protocol["model"]["audit_path"]),
        str(protocol["model"]["audit_sha256"]),
    )
    preflight = protocol["resource_preflight"]
    preflight_path = args.repo_root / str(preflight["path"])
    _require_hash(preflight_path, str(preflight["sha256"]))
    if not _read_json(preflight_path).get("full_propagation_audit_resource_preflight_passed"):
        raise ValueError("Registered DDA resource preflight did not pass")

    views, source_records = _resolve_views(args.repo_root, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_output = args.output_dir / "worker-result.json"
    command = [
        sys.executable,
        str(script_path),
        "--worker",
        "--repo-root",
        str(args.repo_root),
        "--protocol",
        str(args.protocol),
        "--output-dir",
        str(args.output_dir),
        "--worker-output",
        str(worker_output),
    ]
    if args.resume:
        command.append("--resume")
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(protocol["execution"]["cpu_threads"]),
            "MKL_NUM_THREADS": str(protocol["execution"]["cpu_threads"]),
            "DDA_CPU_THREADS": str(protocol["execution"]["cpu_threads"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    monitor = _load_script(monitor_path, "demirror_dda_resource_monitor")
    execution = protocol["execution"]
    observation = monitor._run_worker(
        command,
        environment,
        maximum_working_set_bytes=int(execution["maximum_working_set_bytes"]),
        timeout_seconds=float(execution["timeout_seconds"]),
        sample_interval_seconds=float(execution["sample_interval_seconds"]),
    )
    if observation["returncode"] != 0 or observation["stop_reason"] is not None:
        failure = _failure_report(
            protocol_hash,
            observation,
            args.output_dir / "scores.partial.json",
        )
        _atomic_write_json(args.failure_output, failure)
        return None
    if not worker_output.is_file():
        raise ValueError("DDA propagation worker did not write its result")
    worker_result = _read_json(worker_output)
    if worker_result.get("protocol_sha256") != protocol_hash:
        raise ValueError("DDA propagation worker result protocol mismatch")
    scores = _completed_rows(
        args.output_dir / "scores.partial.json",
        protocol_hash,
        views,
    )
    if len(scores) != len(views) or int(worker_result.get("row_count", -1)) != len(views):
        raise ValueError("DDA propagation worker result is incomplete")
    summary = _summarize(args.repo_root, protocol, scores, source_records)
    report = {
        "schema_version": "demirror-dda-propagation-audit-v1",
        "purpose": "Frozen-threshold propagation audit only; no threshold, product score, dependency or web-policy change is authorized.",
        "protocol_sha256": protocol_hash,
        "score_checkpoint_sha256": _sha256(args.output_dir / "scores.partial.json"),
        "view_count": len(views),
        "runtime_equivalence": worker_result["runtime_equivalence"],
        "resource": {
            "maximum_working_set_bytes": int(execution["maximum_working_set_bytes"]),
            "maximum_sampled_working_set_bytes": int(
                observation["maximum_sampled_working_set_bytes"]
            ),
            "maximum_sampled_process_count": int(observation["maximum_sampled_process_count"]),
            "sampling_scope": "launcher_and_all_descendants",
            "elapsed_seconds": float(observation["elapsed_seconds"]),
        },
        "worker_warning_codes": observation["warning_codes"],
        "summary": summary,
        "deployment_eligible": False,
        "runtime_policy_changed": False,
    }
    _atomic_write_json(args.audit_output, report)
    return report


def _path(value: str) -> Path:
    return Path(value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-root", type=_path, default=Path.cwd())
    parser.add_argument(
        "--protocol",
        type=_path,
        default=Path(
            "research/records/2026-08-20/pixel/dda_propagation_audit_protocol_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=Path("outputs/research/dda_propagation_audit_v1"),
    )
    parser.add_argument(
        "--audit-output",
        type=_path,
        default=Path("research/records/2026-08-20/pixel/dda_propagation_audit_v1.json"),
    )
    parser.add_argument(
        "--failure-output",
        type=_path,
        default=Path(
            "research/records/2026-08-20/pixel/dda_propagation_audit_failure_v1.json"
        ),
    )
    parser.add_argument("--worker-output", type=_path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.worker_output is None:
            parser.error("worker mode requires --worker-output")
        _score_worker(args)
        return 0
    report = run(args)
    if report is None:
        return 1
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
