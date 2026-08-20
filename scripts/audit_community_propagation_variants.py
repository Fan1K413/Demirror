"""Audit frozen propagation views with the registered Community detector.

The model is loaded once in a research-only CPU process.  Scores are
checkpointed without labels; labels are read only after scoring to summarize
the already-frozen high and limited thresholds.  This cannot change any
product threshold, score component, or web behaviour.
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


def _load_script(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("demirror_community_batch_support", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load registered implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_views(repo_root: Path, protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_registration = protocol["source_manifest"]
    source_path = repo_root / str(source_registration["path"])
    _require_hash(source_path, str(source_registration["sha256"]))
    source_records = _read_json(source_path).get("records")
    if not isinstance(source_records, list) or len(source_records) != int(source_registration["record_count"]):
        raise ValueError("Registered source record count mismatch")

    variants_registration = protocol["variant_manifest"]
    variants_path = repo_root / str(variants_registration["path"])
    _require_hash(variants_path, str(variants_registration["sha256"]))
    variants = _read_json(variants_path).get("records")
    if not isinstance(variants, list) or len(variants) != int(variants_registration["record_count"]):
        raise ValueError("Registered variant record count mismatch")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in variants:
        if not isinstance(record, dict):
            raise ValueError("Invalid variant record")
        key = (str(record.get("source_asset_sha256", "")).lower(), str(record.get("profile", "")))
        if key in by_key:
            raise ValueError("Duplicate source/profile variant record")
        by_key[key] = record

    views: list[dict[str, Any]] = []
    for index, source in enumerate(source_records):
        if not isinstance(source, dict):
            raise ValueError("Invalid source record")
        asset = str(source.get("asset_sha256", "")).lower()
        source_image = (source_path.parent / str(source.get("relative_path", ""))).resolve()
        try:
            source_image.relative_to(source_path.parent.resolve())
        except ValueError as exc:
            raise ValueError("Source path escapes manifest root") from exc
        _require_hash(source_image, asset)
        views.append(
            {
                "source_manifest_index": index,
                "source_asset_sha256": asset,
                "profile": "original_decode",
                "path": source_image,
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
            except ValueError as exc:
                raise ValueError("Variant path escapes variant root") from exc
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
    return views, source_records


def _partial(protocol_hash: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "demirror-community-propagation-score-partial-v1",
        "purpose": "Label-free resumable raw-score checkpoint; not an audit result.",
        "protocol_sha256": protocol_hash,
        "rows": rows,
    }


def _completed_rows(path: Path, protocol_hash: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    partial = _read_json(path)
    if partial.get("protocol_sha256") != protocol_hash:
        raise ValueError("Partial score checkpoint belongs to a different protocol")
    rows = partial.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Partial score checkpoint has no row list")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _score_views(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    views: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    implementation = args.repo_root / str(protocol["implementation"]["batch_support_path"])
    _require_hash(implementation, str(protocol["implementation"]["batch_support_sha256"]))
    runtime_adapter = args.repo_root / str(protocol["implementation"]["runtime_adapter_path"])
    _require_hash(runtime_adapter, str(protocol["implementation"]["runtime_adapter_sha256"]))
    support = _load_script(implementation)
    model_root = args.repo_root / str(protocol["model"]["root"])
    config_path = model_root / "config.json"
    weights_path = model_root / "model.safetensors"
    _require_hash(config_path, str(protocol["model"]["config_sha256"]))
    _require_hash(weights_path, str(protocol["model"]["weights_sha256"]))
    config = _read_json(config_path)
    if int(config.get("input_size", 0)) != 224:
        raise ValueError("Registered Community input size mismatch")

    partial_path = args.output_dir / "scores.partial.json"
    rows = _completed_rows(partial_path, protocol_hash) if args.resume else []
    completed = {(str(row.get("source_asset_sha256")), str(row.get("profile"))) for row in rows}
    if rows and not args.resume:
        raise FileExistsError("Existing score checkpoint requires --resume")

    torch.set_num_threads(int(protocol["execution"]["cpu_threads"]))
    torch.set_num_interop_threads(1)
    model = support._load_model(args.repo_root / "data/vendor/community-forensics", weights_path, config)
    transform = support._test_transform(224)
    equivalence = protocol["implementation"]["equivalence_probe"]
    equivalent_view = next(
        view
        for view in views
        if view["source_asset_sha256"] == str(equivalence["source_asset_sha256"])
        and view["profile"] == str(equivalence["profile"])
    )
    batch_score = float(
        torch.sigmoid(
            model(support._model_input(equivalent_view["path"], transform, None, None, 1.0))
        ).item()
    )
    from image_trust.ai_likelihood.community_forensics import score_community_forensics_isolated

    isolated_score = float(score_community_forensics_isolated(equivalent_view["path"]).score)
    difference = abs(batch_score - isolated_score)
    maximum = float(equivalence["absolute_difference_maximum"])
    if difference > maximum:
        raise ValueError(
            f"Batch/isolated Community score mismatch: {difference} exceeds {maximum}"
        )
    equivalence_result = {
        "source_asset_sha256": equivalent_view["source_asset_sha256"],
        "profile": equivalent_view["profile"],
        "batch_score": batch_score,
        "isolated_score": isolated_score,
        "absolute_difference": difference,
        "maximum": maximum,
    }
    batch_size = int(protocol["execution"]["batch_size"])
    pending = [view for view in views if (view["source_asset_sha256"], view["profile"]) not in completed]
    with torch.inference_mode():
        for offset in range(0, len(pending), batch_size):
            group = pending[offset : offset + batch_size]
            values = torch.cat(
                [support._model_input(view["path"], transform, None, None, 1.0) for view in group], dim=0
            )
            scores = torch.sigmoid(model(values)).reshape(-1).tolist()
            for view, score in zip(group, scores, strict=True):
                rows.append(
                    {
                        "source_manifest_index": view["source_manifest_index"],
                        "source_asset_sha256": view["source_asset_sha256"],
                        "profile": view["profile"],
                        "artifact_sha256": view["artifact_sha256"],
                        "score": float(score),
                    }
                )
            _atomic_write_json(partial_path, _partial(protocol_hash, rows))
            print(f"community_scored={len(rows)}/{len(views)}", flush=True)
    if len(rows) != len(views):
        raise ValueError("Incomplete propagation score rows")
    return rows, equivalence_result


def _rate(rows: list[Mapping[str, Any]], threshold: float) -> dict[str, int | float]:
    if not rows:
        raise ValueError("Cannot summarize empty rate")
    hits = sum(float(row["score"]) >= threshold for row in rows)
    return {"hits": hits, "total": len(rows), "rate": hits / len(rows)}


def _summarize(protocol: Mapping[str, Any], scores: list[dict[str, Any]], source_records: list[dict[str, Any]]) -> dict[str, Any]:
    source_by_asset = {str(record["asset_sha256"]).lower(): record for record in source_records}
    if len(source_by_asset) != len(source_records):
        raise ValueError("Duplicate source assets are not allowed")
    audits = _read_json(Path(protocol["model"]["audit_path"]))
    high = float(audits["high_confidence_threshold"])
    limited = float(audits["limited_review_threshold"])
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_profile[str(row["profile"])].append(row)
    profiles: dict[str, Any] = {}
    for profile in PROFILES:
        rows = by_profile[profile]
        generated = [row for row in rows if source_by_asset[str(row["source_asset_sha256"])]["label"] == "fake"]
        real = [row for row in rows if source_by_asset[str(row["source_asset_sha256"])]["label"] == "real"]
        profiles[profile] = {
            "generated_high_threshold": _rate(generated, high),
            "real_high_threshold": _rate(real, high),
            "generated_limited_threshold": _rate(generated, limited),
            "real_limited_threshold": _rate(real, limited),
            "product_high_eligible": profile != "webp_reencode_quality=85",
        }
    return {
        "high_confidence_threshold": high,
        "limited_review_threshold": limited,
        "profile_metrics": profiles,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    views, sources = _resolve_views(args.repo_root, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores, equivalence = _score_views(args, protocol, protocol_hash, views)
    summary = _summarize(protocol, scores, sources)
    report = {
        "schema_version": "demirror-community-propagation-audit-v1",
        "purpose": "Frozen threshold propagation audit only; no threshold, product score, dependency or web-policy change is authorized.",
        "protocol_sha256": protocol_hash,
        "score_checkpoint_sha256": _sha256(args.output_dir / "scores.partial.json"),
        "view_count": len(views),
        "runtime_equivalence": equivalence,
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
    parser.add_argument("--repo-root", type=_path, default=Path.cwd())
    parser.add_argument("--protocol", type=_path, default=Path("research/records/2026-08-19/pixel/community_propagation_audit_protocol_v1.json"))
    parser.add_argument("--output-dir", type=_path, default=Path("outputs/research/community_propagation_audit_v1"))
    parser.add_argument("--audit-output", type=_path, default=Path("research/records/2026-08-19/pixel/community_propagation_audit_v1.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
