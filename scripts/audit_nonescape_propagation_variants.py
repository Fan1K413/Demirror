"""Audit frozen propagation views with the registered Nonescape Mini detector.

This is a research-only CPU batch runner.  It duplicates the product worker's
architecture and preprocessing, verifies a fixed isolated-worker equivalence
probe before batch scoring, and keeps label-free score checkpoints.  It cannot
change product thresholds, score components, or web behaviour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def _resolve_views(
    repo_root: Path, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        "schema_version": "demirror-nonescape-propagation-score-partial-v1",
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
    expected = {
        "source_manifest_index",
        "source_asset_sha256",
        "profile",
        "artifact_sha256",
        "score",
    }
    completed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("Partial score row is not a label-free registered row")
        completed.append(dict(row))
    return completed


def _load_runtime(checkpoint_path: Path, cpu_threads: int) -> tuple[Any, Any, Any]:
    """Build exactly the architecture and transform used by the product worker."""

    import torch
    import torchvision.models as models
    import torchvision.transforms.v2 as transforms
    from safetensors.torch import load_file
    from torch import nn

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)

    class _NonescapeMini(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = models.efficientnet_v2_s(
                weights=None,
                num_classes=1024,
                dropout=0.2,
            )
            self.head = nn.Linear(1024, 2)

        def forward(self, values: Any) -> Any:
            embedding = self.backbone(values)
            return torch.softmax(self.head(embedding), dim=-1)

    model = _NonescapeMini()
    state = load_file(str(checkpoint_path), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.JPEG(quality=100),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return torch, model, transform


def _model_input(path: Path, transform: Any) -> Any:
    from PIL import Image

    with Image.open(path) as source:
        return transform(source.convert("RGB")).unsqueeze(0)


def _score_views(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    views: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    implementation = args.repo_root / str(protocol["implementation"]["audit_script_path"])
    _require_hash(implementation, str(protocol["implementation"]["audit_script_sha256"]))
    runtime_adapter = args.repo_root / str(protocol["implementation"]["runtime_adapter_path"])
    _require_hash(runtime_adapter, str(protocol["implementation"]["runtime_adapter_sha256"]))
    checkpoint_path = args.repo_root / str(protocol["model"]["checkpoint_path"])
    _require_hash(checkpoint_path, str(protocol["model"]["checkpoint_sha256"]))

    partial_path = args.output_dir / "scores.partial.json"
    rows = _completed_rows(partial_path, protocol_hash) if args.resume else []
    completed = {(str(row["source_asset_sha256"]), str(row["profile"])) for row in rows}
    if rows and not args.resume:
        raise FileExistsError("Existing score checkpoint requires --resume")
    if len(completed) != len(rows):
        raise ValueError("Partial score checkpoint has duplicate source/profile rows")

    execution = protocol["execution"]
    torch, model, transform = _load_runtime(checkpoint_path, int(execution["cpu_threads"]))
    equivalence = protocol["implementation"]["equivalence_probe"]
    equivalent_view = next(
        view
        for view in views
        if view["source_asset_sha256"] == str(equivalence["source_asset_sha256"])
        and view["profile"] == str(equivalence["profile"])
    )
    with torch.inference_mode():
        batch_score = float(model(_model_input(equivalent_view["path"], transform))[0, 1].item())
    from image_trust.ai_likelihood.nonescape import score_nonescape_mini_isolated

    isolated_score = float(score_nonescape_mini_isolated(equivalent_view["path"]).score)
    difference = abs(batch_score - isolated_score)
    maximum = float(equivalence["absolute_difference_maximum"])
    if difference > maximum:
        raise ValueError(
            f"Batch/isolated Nonescape score mismatch: {difference} exceeds {maximum}"
        )
    equivalence_result = {
        "source_asset_sha256": equivalent_view["source_asset_sha256"],
        "profile": equivalent_view["profile"],
        "batch_score": batch_score,
        "isolated_score": isolated_score,
        "absolute_difference": difference,
        "maximum": maximum,
    }

    batch_size = int(execution["batch_size"])
    pending = [view for view in views if (view["source_asset_sha256"], view["profile"]) not in completed]
    with torch.inference_mode():
        for offset in range(0, len(pending), batch_size):
            group = pending[offset : offset + batch_size]
            values = torch.cat([_model_input(view["path"], transform) for view in group], dim=0)
            scores = model(values)[:, 1].tolist()
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
            print(f"nonescape_scored={len(rows)}/{len(views)}", flush=True)
    if len(rows) != len(views):
        raise ValueError("Incomplete propagation score rows")
    return rows, equivalence_result


def _rate(rows: list[Mapping[str, Any]], threshold: float) -> dict[str, int | float]:
    if not rows:
        raise ValueError("Cannot summarize empty rate")
    hits = sum(float(row["score"]) >= threshold for row in rows)
    return {"hits": hits, "total": len(rows), "rate": hits / len(rows)}


def _summarize(
    protocol: Mapping[str, Any], scores: list[dict[str, Any]], source_records: list[dict[str, Any]]
) -> dict[str, Any]:
    source_by_asset = {str(record["asset_sha256"]).lower(): record for record in source_records}
    if len(source_by_asset) != len(source_records):
        raise ValueError("Duplicate source assets are not allowed")
    audit_path = Path(protocol["model"]["audit_path"])
    audit = _read_json(audit_path)
    high = float(audit["high_confidence_threshold"])
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_profile[str(row["profile"])].append(row)
    profile_metrics: dict[str, Any] = {}
    for profile in PROFILES:
        rows = by_profile[profile]
        generated = [row for row in rows if source_by_asset[str(row["source_asset_sha256"])] ["label"] == "fake"]
        real = [row for row in rows if source_by_asset[str(row["source_asset_sha256"])] ["label"] == "real"]
        profile_metrics[profile] = {
            "generated_high_threshold": _rate(generated, high),
            "real_high_threshold": _rate(real, high),
            "product_high_eligible": profile != "webp_reencode_quality=85",
        }
    return {"high_confidence_threshold": high, "profile_metrics": profile_metrics}


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    views, sources = _resolve_views(args.repo_root, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores, equivalence = _score_views(args, protocol, protocol_hash, views)
    summary = _summarize(protocol, scores, sources)
    report = {
        "schema_version": "demirror-nonescape-propagation-audit-v1",
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
    parser.add_argument(
        "--protocol",
        type=_path,
        default=Path("research/records/2026-08-19/pixel/nonescape_propagation_audit_protocol_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=Path("outputs/research/nonescape_propagation_audit_v1"),
    )
    parser.add_argument(
        "--audit-output",
        type=_path,
        default=Path("research/records/2026-08-19/pixel/nonescape_propagation_audit_v1.json"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
