"""Audit frozen propagation views with the registered SAFE detector.

The batch path copies the product worker's bounded CPU model construction,
wavelet preprocessing, centre crop and score direction.  A registered isolated
worker probe must match before any label-free batch checkpoint is written.
"""

from __future__ import annotations

import argparse
import gc
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
        original = (source_path.parent / str(source.get("relative_path", ""))).resolve()
        try:
            original.relative_to(source_path.parent.resolve())
        except ValueError as exc:
            raise ValueError("Source path escapes manifest root") from exc
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
        "schema_version": "demirror-safe-propagation-score-partial-v2",
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
    common = {
        "source_manifest_index",
        "source_asset_sha256",
        "profile",
        "artifact_sha256",
        "status",
    }
    completed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Partial score row is not a label-free registered row")
        status = row.get("status")
        expected = common | ({"score"} if status == "available" else {"reason"})
        if status not in {"available", "unavailable"} or set(row) != expected:
            raise ValueError("Partial score row is not a label-free registered row")
        completed.append(dict(row))
    return completed


def _load_runtime(checkpoint_path: Path, cpu_threads: int) -> tuple[Any, Any]:
    """Construct exactly the product SAFE worker architecture and state checks."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from pytorch_wavelets import DWTForward

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Any = None) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, value: Any) -> Any:
            identity = value
            value = self.relu(self.bn1(self.conv1(value)))
            value = self.relu(self.bn2(self.conv2(value)))
            value = self.bn3(self.conv3(value))
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.relu(value + identity)

    class SafeResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 3)
            self.layer2 = self._make_layer(128, 4, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(512, 2)
            self.dwt = DWTForward(J=1, mode="symmetric", wave="bior1.3")

        def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> Any:
            downsample = None
            if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * Bottleneck.expansion),
                )
            layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * Bottleneck.expansion
            layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, value: Any) -> Any:
            _, high = self.dwt(value)
            value = functional.interpolate(
                high[0][:, :, 2],
                size=value.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            value = self.maxpool(self.relu(self.bn1(self.conv1(value))))
            value = self.layer2(self.layer1(value))
            return self.fc1(self.avgpool(value).flatten(1))

    model = SafeResNet()
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("safe_checkpoint_model_state_missing")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("dwt.") for name in missing):
        raise ValueError(f"safe_checkpoint_state_mismatch:{missing}:{unexpected}")
    del checkpoint
    gc.collect()
    model.eval()
    return torch, model


def _model_input(path: Path) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    if min(image.size) < 256:
        raise ValueError("safe_input_too_small")
    left = (image.width - 256) // 2
    top = (image.height - 256) // 2
    image = image.crop((left, top, left + 256, top + 256))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


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

    torch, model = _load_runtime(checkpoint_path, int(protocol["execution"]["cpu_threads"]))
    equivalence = protocol["implementation"]["equivalence_probe"]
    equivalent_view = next(
        view
        for view in views
        if view["source_asset_sha256"] == str(equivalence["source_asset_sha256"])
        and view["profile"] == str(equivalence["profile"])
    )
    with torch.inference_mode():
        batch_score = float(torch.softmax(model(_model_input(equivalent_view["path"]).unsqueeze(0)), dim=1)[0, 1].item())
    from image_trust.ai_likelihood.safe import score_safe_isolated

    isolated_score = float(score_safe_isolated(equivalent_view["path"]).score)
    difference = abs(batch_score - isolated_score)
    maximum = float(equivalence["absolute_difference_maximum"])
    if difference > maximum:
        raise ValueError(f"Batch/isolated SAFE score mismatch: {difference} exceeds {maximum}")
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
            available: list[tuple[dict[str, Any], Any]] = []
            for view in group:
                try:
                    tensor = _model_input(view["path"])
                except ValueError as error:
                    if str(error) != "safe_input_too_small":
                        raise
                    rows.append(
                        {
                            "source_manifest_index": view["source_manifest_index"],
                            "source_asset_sha256": view["source_asset_sha256"],
                            "profile": view["profile"],
                            "artifact_sha256": view["artifact_sha256"],
                            "status": "unavailable",
                            "reason": "safe_input_too_small",
                        }
                    )
                else:
                    available.append((view, tensor))
            if available:
                values = torch.stack([tensor for _, tensor in available])
                scores = torch.softmax(model(values), dim=1)[:, 1].tolist()
                for (view, _), score in zip(available, scores, strict=True):
                    rows.append(
                        {
                            "source_manifest_index": view["source_manifest_index"],
                            "source_asset_sha256": view["source_asset_sha256"],
                            "profile": view["profile"],
                            "artifact_sha256": view["artifact_sha256"],
                            "status": "available",
                            "score": float(score),
                        }
                    )
            _atomic_write_json(partial_path, _partial(protocol_hash, rows))
            print(f"safe_scored={len(rows)}/{len(views)}", flush=True)
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
    audit = _read_json(Path(protocol["model"]["audit_path"]))
    high = float(audit["high_confidence_threshold"])
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_profile[str(row["profile"])].append(row)
    profile_metrics: dict[str, Any] = {}
    for profile in PROFILES:
        rows = by_profile[profile]
        available = [row for row in rows if row["status"] == "available"]
        unavailable = [row for row in rows if row["status"] == "unavailable"]
        generated = [row for row in available if source_by_asset[str(row["source_asset_sha256"])]["label"] == "fake"]
        real = [row for row in available if source_by_asset[str(row["source_asset_sha256"])]["label"] == "real"]
        reasons: dict[str, int] = defaultdict(int)
        for row in unavailable:
            reasons[str(row["reason"])] += 1
        profile_metrics[profile] = {
            "generated_high_threshold": _rate(generated, high),
            "real_high_threshold": _rate(real, high),
            "available_count": len(available),
            "unavailable_count": len(unavailable),
            "unavailable_reasons": dict(sorted(reasons.items())),
            "product_high_eligible": profile == "original_decode",
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
        "schema_version": "demirror-safe-propagation-audit-v2",
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
        default=Path("research/records/2026-08-19/pixel/safe_propagation_audit_protocol_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=Path("outputs/research/safe_propagation_audit_v2"),
    )
    parser.add_argument(
        "--audit-output",
        type=_path,
        default=Path("research/records/2026-08-19/pixel/safe_propagation_audit_v2.json"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
