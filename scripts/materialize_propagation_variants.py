"""Materialize deterministic, metadata-free propagation views for research.

The generated files live below the ignored ``outputs/`` directory.  This tool
only verifies input hashes and applies fixed raster operations; it does not
read labels for any decision, load a detector, select thresholds, or modify
the product.  A downstream candidate must bind this output manifest before it
can report any transform-specific score.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


PROFILE_ORDER = (
    "jpeg_reencode_quality=85",
    "webp_reencode_quality=85",
    "resize_longest=1024_restore_png",
    "screenshot_raster_png_longest=1600",
)
_PROFILE_EXTENSIONS = {
    "jpeg_reencode_quality=85": ".jpg",
    "webp_reencode_quality=85": ".webp",
    "resize_longest=1024_restore_png": ".png",
    "screenshot_raster_png_longest=1600": ".png",
}


@dataclass(frozen=True)
class SourceRecord:
    manifest_index: int
    asset_sha256: str
    path: Path


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


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _safe_relative_path(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source relative path escapes manifest root: {value!r}") from exc
    return candidate


def _read_source_records(manifest_path: Path, expected_count: int) -> list[SourceRecord]:
    manifest = _read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} records in {manifest_path}, got "
            f"{len(records) if isinstance(records, list) else 'non-list'}"
        )
    output: list[SourceRecord] = []
    seen_hashes: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid source manifest record at index {index}")
        asset_sha256 = str(record.get("asset_sha256", "")).lower()
        if len(asset_sha256) != 64 or any(char not in "0123456789abcdef" for char in asset_sha256):
            raise ValueError(f"Invalid asset SHA-256 at index {index}")
        if asset_sha256 in seen_hashes:
            raise ValueError(f"Duplicate asset SHA-256 in source manifest: {asset_sha256}")
        seen_hashes.add(asset_sha256)
        path = _safe_relative_path(manifest_path.parent, str(record.get("relative_path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"Source image is missing: {path}")
        if _sha256(path) != asset_sha256:
            raise ValueError(f"Source image SHA-256 mismatch: {path}")
        output.append(SourceRecord(index, asset_sha256, path))
    return output


def _decode_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
    return image


def _resize_longest(image: Image.Image, longest_edge: int) -> Image.Image:
    longest = max(image.size)
    if longest <= longest_edge:
        return image.copy()
    size = tuple(max(1, round(value * longest_edge / longest)) for value in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def _encode_variant(image: Image.Image, profile: str) -> tuple[bytes, dict[str, Any]]:
    """Return exact bytes and a non-label-bearing transform receipt."""
    original_size = image.size
    buffer = io.BytesIO()
    receipt: dict[str, Any] = {"input_size": list(original_size), "profile": profile}
    if profile == "jpeg_reencode_quality=85":
        image.save(buffer, format="JPEG", quality=85)
        receipt.update({"container": "JPEG", "quality": 85})
    elif profile == "webp_reencode_quality=85":
        image.save(buffer, format="WEBP", quality=85, lossless=False, method=4)
        receipt.update({"container": "WEBP", "quality": 85, "lossless": False, "method": 4})
    elif profile == "resize_longest=1024_restore_png":
        reduced = _resize_longest(image, 1024)
        changed = reduced.size != original_size
        restored = reduced.resize(original_size, Image.Resampling.LANCZOS) if changed else reduced
        restored.save(buffer, format="PNG")
        receipt.update(
            {
                "container": "PNG",
                "resize_longest": 1024,
                "resized": changed,
                "reduced_size": list(reduced.size),
                "restored_size": list(restored.size),
            }
        )
    elif profile == "screenshot_raster_png_longest=1600":
        raster = _resize_longest(image, 1600)
        raster.save(buffer, format="PNG")
        receipt.update(
            {
                "container": "PNG",
                "raster_longest": 1600,
                "raster_size": list(raster.size),
            }
        )
    else:
        raise ValueError(f"Unsupported propagation profile: {profile}")
    return buffer.getvalue(), receipt


def _verify_materialized(path: Path, expected_hash: str) -> dict[str, Any]:
    if _sha256(path) != expected_hash:
        raise ValueError(f"Materialized artifact SHA-256 mismatch: {path}")
    with Image.open(path) as opened:
        opened.load()
        info = dict(opened.info)
        mode = opened.mode
        size = list(opened.size)
        exif_present = bool(opened.getexif())
    metadata_keys = sorted(key for key in ("exif", "icc_profile", "xmp") if info.get(key))
    if metadata_keys or exif_present:
        raise ValueError(f"Materialized artifact retains forbidden metadata: {path}")
    if mode != "RGB":
        raise ValueError(f"Materialized artifact is not RGB: {path}")
    return {"decoded_mode": mode, "decoded_size": size, "forbidden_metadata_keys": metadata_keys}


def _ensure_output_inside_repo(repo_root: Path, output_dir: Path) -> None:
    try:
        output_dir.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError("Output directory must be inside the repository root") from exc


def _partial_report(
    *,
    protocol_hash: str,
    registration: Mapping[str, Any],
    profiles: tuple[str, ...],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "demirror-propagation-variant-partial-v1",
        "purpose": "Resumable atomic checkpoint; never use this partial file as an evaluation manifest.",
        "protocol_sha256": protocol_hash,
        "source_manifest": {
            "path": str(registration["path"]),
            "sha256": str(registration["sha256"]),
            "record_count": int(registration["record_count"]),
        },
        "profiles": list(profiles),
        "records": records,
    }


def _resume_records(
    partial_path: Path,
    *,
    protocol_hash: str,
    registration: Mapping[str, Any],
    profiles: tuple[str, ...],
) -> list[dict[str, Any]]:
    partial = _read_json(partial_path)
    expected = _partial_report(
        protocol_hash=protocol_hash,
        registration=registration,
        profiles=profiles,
        records=[],
    )
    for key in ("schema_version", "protocol_sha256", "source_manifest", "profiles"):
        if partial.get(key) != expected[key]:
            raise ValueError(f"Partial manifest does not match the registered protocol: {key}")
    records = partial.get("records")
    if not isinstance(records, list):
        raise ValueError("Partial manifest has no record list")
    seen: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Partial manifest contains a non-object record")
        key = (str(record.get("source_asset_sha256")), str(record.get("profile")))
        if key in seen or key[1] not in profiles:
            raise ValueError("Partial manifest contains a duplicate or unsupported record")
        seen.add(key)
    return records


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    _ensure_output_inside_repo(args.repo_root, args.output_dir)
    registration = protocol["source_manifest"]
    manifest_path = args.repo_root / str(registration["path"])
    if _sha256(manifest_path) != str(registration["sha256"]):
        raise ValueError(f"Registered source manifest SHA-256 mismatch: {manifest_path}")
    sources = _read_source_records(manifest_path, int(registration["record_count"]))
    profiles = tuple(str(profile) for profile in protocol["profiles"])
    if profiles != PROFILE_ORDER:
        raise ValueError("Protocol profiles do not match this materializer's fixed profile order")
    partial_path = args.output_dir / "manifest.partial.json"
    if args.output_dir.exists():
        if not args.resume:
            if any(args.output_dir.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite non-empty output directory: {args.output_dir}"
                )
            args.output_dir.rmdir()
            args.output_dir.mkdir(parents=True, exist_ok=False)
            records: list[dict[str, Any]] = []
        elif not partial_path.is_file():
            raise FileExistsError(
                "Refusing to resume output without an atomic partial manifest; "
                f"remove the explicitly failed output before restarting: {args.output_dir}"
            )
        else:
            records = _resume_records(
                partial_path,
                protocol_hash=protocol_hash,
                registration=registration,
                profiles=profiles,
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        records = []
    completed = {
        (str(record["source_asset_sha256"]), str(record["profile"])) for record in records
    }
    for source in sources:
        image = _decode_rgb(source.path)
        for profile in profiles:
            if (source.asset_sha256, profile) in completed:
                continue
            payload, receipt = _encode_variant(image, profile)
            extension = _PROFILE_EXTENSIONS[profile]
            relative = Path(profile.replace("=", "_")) / source.asset_sha256[:2] / (
                source.asset_sha256 + extension
            )
            artifact = args.output_dir / relative
            _atomic_write_bytes(artifact, payload)
            artifact_hash = _sha256(artifact)
            decoded = _verify_materialized(artifact, artifact_hash)
            records.append(
                {
                    "source_manifest_index": source.manifest_index,
                    "source_asset_sha256": source.asset_sha256,
                    "profile": profile,
                    "relative_path": relative.as_posix(),
                    "artifact_sha256": artifact_hash,
                    "byte_size": artifact.stat().st_size,
                    "transform": receipt,
                    "verification": decoded,
                }
            )
            _atomic_write_json(
                partial_path,
                _partial_report(
                    protocol_hash=protocol_hash,
                    registration=registration,
                    profiles=profiles,
                    records=records,
                ),
            )
            completed.add((source.asset_sha256, profile))
        if (source.manifest_index + 1) % 8 == 0 or source.manifest_index + 1 == len(sources):
            print(f"materialized_sources={source.manifest_index + 1}/{len(sources)}", flush=True)

    report = {
        "schema_version": "demirror-propagation-variant-manifest-v1",
        "purpose": "Deterministic metadata-free raster variants for offline robustness evaluation only.",
        "protocol_sha256": protocol_hash,
        "source_manifest": {
            "path": str(registration["path"]),
            "sha256": str(registration["sha256"]),
            "record_count": len(sources),
        },
        "profiles": list(profiles),
        "pillow_version": Image.__version__,
        "record_count": len(records),
        "label_handling": "Labels, generator names and source directories are not emitted in this derivative manifest and are not used by materialization.",
        "deployment_eligible": False,
        "runtime_policy_changed": False,
        "records": records,
    }
    _atomic_write_json(args.output_dir / "manifest.json", report)
    partial_path.unlink()
    return report


def _path(value: str) -> Path:
    return Path(value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=_path, default=Path.cwd())
    parser.add_argument(
        "--protocol",
        type=_path,
        default=Path("research/records/2026-08-19/pixel/propagation_variant_protocol_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=Path("outputs/research/propagation_variants_openfake_confirmation_v1"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from a matching atomic manifest.partial.json checkpoint.",
    )
    args = parser.parse_args()
    result = materialize(args)
    print(
        json.dumps(
            {
                "record_count": result["record_count"],
                "profiles": result["profiles"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
