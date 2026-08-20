"""Run one source- and label-blind compatibility probe through registered pixels.

This script deliberately scores only the exact source asset and propagation
profile registered in its protocol.  Each existing detector remains in its own
short-lived production worker; the resulting record is a compatibility probe,
not a threshold-selection or accuracy experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping


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
        raise FileNotFoundError(f"Missing registered input: {path}")
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def _source_record(source_manifest: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    records = source_manifest.get("records")
    if not isinstance(records, list) or not 0 <= index < len(records):
        raise ValueError("Registered source manifest index is unavailable")
    record = records[index]
    if not isinstance(record, dict):
        raise ValueError("Registered source manifest record is invalid")
    return record


def _resolve_input(
    repo_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[Path, str, int, str]:
    source_registration = protocol["source_manifest"]
    source_path = repo_root / str(source_registration["path"])
    _require_hash(source_path, str(source_registration["sha256"]))
    source = _source_record(_read_json(source_path), int(protocol["selection"]["manifest_index"]))
    asset_hash = str(source.get("asset_sha256", "")).lower()
    if asset_hash != str(protocol["selection"]["asset_sha256"]).lower():
        raise ValueError("Source asset does not match the frozen selection")
    profile = str(protocol["selection"]["profile"])
    if profile == "original_decode":
        candidate = (source_path.parent / str(source.get("relative_path", ""))).resolve()
        try:
            candidate.relative_to(source_path.parent.resolve())
        except ValueError as exc:
            raise ValueError("Source relative path escapes manifest root") from exc
        _require_hash(candidate, asset_hash)
        return candidate, profile, int(protocol["selection"]["manifest_index"]), asset_hash

    variants_registration = protocol["variant_manifest"]
    variants_path = repo_root / str(variants_registration["path"])
    _require_hash(variants_path, str(variants_registration["sha256"]))
    variants = _read_json(variants_path)
    matching = [
        record
        for record in variants.get("records", [])
        if isinstance(record, dict)
        and str(record.get("source_asset_sha256", "")).lower() == asset_hash
        and str(record.get("profile", "")) == profile
    ]
    if len(matching) != 1:
        raise ValueError("Expected exactly one registered propagation artifact")
    artifact = (variants_path.parent / str(matching[0].get("relative_path", ""))).resolve()
    try:
        artifact.relative_to(variants_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("Propagation relative path escapes variant root") from exc
    _require_hash(artifact, str(matching[0].get("artifact_sha256", "")))
    return artifact, profile, int(protocol["selection"]["manifest_index"]), asset_hash


def _scorers() -> dict[str, Callable[[Path], Any]]:
    from image_trust.ai_likelihood.community_forensics import score_community_forensics_isolated
    from image_trust.ai_likelihood.dda import score_dda_isolated
    from image_trust.ai_likelihood.forensic_clip import score_forensic_clip_isolated
    from image_trust.ai_likelihood.nonescape import score_nonescape_mini_isolated
    from image_trust.ai_likelihood.safe import score_safe_isolated

    return {
        "dda": score_dda_isolated,
        "safe": score_safe_isolated,
        "forensic_clip": score_forensic_clip_isolated,
        "community_forensics": score_community_forensics_isolated,
        "nonescape_mini": score_nonescape_mini_isolated,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _read_json(args.protocol)
    protocol_hash = _sha256(args.protocol)
    input_path, profile, manifest_index, asset_hash = _resolve_input(args.repo_root, protocol)
    expected_order = [str(value) for value in protocol["detectors"]]
    available = _scorers()
    if expected_order != list(available):
        raise ValueError("Probe detector order differs from the registered five-channel order")

    rows: list[dict[str, Any]] = []
    for name in expected_order:
        started = time.perf_counter()
        try:
            result = available[name](input_path)
            row: dict[str, Any] = {
                "detector": name,
                "status": "available",
                "score": float(result.score),
                "preprocessing": str(result.preprocessing),
            }
        except Exception as error:  # Per-channel failure must remain observable.
            row = {
                "detector": name,
                "status": "unavailable",
                "error": f"{type(error).__name__}:{error}",
            }
        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)

    report = {
        "schema_version": "demirror-current-pixel-variant-compatibility-probe-v1",
        "purpose": "A single fixed, label-blind functional probe. It cannot estimate accuracy, choose a threshold, change a detector, or alter runtime policy.",
        "protocol_sha256": protocol_hash,
        "input": {
            "source_manifest_index": manifest_index,
            "source_asset_sha256": asset_hash,
            "profile": profile,
            "artifact_sha256": _sha256(input_path),
        },
        "rows": rows,
        "deployment_eligible": False,
        "runtime_policy_changed": False,
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
        default=Path(
            "research/records/2026-08-19/pixel/current_pixel_variant_compatibility_probe_protocol_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=_path,
        default=Path(
            "research/records/2026-08-19/pixel/current_pixel_variant_compatibility_probe_audit_v1.json"
        ),
    )
    args = parser.parse_args()
    report = run_probe(args)
    print(json.dumps({"statuses": [row["status"] for row in report["rows"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
