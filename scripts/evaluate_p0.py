"""Run P0 over a local fixture directory and write an auditable evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from image_trust.pipeline import analyze_image
from image_trust.utils.config import load_config


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_expectations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    fixtures = payload.get("fixtures", [])
    if not isinstance(fixtures, list):
        raise ValueError("Fixture manifest field 'fixtures' must be a list.")
    return {
        str(item["relative_path"]): item
        for item in fixtures
        if isinstance(item, dict) and "relative_path" in item
    }


def _iter_paths(input_dir: Path, recursive: bool) -> list[Path]:
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path for path in candidates if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _expectation_errors(result: dict[str, Any], fixture: dict[str, Any] | None) -> list[str]:
    if not fixture:
        return []
    expected = fixture.get("expected")
    if not isinstance(expected, dict):
        return []
    errors: list[str] = []
    for field in ("run_status", "observation"):
        expected_value = expected.get(field)
        if expected_value is not None and result["evidence"][field] != expected_value:
            errors.append(
                f"expected_{field}={expected_value}; actual={result['evidence'][field]}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Optional generated fixture manifest.")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    config = load_config(args.config)
    expectations = _load_expectations(args.manifest)
    paths = _iter_paths(args.input_dir, args.recursive)
    if not paths:
        parser.error("No PNG, JPEG, or WebP files found in input directory.")

    args.output.mkdir(parents=True, exist_ok=True)
    artifact_root = args.output / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "evaluation.jsonl"
    status_counts: Counter[str] = Counter()
    observation_counts: Counter[str] = Counter()
    mismatches = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for path in paths:
            relative_path = path.relative_to(args.input_dir).as_posix()
            digest = _sha256(path)
            artifact_dir = artifact_root / f"{path.stem}-{digest[:12]}"
            result = analyze_image(path, config, artifact_dir)
            dumped = result.model_dump(mode="json")
            errors = _expectation_errors(dumped, expectations.get(relative_path))
            mismatches += bool(errors)
            status_counts[dumped["evidence"]["run_status"]] += 1
            observation_counts[dumped["evidence"]["observation"]] += 1
            record = {
                "relative_path": relative_path,
                "sha256": digest,
                "artifact_dir": artifact_dir.relative_to(args.output).as_posix(),
                "result": dumped,
                "expectation_errors": errors,
            }
            manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "schema_version": "p0-evaluation-summary-v1",
        "input_dir": str(args.input_dir),
        "config_version": config.config_version,
        "image_count": len(paths),
        "run_status_counts": dict(sorted(status_counts.items())),
        "observation_counts": dict(sorted(observation_counts.items())),
        "expectation_mismatch_count": mismatches,
        "manifest": manifest_path.name,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
