"""Register an existing local P0 F6 set as a P1 control-only cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from image_trust.camera.dataset import (
    CalibrationDatasetPurpose,
    CalibrationDatasetRegistry,
    CalibrationDatasetSplit,
    CalibrationRegistryEntry,
    write_calibration_registry,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list):
        parser.error("source manifest must contain a fixtures list")
    entries: list[CalibrationRegistryEntry] = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            parser.error("source manifest fixtures must be JSON objects")
        relative_path = str(fixture["relative_path"])
        path = args.dataset_root / Path(*Path(relative_path).parts)
        if not path.is_file():
            parser.error(f"fixture image is missing: {relative_path}")
        with Image.open(path) as image:
            resolution = ImageOps.exif_transpose(image).size
        actual_hash = _sha256(path)
        expected_hash = str(fixture["original_file_hash"])
        if actual_hash != expected_hash:
            parser.error(f"fixture hash mismatch: {relative_path}")
        author = str(fixture.get("author", "unknown"))
        entries.append(
            CalibrationRegistryEntry(
                image_id=str(fixture["image_id"]),
                split=CalibrationDatasetSplit.CONTROL,
                relative_path=relative_path,
                source_type="camera",
                source_url_or_internal_provenance=str(fixture["source_url"]),
                license=str(fixture["license"]),
                capture_or_generator_family=f"commons-author:{author}",
                original_file_hash=actual_hash,
                transformations=[],
                resolution=resolution,
                c2pa_or_metadata_state="not_observed_not_checked",
            )
        )
    registry = CalibrationDatasetRegistry(
        cohort_name="p0-f6-real-camera-control-v2",
        intended_use=CalibrationDatasetPurpose.CONTROL_SMOKE,
        entries=entries,
    )
    write_calibration_registry(args.output, registry)
    print(f"registered_control_entries={len(entries)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
