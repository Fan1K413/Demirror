from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_current_pixel_variant", ROOT / "scripts" / "probe_current_pixel_variant.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resolve_input_uses_registered_variant_without_reading_label(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    image_path = source_root / "fake" / "image.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (16, 16), (3, 4, 5)).save(image_path)
    asset_hash = _hash(image_path)
    source_manifest = source_root / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {"records": [{"relative_path": "fake/image.jpg", "asset_sha256": asset_hash, "label": "fake"}]}
        ),
        encoding="utf-8",
    )
    variant_root = tmp_path / "variants"
    variant_root.mkdir()
    artifact = variant_root / "view.png"
    Image.new("RGB", (16, 16), (5, 4, 3)).save(artifact)
    artifact_hash = _hash(artifact)
    variant_manifest = variant_root / "manifest.json"
    variant_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_asset_sha256": asset_hash,
                        "profile": "screenshot_raster_png_longest=1600",
                        "relative_path": "view.png",
                        "artifact_sha256": artifact_hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    protocol = {
        "source_manifest": {"path": "source/manifest.json", "sha256": _hash(source_manifest)},
        "variant_manifest": {"path": "variants/manifest.json", "sha256": _hash(variant_manifest)},
        "selection": {
            "manifest_index": 0,
            "asset_sha256": asset_hash,
            "profile": "screenshot_raster_png_longest=1600",
        },
    }

    path, profile, index, output_hash = probe._resolve_input(tmp_path, protocol)

    assert path == artifact.resolve()
    assert profile == "screenshot_raster_png_longest=1600"
    assert index == 0
    assert output_hash == asset_hash


def test_resolve_input_rejects_wrong_registered_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"records": []}), encoding="utf-8")
    protocol = {
        "source_manifest": {"path": "source.json", "sha256": _hash(source)},
        "selection": {"manifest_index": 0, "asset_sha256": "0" * 64, "profile": "original_decode"},
    }
    with pytest.raises(ValueError, match="index"):
        probe._resolve_input(tmp_path, protocol)
