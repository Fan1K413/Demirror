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
    "materialize_propagation_variants", ROOT / "scripts" / "materialize_propagation_variants.py"
)
assert SPEC is not None and SPEC.loader is not None
variants = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = variants
SPEC.loader.exec_module(variants)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_profiles_produce_expected_container_and_dimensions() -> None:
    image = Image.new("RGB", (2000, 1000), (12, 34, 56))

    expected = {
        "jpeg_reencode_quality=85": ("JPEG", (2000, 1000)),
        "webp_reencode_quality=85": ("WEBP", (2000, 1000)),
        "resize_longest=1024_restore_png": ("PNG", (2000, 1000)),
        "screenshot_raster_png_longest=1600": ("PNG", (1600, 800)),
    }
    for profile, (container, size) in expected.items():
        payload, receipt = variants._encode_variant(image, profile)
        with Image.open(__import__("io").BytesIO(payload)) as result:
            assert result.format == container
            assert result.mode == "RGB"
            assert result.size == size
        assert receipt["profile"] == profile


def test_safe_relative_path_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        variants._safe_relative_path(tmp_path, "../outside.jpg")


def test_materialize_keeps_labels_out_of_output_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    image_path = source_root / "fake" / "sample.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (32, 16), (1, 2, 3)).save(image_path, format="JPEG")
    asset_hash = _sha256(image_path)
    manifest_path = source_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "relative_path": "fake/sample.jpg",
                        "asset_sha256": asset_hash,
                        "label": "fake",
                        "model": "not-for-output",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "source_manifest": {
                    "path": "source/manifest.json",
                    "sha256": _sha256(manifest_path),
                    "record_count": 1,
                },
                "profiles": list(variants.PROFILE_ORDER),
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    args = type(
        "Args",
        (),
        {
            "repo_root": tmp_path,
            "protocol": protocol_path,
            "output_dir": output_dir,
            "resume": False,
        },
    )()
    result = variants.materialize(args)

    assert result["record_count"] == 4
    serialised = (output_dir / "manifest.json").read_text(encoding="utf-8")
    assert "not-for-output" not in serialised
    assert '"label"' not in serialised
    assert "fake/sample.jpg" not in serialised
    for record in result["records"]:
        artifact = output_dir / record["relative_path"]
        assert artifact.is_file()
        assert record["verification"]["forbidden_metadata_keys"] == []
    with pytest.raises(FileExistsError, match="non-empty"):
        variants.materialize(args)


def test_resume_requires_matching_atomic_partial_manifest(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    partial = output / "manifest.partial.json"
    partial.write_text("{}", encoding="utf-8")
    registration = {"path": "source/manifest.json", "sha256": "a" * 64, "record_count": 1}

    with pytest.raises(ValueError, match="registered protocol"):
        variants._resume_records(
            partial,
            protocol_hash="b" * 64,
            registration=registration,
            profiles=variants.PROFILE_ORDER,
        )
