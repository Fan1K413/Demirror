"""Generate reproducible, local-only fixtures for P0 measurement validation.

The generated files live under ``data/`` (which is intentionally ignored by Git).
They are synthetic project fixtures, not source-provenance evidence or AI examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, PngImagePlugin


GENERATOR = "scripts/generate_p0_fixtures.py"
LICENSE = "project-generated internal test fixture"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_image(size: tuple[int, int] = (640, 480)) -> Image.Image:
    return Image.new("RGB", size, (245, 245, 245))


def _write_vp_image(path: Path, variant: int, multi: bool) -> None:
    image = _base_image()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    left_vp = (90 + variant * 9, 110 + variant * 4)
    right_vp = (550 - variant * 7, 105 + variant * 3)
    anchors = (70, 145, 220, 300, 380, 465, 545, 610)
    for index, x in enumerate(anchors):
        draw.line((left_vp[0], left_vp[1], x, height - 24 - (index % 2) * 18), fill=(30, 30, 30), width=3)
    if multi:
        for index, x in enumerate(anchors):
            draw.line((right_vp[0], right_vp[1], x, height - 50 - (index % 3) * 16), fill=(55, 55, 55), width=3)
        for y in (220, 290, 360, 430):
            draw.line((55, y, width - 55, y), fill=(85, 85, 85), width=2)
    draw.rectangle((25, 25, width - 25, height - 25), outline=(110, 110, 110), width=2)
    image.save(path)


def _write_low_geometry(path: Path, variant: int) -> None:
    colors = (
        (120, 120, 120),
        (150, 150, 150),
        (100, 120, 140),
        (140, 110, 100),
        (190, 190, 190),
        (80, 95, 100),
        (220, 215, 205),
        (110, 90, 125),
    )
    image = Image.new("RGB", (320, 240), colors[variant])
    if variant in {4, 5}:
        draw = ImageDraw.Draw(image)
        draw.line((30, 120, 290, 120), fill=(colors[variant][0] + 5, colors[variant][1] + 5, colors[variant][2] + 5), width=1)
    image.save(path)


def _write_special_png(path: Path, marker: str) -> None:
    image = _base_image()
    draw = ImageDraw.Draw(image)
    draw.line((70, 420, 320, 100), fill=(20, 20, 20), width=4)
    draw.line((570, 420, 320, 100), fill=(20, 20, 20), width=4)
    info = PngImagePlugin.PngInfo()
    info.add_text("XMP", marker)
    image.save(path, pnginfo=info)


def _record(path: Path, group: str, expected: dict[str, str] | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "image_id": path.stem,
        "group": group,
        "relative_path": path.name,
        "generator_script": GENERATOR,
        "license": LICENSE,
        "sha256": _sha256(path),
        "redistribution": "not_required; generated under ignored data/",
    }
    if expected:
        record["expected"] = expected
    return record


def generate(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    # F1: twelve file-gate fixtures. Some are evaluated with deliberately lower limits.
    valid = _base_image((128, 96))
    for suffix, image_format in (("png", "PNG"), ("jpg", "JPEG"), ("webp", "WEBP")):
        path = output / f"f1_valid.{suffix}"
        valid.save(path, format=image_format)
        records.append(_record(path, "F1_file_gate"))
    mismatch = output / "f1_png_named_jpg.jpg"
    valid.save(mismatch, format="PNG")
    records.append(_record(mismatch, "F1_file_gate"))
    for index, color in enumerate(((10, 10, 10), (30, 30, 30), (60, 60, 60)), start=1):
        path = output / f"f1_small_{index}.png"
        Image.new("RGB", (20 + index, 20 + index), color).save(path)
        records.append(_record(path, "F1_file_gate"))
    for index in range(2):
        path = output / f"f1_corrupt_{index + 1}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\ntruncated-payload")
        records.append(_record(path, "F1_file_gate"))
    for index in range(2):
        path = output / f"f1_unsupported_{index + 1}.png"
        path.write_bytes(f"not an image {index}".encode("ascii"))
        records.append(_record(path, "F1_file_gate"))
    oversized_pixels = output / "f1_large_pixels.png"
    _base_image((300, 300)).save(oversized_pixels)
    records.append(_record(oversized_pixels, "F1_file_gate"))
    rotated = output / "f1_exif_orientation.jpg"
    exif = valid.getexif()
    exif[274] = 6
    valid.save(rotated, format="JPEG", exif=exif)
    records.append(_record(rotated, "F1_file_gate"))

    for variant in range(1, 7):
        path = output / f"f2_single_vp_{variant}.png"
        _write_vp_image(path, variant, multi=False)
        records.append(_record(path, "F2_single_vp", {"run_status": "ok"}))
    for variant in range(1, 7):
        path = output / f"f3_multi_vp_{variant}.png"
        _write_vp_image(path, variant, multi=True)
        records.append(_record(path, "F3_multi_vp", {"run_status": "ok"}))
    for variant in range(8):
        path = output / f"f4_low_geometry_{variant + 1}.png"
        _write_low_geometry(path, variant)
        records.append(
            _record(path, "F4_low_geometry", {"run_status": "not_applicable"})
        )

    special_markers = (
        ("f5_panorama_metadata.png", "GPano:ProjectionType=equirectangular"),
        ("f5_equirectangular_metadata.png", "ProjectionType=equirectangular"),
        ("f5_fisheye_metadata.png", "LensModel=Fisheye"),
    )
    for filename, marker in special_markers:
        path = output / filename
        _write_special_png(path, marker)
        records.append(
            _record(path, "F5_special_imaging", {"run_status": "not_applicable"})
        )
    animated = output / "f5_animated.webp"
    first = Image.new("RGB", (128, 96), (10, 10, 10))
    second = Image.new("RGB", (128, 96), (230, 230, 230))
    first.save(animated, format="WEBP", save_all=True, append_images=[second], duration=100, loop=0)
    records.append(_record(animated, "F5_special_imaging", {"run_status": "rejected"}))

    manifest = {
        "schema_version": "p0-fixture-manifest-v1",
        "generator": GENERATOR,
        "fixture_root": str(output),
        "fixtures": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reproducible P0 fixtures.")
    parser.add_argument("--output", type=Path, default=Path("data/p0_fixtures"))
    args = parser.parse_args()
    manifest_path = generate(args.output)
    print(f"Generated P0 fixtures and manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
