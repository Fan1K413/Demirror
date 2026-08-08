from pathlib import Path

import pytest
from PIL import Image

from image_trust.ingest import InputRejectedError, detect_format, ingest_image
from image_trust.schemas import IngestConfig


def test_detect_format_by_magic() -> None:
    assert detect_format(b"\x89PNG\r\n\x1a\nrest") == "png"
    assert detect_format(b"\xff\xd8\xff\xe0") == "jpeg"
    assert detect_format(b"RIFF\x00\x00\x00\x00WEBP") == "webp"
    assert detect_format(b"not-an-image") is None


def test_ingest_keeps_magic_and_reports_filename_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "misleading.jpg"
    Image.new("RGB", (128, 96), (255, 0, 0)).save(path, format="PNG")
    image = ingest_image(path, IngestConfig(), 1024)
    assert image.summary.detected_format == "png"
    assert image.summary.filename_format_mismatch is True
    assert "filename_format_mismatch" in image.limitations


def test_ingest_rejects_small_image(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    Image.new("RGB", (20, 20), (0, 0, 0)).save(path)
    with pytest.raises(InputRejectedError) as raised:
        ingest_image(path, IngestConfig(min_side_px=64), 1024)
    assert raised.value.code == "image_too_small"


def test_ingest_normalizes_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (160, 96), (0, 100, 200))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    ingested = ingest_image(path, IngestConfig(), 1024)
    assert ingested.summary.encoded_size == (160, 96)
    assert ingested.summary.canonical_size == (96, 160)
    assert ingested.transform.orientation_applied is True
    assert ingested.summary.coordinate_transform.exif_orientation == 6


@pytest.mark.parametrize(
    ("suffix", "image_format"),
    [("png", "PNG"), ("jpg", "JPEG"), ("webp", "WEBP")],
)
def test_ingest_accepts_each_supported_static_format(
    tmp_path: Path,
    suffix: str,
    image_format: str,
) -> None:
    path = tmp_path / f"input.{suffix}"
    Image.new("RGB", (128, 96), (0, 40, 80)).save(path, format=image_format)
    ingested = ingest_image(path, IngestConfig(), 1024)
    assert ingested.summary.detected_format == ("jpeg" if suffix == "jpg" else suffix)
    assert ingested.summary.validation_limits["max_pixels"] == 40_000_000
    assert ingested.summary.exif_summary["has_icc_profile"] is False
    assert ingested.summary.exif_summary["icc_profile_bytes"] == 0


def test_ingest_rejects_corrupt_magic_and_strict_filename_mismatch(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-complete-png")
    with pytest.raises(InputRejectedError) as corrupt_error:
        ingest_image(corrupt, IngestConfig(), 1024)
    assert corrupt_error.value.code == "decode_failed"

    mismatch = tmp_path / "mismatch.jpg"
    Image.new("RGB", (128, 96)).save(mismatch, format="PNG")
    with pytest.raises(InputRejectedError) as mismatch_error:
        ingest_image(
            mismatch,
            IngestConfig(allow_filename_format_mismatch=False),
            1024,
        )
    assert mismatch_error.value.code == "filename_format_mismatch"


def test_ingest_enforces_file_and_pixel_limits(tmp_path: Path) -> None:
    path = tmp_path / "normal.png"
    Image.new("RGB", (128, 96)).save(path)
    with pytest.raises(InputRejectedError) as file_error:
        ingest_image(path, IngestConfig(max_file_bytes=1), 1024)
    assert file_error.value.code == "file_too_large"

    with pytest.raises(InputRejectedError) as pixel_error:
        ingest_image(path, IngestConfig(max_pixels=1_000), 1024)
    assert pixel_error.value.code == "pixel_limit_exceeded"


def test_ingest_rejects_animated_webp(tmp_path: Path) -> None:
    path = tmp_path / "animated.webp"
    first = Image.new("RGB", (128, 96), (0, 0, 0))
    second = Image.new("RGB", (128, 96), (255, 255, 255))
    first.save(path, format="WEBP", save_all=True, append_images=[second], duration=100, loop=0)
    with pytest.raises(InputRejectedError) as raised:
        ingest_image(path, IngestConfig(), 1024)
    assert raised.value.code == "animated_image_not_supported"
