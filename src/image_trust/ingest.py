"""File validation, decoding, hashing, and canonical-image preparation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from image_trust.schemas import IngestConfig, InputSummary
from image_trust.utils.coordinates import CoordinateTransform, calculate_analysis_size


class InputRejectedError(ValueError):
    """A user-facing input validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class IngestedImage:
    canonical_rgb: np.ndarray
    summary: InputSummary
    transform: CoordinateTransform
    limitations: list[str]


_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
_PNG_EXTENSIONS = {".png"}
_WEBP_EXTENSIONS = {".webp"}


def detect_format(header: bytes) -> str | None:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    return None


def _extension_matches(suffix: str, detected_format: str) -> bool:
    suffix = suffix.lower()
    if detected_format == "jpeg":
        return suffix in _JPEG_EXTENSIONS
    if detected_format == "png":
        return suffix in _PNG_EXTENSIONS
    if detected_format == "webp":
        return suffix in _WEBP_EXTENSIONS
    return False


def _special_imaging_hints(raw: bytes) -> list[str]:
    lower = raw.lower()
    hints: list[str] = []
    if b"gpano:" in lower or b"projectiontype" in lower:
        hints.append("panorama_metadata_detected")
    if b"equirectangular" in lower:
        hints.append("equirectangular_projection_detected")
    if b"fisheye" in lower or b"fish-eye" in lower:
        hints.append("fisheye_metadata_detected")
    if b"photostitch" in lower or b"image_stitch" in lower:
        hints.append("stitched_image_metadata_detected")
    return hints


def ingest_image(
    path: Path,
    config: IngestConfig,
    max_long_side: int,
) -> IngestedImage:
    if not path.is_file():
        raise InputRejectedError("file_not_found", f"Input file does not exist: {path}")

    file_size = path.stat().st_size
    if file_size > config.max_file_bytes:
        raise InputRejectedError(
            "file_too_large",
            f"File is {file_size} bytes; limit is {config.max_file_bytes} bytes.",
        )

    raw = path.read_bytes()
    detected_format = detect_format(raw[:32])
    if detected_format is None:
        raise InputRejectedError(
            "unsupported_magic",
            "File magic is not PNG, JPEG, or static WebP.",
        )

    mismatch = not _extension_matches(path.suffix, detected_format)
    if mismatch and not config.allow_filename_format_mismatch:
        raise InputRejectedError(
            "filename_format_mismatch",
            "Filename extension does not match file magic.",
        )

    try:
        with Image.open(path) as verify_image:
            verify_image.verify()
        with Image.open(path) as source:
            frame_count = getattr(source, "n_frames", 1)
            if frame_count != 1 or getattr(source, "is_animated", False):
                raise InputRejectedError(
                    "animated_image_not_supported",
                    "Animated or multi-frame images are not supported in P0.",
                )
            encoded_size = source.size
            encoded_mode = source.mode
            encoded_pixel_count = encoded_size[0] * encoded_size[1]
            # Read dimensions from the container header before any operation
            # that can force a full decode or RGB allocation.  EXIF rotation
            # preserves the total pixel count, so this is also the canonical
            # image limit.
            if encoded_pixel_count > config.max_pixels:
                raise InputRejectedError(
                    "pixel_limit_exceeded",
                    f"Image has {encoded_pixel_count} pixels; limit is {config.max_pixels}.",
                )
            if min(encoded_size) < config.min_side_px:
                raise InputRejectedError(
                    "image_too_small",
                    f"Image side is below minimum {config.min_side_px} pixels.",
                )
            exif = source.getexif()
            exif_orientation = int(exif.get(274, 1))
            icc_profile = source.info.get("icc_profile")
            canonical = ImageOps.exif_transpose(source).convert("RGB")
    except InputRejectedError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputRejectedError("decode_failed", f"Unable to decode image: {exc}") from exc

    canonical_size = canonical.size
    pixel_count = canonical_size[0] * canonical_size[1]

    analysis_size = calculate_analysis_size(canonical_size, max_long_side)
    transform = CoordinateTransform(
        encoded_size=encoded_size,
        canonical_size=canonical_size,
        analysis_size=analysis_size,
        exif_orientation=exif_orientation,
        orientation_applied=exif_orientation != 1,
    )
    exif_summary = {
        "orientation": exif_orientation,
        "has_exif": bool(exif),
        "encoded_mode": encoded_mode,
        "has_icc_profile": bool(icc_profile),
        "icc_profile_bytes": len(icc_profile) if icc_profile else 0,
    }
    limitations = _special_imaging_hints(raw)
    if mismatch:
        limitations.append("filename_format_mismatch")

    summary = InputSummary(
        sha256=sha256(raw).hexdigest(),
        detected_format=detected_format,
        original_filename=path.name,
        filename_format_mismatch=mismatch,
        file_size_bytes=file_size,
        encoded_size=encoded_size,
        canonical_size=canonical_size,
        analysis_size=analysis_size,
        pixel_count=pixel_count,
        color_mode=encoded_mode,
        validation_limits={
            "max_file_bytes": config.max_file_bytes,
            "max_pixels": config.max_pixels,
            "min_side_px": config.min_side_px,
            "allow_filename_format_mismatch": config.allow_filename_format_mismatch,
        },
        coordinate_transform=transform.summary(),
        exif_summary=exif_summary,
    )
    return IngestedImage(
        canonical_rgb=np.asarray(canonical).copy(),
        summary=summary,
        transform=transform,
        limitations=limitations,
    )
