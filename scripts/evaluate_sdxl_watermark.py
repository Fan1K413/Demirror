"""Reproduce the local SD/SDXL fixed-watermark operating-point screen."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import time

import cv2
from imwatermark import WatermarkDecoder, WatermarkEncoder
import numpy as np
from PIL import Image, UnidentifiedImageError

from image_trust.watermark.sdxl import (
    INVISIBLE_WATERMARK_VERSION,
    PYWAVELETS_VERSION,
    SDXL_DECODER_VERSION,
    SDXL_MATCH_THRESHOLD,
    SDXL_MIN_MATCHED_BITS,
    SDXL_THRESHOLD_ID,
    SDXL_WATERMARK_BITS,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
POSITIVE_SOURCE_LIMIT = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_paths(paths: list[Path]) -> list[Path]:
    by_hash: dict[str, Path] = {}
    for path in sorted(paths):
        by_hash.setdefault(_sha256(path), path)
    return [by_hash[key] for key in sorted(by_hash)]


def _decode_matches(decoder: WatermarkDecoder, path: Path, expected: np.ndarray) -> int | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 256:
        return None
    decoded = np.asarray(decoder.decode(image, "dwtDct"), dtype=np.uint8).reshape(-1)
    return int(np.equal(decoded, expected).sum())


def _screen_paths(
    decoder: WatermarkDecoder,
    paths: list[Path],
    expected: np.ndarray,
) -> dict[str, object]:
    values: list[int] = []
    started = time.perf_counter()
    for path in paths:
        matched = _decode_matches(decoder, path, expected)
        if matched is not None:
            values.append(matched)
    positives = sum(value >= SDXL_MIN_MATCHED_BITS for value in values)
    return {
        "eligible_images": len(values),
        "positive_matches": positives,
        "positive_rate": positives / len(values) if values else None,
        "max_matched_bits": max(values) if values else None,
        "matched_bit_histogram": dict(sorted(Counter(values).items())),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _prepared_positive_source(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 512:
        return None
    height, width = image.shape[:2]
    if max(height, width) > 1024:
        scale = 1024 / max(height, width)
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image if min(image.shape[:2]) >= 256 else None


def _transform(name: str, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if name.startswith("jpeg"):
        quality = int(name.removeprefix("jpeg"))
        encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if name == "webp80":
        encoded = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, 80])[1]
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if name == "resize075":
        return cv2.resize(image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    if name == "crop10":
        return image[round(0.05 * height) : round(0.95 * height), round(0.05 * width) : round(0.95 * width)]
    if name == "screenshot125":
        return cv2.resize(image, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    return image


def _screen_self_encoded_positives(
    encoder: WatermarkEncoder,
    decoder: WatermarkDecoder,
    paths: list[Path],
    expected: np.ndarray,
) -> dict[str, object]:
    transforms = ["original", "jpeg95", "jpeg85", "webp80", "resize075", "crop10", "screenshot125"]
    values: dict[str, list[int]] = defaultdict(list)
    used = 0
    started = time.perf_counter()
    for path in paths:
        source = _prepared_positive_source(path)
        if source is None:
            continue
        marked = encoder.encode(source, "dwtDct")
        used += 1
        for name in transforms:
            transformed = _transform(name, marked)
            if min(transformed.shape[:2]) < 256:
                continue
            decoded = np.asarray(decoder.decode(transformed, "dwtDct"), dtype=np.uint8).reshape(-1)
            values[name].append(int(np.equal(decoded, expected).sum()))
        if used >= POSITIVE_SOURCE_LIMIT:
            break
    result: dict[str, object] = {}
    for name in transforms:
        matches = values[name]
        detected = sum(value >= SDXL_MIN_MATCHED_BITS for value in matches)
        result[name] = {
            "eligible_images": len(matches),
            "detected": detected,
            "recall": detected / len(matches) if matches else None,
            "min_matched_bits": min(matches) if matches else None,
            "mean_matched_bits": round(sum(matches) / len(matches), 4) if matches else None,
        }
    return {
        "source_images": used,
        "transforms": result,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _zero_error_one_sided_upper(n: int, alpha: float = 0.05) -> float | None:
    if n <= 0:
        return None
    return 1.0 - alpha ** (1.0 / n)


def evaluate(data_root: Path) -> dict[str, object]:
    all_images = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    real_paths = _unique_paths([path for path in all_images if path.parent.name == "real"])
    non_sdxl_generated = _unique_paths(
        [
            path
            for path in all_images
            if path.parent.name == "gen" and "sdxl" not in str(path).lower()
        ]
    )
    sdxl_generated = _unique_paths(
        [
            path
            for path in all_images
            if path.parent.name == "gen" and "sdxl" in str(path).lower()
        ]
    )
    high_resolution_sources: list[Path] = []
    for path in _unique_paths(all_images):
        try:
            with Image.open(path) as image:
                if min(image.size) >= 512:
                    high_resolution_sources.append(path)
        except (OSError, UnidentifiedImageError, ValueError):
            continue

    expected = np.asarray([int(bit) for bit in SDXL_WATERMARK_BITS], dtype=np.uint8)
    decoder = WatermarkDecoder("bits", len(expected))
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", expected.tolist())
    real_screen = _screen_paths(decoder, real_paths, expected)
    other_ai_screen = _screen_paths(decoder, non_sdxl_generated, expected)
    native_sdxl_screen = _screen_paths(decoder, sdxl_generated, expected)
    self_encoded = _screen_self_encoded_positives(
        encoder,
        decoder,
        high_resolution_sources,
        expected,
    )
    negative_n = int(real_screen["eligible_images"]) + int(other_ai_screen["eligible_images"])
    negative_matches = int(real_screen["positive_matches"]) + int(other_ai_screen["positive_matches"])
    return {
        "schema_version": "sdxl-watermark-audit-v1",
        "audit_date": date.today().isoformat(),
        "adapter_id": "sdxl_dwt_dct_v1",
        "runtime_decoder": SDXL_DECODER_VERSION,
        "runtime_dependency": f"PyWavelets=={PYWAVELETS_VERSION}",
        "audit_encoder": f"invisible-watermark=={INVISIBLE_WATERMARK_VERSION}",
        "method": "dwtDct",
        "expected_bits": SDXL_WATERMARK_BITS,
        "operating_point": {
            "minimum_matched_bits": SDXL_MIN_MATCHED_BITS,
            "bit_length": len(SDXL_WATERMARK_BITS),
            "match_rate_threshold": SDXL_MATCH_THRESHOLD,
            "threshold_id": SDXL_THRESHOLD_ID,
        },
        "negative_screen": {
            "definition": "unique camera-photo files plus unique non-SDXL generated files found in the local Projective Geometry corpora",
            "eligible_images": negative_n,
            "positive_matches": negative_matches,
            "false_positive_rate": negative_matches / negative_n if negative_n else None,
            "zero_error_one_sided_95_upper": (
                _zero_error_one_sided_upper(negative_n) if negative_matches == 0 else None
            ),
            "camera_photos": real_screen,
            "non_sdxl_generated": other_ai_screen,
        },
        "native_sdxl_observation": {
            "label_warning": "The dataset does not prove that the fixed watermark was enabled or retained; this is a coverage observation, not a positive-ground-truth recall set.",
            **native_sdxl_screen,
        },
        "self_encoded_positive_screen": self_encoded,
        "decision_policy": {
            "evidence_strength": "limited",
            "decision_eligible_on_positive": True,
            "negative_contribution": "neutral",
            "rationale": "A match adds a high-specificity but forgeable open-ecosystem signal. Low recall limits coverage but does not invert a positive match into camera provenance.",
        },
        "limitations": [
            "The encoder is public, so a positive match cannot identify the creator or prove provenance.",
            "The self-encoded screen shows severe recall loss after JPEG, WebP, resize, crop, and screenshot-like resampling.",
            "No-match results are always neutral.",
            "The local screen is reproducible but is not a future-generator or deployment-prevalence guarantee.",
        ],
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "models" / "implicit_watermark_sdxl_dwt_dct_v1.json",
    )
    args = parser.parse_args()
    result = evaluate(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
