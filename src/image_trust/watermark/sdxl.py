"""Offline detector for the public fixed SD/SDXL DWT-DCT watermark.

The compact decoder is compatible with the MIT-licensed ShieldMnt
``invisible-watermark`` 0.2.0 DWT-DCT implementation.  Keeping only the
frequency decoder avoids importing its optional RivaGAN/PyTorch path in every
short-lived worker.  See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from PIL import Image, UnidentifiedImageError

from image_trust.watermark.contracts import (
    WatermarkAdapterResult,
    WatermarkCoverage,
    WatermarkPayload,
    WatermarkScore,
)


SDXL_WATERMARK_BITS = "101100111110110010010000011110111011000110011110"
SDXL_WATERMARK_BIT_LENGTH = len(SDXL_WATERMARK_BITS)
SDXL_MIN_MATCHED_BITS = 45
SDXL_MATCH_THRESHOLD = SDXL_MIN_MATCHED_BITS / SDXL_WATERMARK_BIT_LENGTH
SDXL_THRESHOLD_ID = "sdxl-dwt-dct-local-screen-v1-45-of-48"
INVISIBLE_WATERMARK_VERSION = "0.2.0"
PYWAVELETS_VERSION = "1.8.0"
SDXL_DECODER_VERSION = "demirror-dwt-dct-decoder-v1"
SDXL_WORKER_TIMEOUT_SECONDS = 15
SDXL_MAX_INPUT_PIXELS = 40_000_000

SDXL_COVERAGE = WatermarkCoverage(
    ecosystem=["Stable Diffusion / SDXL export paths that retained the public fixed mark"],
    min_short_side=256,
    supported_formats=["jpeg", "png", "webp"],
)


class SdxlWatermarkUnavailableError(RuntimeError):
    """The optional decoder could not form an auditable observation."""


@dataclass(frozen=True)
class SdxlDwtDctAdapter:
    """Callable adapter registered by the local watermark suite."""

    decision_eligible: bool = True
    adapter_id: str = "sdxl_dwt_dct_v1"
    scheme: str = "stable_diffusion_dwt_dct_fixed_48"
    detector_version: str = SDXL_DECODER_VERSION
    coverage: WatermarkCoverage = field(default_factory=lambda: SDXL_COVERAGE)

    def __call__(self, input_path: Path) -> WatermarkAdapterResult:
        return detect_sdxl_fixed_watermark(
            input_path,
            decision_eligible=self.decision_eligible,
        )


def detect_sdxl_fixed_watermark(
    input_path: Path,
    *,
    decision_eligible: bool = True,
    timeout_seconds: int = SDXL_WORKER_TIMEOUT_SECONDS,
) -> WatermarkAdapterResult:
    """Decode the known public message in a bounded, short-lived process."""

    base = {
        "adapter_id": "sdxl_dwt_dct_v1",
        "scheme": "stable_diffusion_dwt_dct_fixed_48",
        "detector_version": SDXL_DECODER_VERSION,
        "coverage": SDXL_COVERAGE,
    }
    try:
        installed_version = version("PyWavelets")
    except PackageNotFoundError:
        return WatermarkAdapterResult(
            **base,
            run_status="unavailable",
            observation="not_observed",
            limitations=["sdxl_watermark_pywavelets_not_available"],
        )
    if installed_version != PYWAVELETS_VERSION:
        return WatermarkAdapterResult(
            **base,
            run_status="unavailable",
            observation="not_observed",
            limitations=["sdxl_watermark_pywavelets_version_not_pinned"],
        )
    try:
        with Image.open(input_path) as image:
            width, height = image.size
            image_format = (image.format or input_path.suffix.lstrip(".")).lower()
            if getattr(image, "is_animated", False):
                return WatermarkAdapterResult(
                    **base,
                    run_status="not_applicable",
                    observation="not_observed",
                    limitations=["sdxl_watermark_animated_image_not_supported"],
                )
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return WatermarkAdapterResult(
            **base,
            run_status="failed",
            observation="not_observed",
            limitations=["sdxl_watermark_input_decode_failed"],
            errors=[{"code": type(error).__name__, "message": str(error)}],
        )
    if min(width, height) < 256:
        return WatermarkAdapterResult(
            **base,
            run_status="not_applicable",
            observation="not_observed",
            limitations=["sdxl_watermark_short_side_below_256"],
        )
    if width * height > SDXL_MAX_INPUT_PIXELS:
        return WatermarkAdapterResult(
            **base,
            run_status="not_applicable",
            observation="not_observed",
            limitations=["sdxl_watermark_input_pixel_limit_exceeded"],
        )
    if image_format not in {"jpeg", "jpg", "png", "webp"}:
        return WatermarkAdapterResult(
            **base,
            run_status="not_applicable",
            observation="not_observed",
            limitations=["sdxl_watermark_image_format_not_supported"],
        )

    try:
        decoded_bits = _decode_isolated(input_path, timeout_seconds=timeout_seconds)
    except SdxlWatermarkUnavailableError as error:
        code = str(error)
        return WatermarkAdapterResult(
            **base,
            run_status="failed",
            observation="not_observed",
            limitations=[code],
            errors=[{"code": code, "message": "SD/SDXL watermark worker did not complete."}],
        )

    matched_bits = sum(
        observed == expected
        for observed, expected in zip(decoded_bits, SDXL_WATERMARK_BITS, strict=True)
    )
    match_rate = matched_bits / SDXL_WATERMARK_BIT_LENGTH
    if not math.isfinite(match_rate):
        raise AssertionError("finite bit match rate expected")
    score = WatermarkScore(
        name="bit_match_rate",
        value=match_rate,
        threshold=SDXL_MATCH_THRESHOLD,
        threshold_id=SDXL_THRESHOLD_ID,
    )
    common_limitations = [
        "sdxl_watermark_is_open_and_can_be_copied_to_non_ai_images",
        "sdxl_watermark_negative_is_not_camera_evidence",
        "sdxl_watermark_coverage_depends_on_generator_export_path",
        "sdxl_watermark_resizing_cropping_or_reencoding_can_destroy_signal",
    ]
    if matched_bits < SDXL_MIN_MATCHED_BITS:
        return WatermarkAdapterResult(
            **base,
            run_status="ok",
            observation="negative",
            score=score,
            limitations=common_limitations,
        )

    payload_hash = hashlib.sha256(decoded_bits.encode("ascii")).hexdigest()
    return WatermarkAdapterResult(
        **base,
        run_status="ok",
        observation="positive",
        evidence_class="known_open_ai_watermark",
        direction="supports_ai",
        strength="limited",
        decision_eligible=decision_eligible,
        score=score,
        payload=WatermarkPayload(
            present=True,
            payload_schema="fixed_public_bits",
            sha256=payload_hash,
            bit_length=SDXL_WATERMARK_BIT_LENGTH,
        ),
        limitations=common_limitations,
    )


def _decode_isolated(input_path: Path, *, timeout_seconds: int) -> str:
    with tempfile.TemporaryDirectory(prefix="demirror-sdxl-watermark-") as temporary_directory:
        output_path = Path(temporary_directory) / "result.json"
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "image_trust.watermark.sdxl",
            "--worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            raise SdxlWatermarkUnavailableError("sdxl_watermark_worker_timed_out") from error
        if completed.returncode != 0 or not output_path.is_file():
            raise SdxlWatermarkUnavailableError("sdxl_watermark_worker_failed")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            bits = str(raw["decoded_bits"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SdxlWatermarkUnavailableError("sdxl_watermark_worker_result_invalid") from error
    if len(bits) != SDXL_WATERMARK_BIT_LENGTH or set(bits) - {"0", "1"}:
        raise SdxlWatermarkUnavailableError("sdxl_watermark_worker_bits_invalid")
    return bits


def _worker(input_path: Path, output_path: Path) -> None:
    import cv2

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("sdxl_watermark_worker_input_decode_failed")
    bits = _decode_dwt_dct_bits(image, bit_length=SDXL_WATERMARK_BIT_LENGTH)
    output_path.write_text(json.dumps({"decoded_bits": bits}), encoding="utf-8")


def _decode_dwt_dct_bits(image, *, bit_length: int) -> str:
    """Decode the public U-channel Haar/DCT scheme without loading PyTorch."""

    import cv2
    import numpy as np
    import pywt

    rows, columns, _channels = image.shape
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    approximation, _details = pywt.dwt2(
        yuv[: rows // 4 * 4, : columns // 4 * 4, 1],
        "haar",
    )
    scores: list[list[int]] = [[] for _ in range(bit_length)]
    block_size = 4
    scale = 36
    block_index = 0
    for row_index in range(approximation.shape[0] // block_size):
        for column_index in range(approximation.shape[1] // block_size):
            block = approximation[
                row_index * block_size : (row_index + 1) * block_size,
                column_index * block_size : (column_index + 1) * block_size,
            ]
            flattened = block.reshape(-1)
            coefficient_index = int(np.argmax(np.abs(flattened[1:]))) + 1
            coefficient = abs(float(flattened[coefficient_index]))
            scores[block_index % bit_length].append(
                int((coefficient % scale) > (scale / 2))
            )
            block_index += 1
    if not all(scores):
        raise ValueError("sdxl_watermark_insufficient_frequency_blocks")
    decoded = [int(np.mean(bit_scores) > 0.5) for bit_scores in scores]
    return "".join(str(bit) for bit in decoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker or args.input is None or args.output is None:
        parser.error("worker mode requires --input and --output")
    _worker(args.input, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in an isolated process
    raise SystemExit(main())
