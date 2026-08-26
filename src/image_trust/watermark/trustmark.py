"""Offline TrustMark Q decoding through Adobe's pinned ONNX model.

TrustMark embeds an arbitrary identifier.  A locally decoded identifier does
not identify its issuer or prove that the image is AI-generated, so positive
results deliberately remain neutral until a trusted provenance record binds
the payload to a source.  The raw payload is never returned.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, UnidentifiedImageError

from image_trust.watermark.contracts import (
    WatermarkAdapterResult,
    WatermarkCoverage,
    WatermarkPayload,
)
from image_trust.runtime_paths import runtime_weights_root


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRUSTMARK_Q_MODEL_PATH = runtime_weights_root(PROJECT_ROOT) / "trustmark" / "Q" / "decoder_Q.onnx"
TRUSTMARK_Q_MODEL_SHA256 = "ee3268f057c9dabef680e169302f5973d0589feea86189ed229a896cc3aa88df"
TRUSTMARK_Q_MODEL_RELEASE = "official-onnx-Q-2026-04"
TRUSTMARK_DECODER_VERSION = "demirror-trustmark-q-onnx-v1"
ONNXRUNTIME_VERSION = "1.23.2"
BCHLIB_VERSION = "2.1.3"
TRUSTMARK_WORKER_TIMEOUT_SECONDS = 20
TRUSTMARK_MIN_SHORT_SIDE = 150
TRUSTMARK_MAX_INPUT_PIXELS = 40_000_000
TRUSTMARK_MAX_ACCEPTED_CORRECTED_BITS = 3

TRUSTMARK_COVERAGE = WatermarkCoverage(
    ecosystem=["Adobe TrustMark Q watermarks retained in the image pixels"],
    min_short_side=TRUSTMARK_MIN_SHORT_SIDE,
    supported_formats=["jpeg", "png", "webp"],
)

_SCHEMAS = {
    0: ("BCH_SUPER", 40, 8),
    1: ("BCH_5", 61, 5),
    2: ("BCH_4", 68, 4),
    3: ("BCH_3", 75, 3),
}


class TrustMarkUnavailableError(RuntimeError):
    """The optional detector could not form an auditable observation."""


@dataclass(frozen=True)
class TrustMarkQAdapter:
    """Callable adapter registered by the local watermark suite."""

    model_path: Path = TRUSTMARK_Q_MODEL_PATH
    adapter_id: str = "trustmark_q_onnx_v1"
    scheme: str = "adobe_trustmark_q_100_bit"
    detector_version: str = TRUSTMARK_DECODER_VERSION
    coverage: WatermarkCoverage = field(default_factory=lambda: TRUSTMARK_COVERAGE)

    def __call__(self, input_path: Path) -> WatermarkAdapterResult:
        return detect_trustmark_q(input_path, model_path=self.model_path)


def detect_trustmark_q(
    input_path: Path,
    *,
    model_path: Path = TRUSTMARK_Q_MODEL_PATH,
    timeout_seconds: int = TRUSTMARK_WORKER_TIMEOUT_SECONDS,
) -> WatermarkAdapterResult:
    """Decode a TrustMark Q identifier in a bounded, network-free worker."""

    base = {
        "adapter_id": "trustmark_q_onnx_v1",
        "scheme": "adobe_trustmark_q_100_bit",
        "detector_version": TRUSTMARK_DECODER_VERSION,
        "coverage": TRUSTMARK_COVERAGE,
    }
    unavailable = _dependency_limitation()
    if unavailable is not None:
        return WatermarkAdapterResult(
            **base,
            run_status="unavailable",
            observation="not_observed",
            limitations=[unavailable],
        )
    if not model_path.is_file():
        return WatermarkAdapterResult(
            **base,
            run_status="unavailable",
            observation="not_observed",
            limitations=["trustmark_q_model_not_available"],
        )
    try:
        model_hash = _sha256(model_path)
    except OSError as error:
        return WatermarkAdapterResult(
            **base,
            run_status="unavailable",
            observation="not_observed",
            limitations=["trustmark_q_model_not_readable"],
            errors=[{"code": type(error).__name__, "message": str(error)}],
        )
    if model_hash != TRUSTMARK_Q_MODEL_SHA256:
        return WatermarkAdapterResult(
            **base,
            run_status="unavailable",
            observation="not_observed",
            limitations=["trustmark_q_model_sha256_mismatch"],
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
                    limitations=["trustmark_q_animated_image_not_supported"],
                )
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return WatermarkAdapterResult(
            **base,
            run_status="failed",
            observation="not_observed",
            limitations=["trustmark_q_input_decode_failed"],
            errors=[{"code": type(error).__name__, "message": str(error)}],
        )
    if min(width, height) < TRUSTMARK_MIN_SHORT_SIDE:
        return WatermarkAdapterResult(
            **base,
            run_status="not_applicable",
            observation="not_observed",
            limitations=["trustmark_q_short_side_below_150"],
        )
    if width * height > TRUSTMARK_MAX_INPUT_PIXELS:
        return WatermarkAdapterResult(
            **base,
            run_status="not_applicable",
            observation="not_observed",
            limitations=["trustmark_q_input_pixel_limit_exceeded"],
        )
    if image_format not in {"jpeg", "jpg", "png", "webp"}:
        return WatermarkAdapterResult(
            **base,
            run_status="not_applicable",
            observation="not_observed",
            limitations=["trustmark_q_image_format_not_supported"],
        )

    try:
        decoded = _decode_isolated(
            input_path,
            model_path=model_path,
            timeout_seconds=timeout_seconds,
        )
    except TrustMarkUnavailableError as error:
        code = str(error)
        return WatermarkAdapterResult(
            **base,
            run_status="failed",
            observation="not_observed",
            limitations=[code],
            errors=[{"code": code, "message": "TrustMark Q worker did not complete."}],
        )

    common_limitations = [
        "trustmark_q_variant_only",
        "trustmark_q_bch5_schema_only",
        "trustmark_q_correction_budget_reduced_to_3",
        "trustmark_identifier_is_not_ai_evidence_without_trusted_provenance",
        "trustmark_identifier_can_be_reencoded_or_removed",
        "trustmark_payload_withheld_from_result",
        "trustmark_remote_resolver_disabled",
        "trustmark_negative_is_not_camera_evidence",
    ]
    if not decoded["detected"]:
        return WatermarkAdapterResult(
            **base,
            run_status="ok",
            observation="negative",
            limitations=common_limitations,
        )

    payload_bits = decoded["payload_bits"]
    return WatermarkAdapterResult(
        **base,
        run_status="ok",
        observation="positive",
        evidence_class="unverified_identifier",
        payload=WatermarkPayload(
            present=True,
            payload_schema=decoded["schema"],
            sha256=hashlib.sha256(payload_bits.encode("ascii")).hexdigest(),
            bit_length=len(payload_bits),
        ),
        limitations=common_limitations,
    )


def _dependency_limitation() -> str | None:
    requirements = {
        "onnxruntime": ONNXRUNTIME_VERSION,
        "bchlib": BCHLIB_VERSION,
    }
    for package_name, expected in requirements.items():
        try:
            installed = version(package_name)
        except PackageNotFoundError:
            return f"trustmark_q_dependency_not_available:{package_name}"
        if installed != expected:
            return f"trustmark_q_dependency_version_not_pinned:{package_name}"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_isolated(
    input_path: Path,
    *,
    model_path: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="demirror-trustmark-q-") as temporary_directory:
        output_path = Path(temporary_directory) / "result.json"
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "ORT_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "image_trust.watermark.trustmark",
            "--worker",
            "--input",
            str(input_path),
            "--model",
            str(model_path),
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
            raise TrustMarkUnavailableError("trustmark_q_worker_timed_out") from error
        if completed.returncode != 0 or not output_path.is_file():
            raise TrustMarkUnavailableError("trustmark_q_worker_failed")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            detected = bool(raw["detected"])
            payload_bits = str(raw.get("payload_bits", ""))
            schema = raw.get("schema")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise TrustMarkUnavailableError("trustmark_q_worker_result_invalid") from error
    if detected:
        if schema not in {item[0] for item in _SCHEMAS.values()}:
            raise TrustMarkUnavailableError("trustmark_q_worker_schema_invalid")
        expected_length = next(item[1] for item in _SCHEMAS.values() if item[0] == schema)
        if len(payload_bits) != expected_length or set(payload_bits) - {"0", "1"}:
            raise TrustMarkUnavailableError("trustmark_q_worker_payload_invalid")
    elif payload_bits or schema is not None:
        raise TrustMarkUnavailableError("trustmark_q_worker_negative_result_invalid")
    return {"detected": detected, "payload_bits": payload_bits, "schema": schema}


def _preprocess_image(input_path: Path):
    """Mirror the official TrustMark Q ONNX decoder preprocessing."""

    import numpy as np

    with Image.open(input_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if max(width / height, height / width) > 2.0:
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def _decode_ecc(bits: str) -> dict[str, object]:
    """Validate the default BCH_5 packet without permissive schema fallback.

    TrustMark's weaker BCH_4/BCH_3 modes make a random neural bitstream pass
    ECC too often for a general-purpose forensic screen.  Demirror therefore
    accepts only the documented default BCH_5 schema and all four version bits.
    """

    import bchlib

    if len(bits) != 100 or set(bits) - {"0", "1"} or bits[96:] != "0001":
        return {"detected": False, "payload_bits": "", "schema": None, "corrected_bits": None}
    schema, data_length, strength = _SCHEMAS[1]
    data_bits = bits[:data_length]
    ecc_bits = bits[data_length:96]
    padded_data = data_bits + "0" * (-len(data_bits) % 8)
    padded_ecc = ecc_bits + "0" * (-len(ecc_bits) % 8)
    data = bytearray(_bits_to_bytes(padded_data))
    ecc = bytearray(_bits_to_bytes(padded_ecc))
    decoder = bchlib.BCH(strength, 137)
    if len(ecc) != decoder.ecc_bytes:
        return {"detected": False, "payload_bits": "", "schema": None, "corrected_bits": None}
    corrected_bits = decoder.decode(data, ecc)
    if corrected_bits < 0 or corrected_bits > TRUSTMARK_MAX_ACCEPTED_CORRECTED_BITS:
        return {"detected": False, "payload_bits": "", "schema": None, "corrected_bits": None}
    decoder.correct(data, ecc)
    corrected_payload = "".join(f"{byte:08b}" for byte in data)[:data_length]
    return {
        "detected": True,
        "payload_bits": corrected_payload,
        "schema": schema,
        "corrected_bits": corrected_bits,
    }


def _bits_to_bytes(bits: str) -> bytes:
    return bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))


def _worker(input_path: Path, model_path: Path, output_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    image = _preprocess_image(input_path)
    output = session.run(None, {session.get_inputs()[0].name: image})[0]
    logits = np.asarray(output).reshape(-1)
    if logits.size != 100 or not np.isfinite(logits).all():
        raise ValueError("trustmark_q_worker_logits_invalid")
    bits = "".join("1" if value >= 0 else "0" for value in logits)
    decoded = _decode_ecc(bits)
    output_path.write_text(json.dumps(decoded), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker or args.input is None or args.model is None or args.output is None:
        parser.error("worker mode requires --input, --model and --output")
    _worker(args.input, args.model, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in an isolated process
    raise SystemExit(main())
