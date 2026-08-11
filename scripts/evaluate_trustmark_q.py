"""Reproduce Demirror's offline TrustMark Q/BCH_5 false-positive screen."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from image_trust.watermark.trustmark import (
    BCHLIB_VERSION,
    ONNXRUNTIME_VERSION,
    TRUSTMARK_DECODER_VERSION,
    TRUSTMARK_MIN_SHORT_SIDE,
    TRUSTMARK_MAX_ACCEPTED_CORRECTED_BITS,
    TRUSTMARK_Q_MODEL_RELEASE,
    TRUSTMARK_Q_MODEL_SHA256,
    _decode_ecc,
    _preprocess_image,
    _sha256,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _unique_paths(paths: list[Path]) -> list[Path]:
    by_hash: dict[str, Path] = {}
    for path in sorted(paths):
        by_hash.setdefault(_sha256(path), path)
    return [by_hash[key] for key in sorted(by_hash)]


def _session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _decode(session: ort.InferenceSession, path: Path) -> dict[str, object] | None:
    try:
        with Image.open(path) as image:
            if min(image.size) < TRUSTMARK_MIN_SHORT_SIDE:
                return None
        input_array = _preprocess_image(path)
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    output = session.run(None, {session.get_inputs()[0].name: input_array})[0]
    logits = np.asarray(output).reshape(-1)
    if logits.size != 100 or not np.isfinite(logits).all():
        return None
    bits = "".join("1" if value >= 0 else "0" for value in logits)
    return _decode_ecc(bits)


def _screen(
    session: ort.InferenceSession,
    paths: list[Path],
    *,
    label: str,
) -> dict[str, object]:
    eligible = 0
    positives = 0
    schemas: dict[str, int] = {}
    positive_examples: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        result = _decode(session, path)
        if result is not None:
            eligible += 1
            if result["detected"]:
                positives += 1
                schema = str(result["schema"])
                schemas[schema] = schemas.get(schema, 0) + 1
                positive_examples.append(
                    {
                        "sha256": _sha256(path),
                        "schema": schema,
                        "corrected_bits": result["corrected_bits"],
                    }
                )
        if index % 250 == 0 or index == len(paths):
            print(f"{label}: {index}/{len(paths)} files, {positives} matches", flush=True)
    return {
        "eligible_images": eligible,
        "positive_matches": positives,
        "positive_rate": positives / eligible if eligible else None,
        "matched_schemas": schemas,
        "positive_examples": positive_examples,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _positive_transforms(
    session: ort.InferenceSession,
    positive_path: Path,
) -> dict[str, dict[str, object]]:
    with Image.open(positive_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    variants: dict[str, tuple[Image.Image, str, dict[str, object]]] = {
        "original": (image, "PNG", {}),
        "jpeg95": (image, "JPEG", {"quality": 95}),
        "jpeg85": (image, "JPEG", {"quality": 85}),
        "webp80": (image, "WEBP", {"quality": 80}),
        "resize075": (
            image.resize((round(width * 0.75), round(height * 0.75)), Image.Resampling.LANCZOS),
            "PNG",
            {},
        ),
        "crop10": (
            image.crop((round(width * 0.05), round(height * 0.05), round(width * 0.95), round(height * 0.95))),
            "PNG",
            {},
        ),
        "screenshot125": (
            image.resize((round(width * 1.25), round(height * 1.25)), Image.Resampling.BICUBIC),
            "PNG",
            {},
        ),
    }
    results: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="demirror-trustmark-audit-") as temporary_directory:
        for name, (variant, output_format, options) in variants.items():
            suffix = ".jpg" if output_format == "JPEG" else ".webp" if output_format == "WEBP" else ".png"
            path = Path(temporary_directory) / f"{name}{suffix}"
            variant.save(path, format=output_format, **options)
            decoded = _decode(session, path)
            results[name] = {
                "detected": bool(decoded and decoded["detected"]),
                "schema": decoded["schema"] if decoded else None,
                "corrected_bits": decoded["corrected_bits"] if decoded else None,
            }
    return results


def _one_sided_upper(errors: int, n: int, alpha: float = 0.05) -> float | None:
    if n <= 0 or errors != 0:
        return None
    return 1.0 - alpha ** (1.0 / n)


def evaluate(data_root: Path, model_path: Path, fixture_root: Path) -> dict[str, object]:
    if _sha256(model_path) != TRUSTMARK_Q_MODEL_SHA256:
        raise ValueError("TrustMark Q model SHA-256 mismatch")
    all_images = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    real_paths = _unique_paths([path for path in all_images if path.parent.name == "real"])
    generated_paths = _unique_paths(
        [
            path
            for path in all_images
            if path.parent.name == "gen" and "sdxl" not in str(path).lower()
        ]
    )
    session = _session(model_path)
    positive_path = fixture_root / "ufo_240_Q.png"
    negative_path = fixture_root / "ufo_240.jpg"
    official_positive = _decode(session, positive_path)
    official_negative = _decode(session, negative_path)
    real = _screen(session, real_paths, label="camera photos")
    generated = _screen(session, generated_paths, label="non-TrustMark generated")
    n = int(real["eligible_images"]) + int(generated["eligible_images"])
    errors = int(real["positive_matches"]) + int(generated["positive_matches"])
    return {
        "schema_version": "trustmark-q-watermark-audit-v1",
        "audit_date": date.today().isoformat(),
        "adapter_id": "trustmark_q_onnx_v1",
        "runtime_decoder": TRUSTMARK_DECODER_VERSION,
        "runtime_dependencies": [
            f"onnxruntime=={ONNXRUNTIME_VERSION}",
            f"bchlib=={BCHLIB_VERSION}",
        ],
        "model": {
            "release": TRUSTMARK_Q_MODEL_RELEASE,
            "variant": "Q",
            "sha256": TRUSTMARK_Q_MODEL_SHA256,
        },
        "acceptance_rule": {
            "neural_bit_threshold": "logit >= 0",
            "accepted_schema": "BCH_5",
            "required_version_bits": "0001",
            "native_bch_correction_capacity": 5,
            "maximum_accepted_corrected_bits": TRUSTMARK_MAX_ACCEPTED_CORRECTED_BITS,
            "permissive_schema_fallback": False,
        },
        "official_fixture_screen": {
            "positive_detected": bool(official_positive and official_positive["detected"]),
            "positive_schema": official_positive["schema"] if official_positive else None,
            "negative_detected": bool(official_negative and official_negative["detected"]),
            "positive_sha256": _sha256(positive_path),
            "negative_sha256": _sha256(negative_path),
            "transforms": _positive_transforms(session, positive_path),
        },
        "negative_screen": {
            "definition": "unique camera-photo files plus unique non-SDXL generated files in the local Projective Geometry corpora",
            "eligible_images": n,
            "positive_matches": errors,
            "false_positive_rate": errors / n if n else None,
            "zero_error_one_sided_95_upper": _one_sided_upper(errors, n),
            "camera_photos": real,
            "non_trustmark_generated": generated,
        },
        "decision_policy": {
            "evidence_class": "unverified_identifier",
            "direction": "neutral",
            "decision_eligible": False,
            "raw_payload_returned": False,
        },
        "limitations": [
            "Only the Q model and default BCH_5 schema are accepted.",
            "A valid local identifier does not identify its issuer or prove AI generation.",
            "Remote identifier resolution is disabled.",
            "No-match results are always neutral.",
            "The local screen is reproducible but is not a deployment-prevalence guarantee.",
        ],
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "weights" / "trustmark" / "Q" / "decoder_Q.onnx",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=project_root / "tests" / "fixtures" / "trustmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "models" / "implicit_watermark_trustmark_q_v1.json",
    )
    args = parser.parse_args()
    result = evaluate(args.data_root.resolve(), args.model.resolve(), args.fixture_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
