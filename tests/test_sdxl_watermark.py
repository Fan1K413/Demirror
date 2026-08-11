from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from image_trust.watermark import sdxl


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_sdxl_adapter_rejects_inputs_below_registered_size(tmp_path: Path) -> None:
    asset = tmp_path / "small.png"
    Image.new("RGB", (255, 400), "white").save(asset)

    result = sdxl.detect_sdxl_fixed_watermark(asset)

    assert result.run_status == "not_applicable"
    assert result.observation == "not_observed"
    assert result.direction == "neutral"
    assert "sdxl_watermark_short_side_below_256" in result.limitations


def test_sdxl_adapter_contains_worker_failure(tmp_path: Path, monkeypatch) -> None:
    asset = tmp_path / "asset.png"
    Image.new("RGB", (300, 300), "white").save(asset)

    def fail(*_args, **_kwargs):
        raise sdxl.SdxlWatermarkUnavailableError("sdxl_watermark_worker_timed_out")

    monkeypatch.setattr(sdxl, "_decode_isolated", fail)
    result = sdxl.detect_sdxl_fixed_watermark(asset)

    assert result.run_status == "failed"
    assert result.observation == "not_observed"
    assert result.decision_eligible is False
    assert result.errors[0]["code"] == "sdxl_watermark_worker_timed_out"


def test_sdxl_adapter_refuses_unpinned_pywavelets_version(tmp_path: Path, monkeypatch) -> None:
    asset = tmp_path / "asset.png"
    Image.new("RGB", (300, 300), "white").save(asset)
    monkeypatch.setattr(sdxl, "version", lambda _: "999.0")

    result = sdxl.detect_sdxl_fixed_watermark(asset)

    assert result.run_status == "unavailable"
    assert "sdxl_watermark_pywavelets_version_not_pinned" in result.limitations


def test_sdxl_fixed_message_round_trip_forms_limited_ai_evidence(tmp_path: Path) -> None:
    imwatermark = pytest.importorskip("imwatermark")
    rng = np.random.default_rng(20260811)
    image = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    encoder = imwatermark.WatermarkEncoder()
    encoder.set_watermark("bits", [int(bit) for bit in sdxl.SDXL_WATERMARK_BITS])
    encoded = encoder.encode(image, "dwtDct")
    reference = imwatermark.WatermarkDecoder("bits", 48).decode(encoded, "dwtDct")
    compatible = sdxl._decode_dwt_dct_bits(encoded, bit_length=48)
    assert compatible == "".join(str(int(bit)) for bit in reference.tolist())
    asset = tmp_path / "encoded.png"
    assert cv2.imwrite(str(asset), encoded)

    result = sdxl.detect_sdxl_fixed_watermark(asset)

    assert result.run_status == "ok"
    assert result.observation == "positive"
    assert result.evidence_class == "known_open_ai_watermark"
    assert result.strength == "limited"
    assert result.decision_eligible is True
    assert result.score is not None
    assert result.score.value == 1.0
    assert result.payload.present is True
    assert result.payload.bit_length == 48


def test_sdxl_negative_observation_never_supports_camera(tmp_path: Path) -> None:
    rng = np.random.default_rng(22)
    image = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    asset = tmp_path / "unmarked.png"
    assert cv2.imwrite(str(asset), image)

    result = sdxl.detect_sdxl_fixed_watermark(asset)

    assert result.run_status == "ok"
    assert result.observation == "negative"
    assert result.direction == "neutral"
    assert result.strength == "none"
    assert result.decision_eligible is False


def test_registered_sdxl_operating_point_matches_reproducible_audit() -> None:
    audit = json.loads(
        (REPOSITORY_ROOT / "models" / "implicit_watermark_sdxl_dwt_dct_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert audit["runtime_decoder"] == sdxl.SDXL_DECODER_VERSION
    assert audit["runtime_dependency"] == f"PyWavelets=={sdxl.PYWAVELETS_VERSION}"
    assert audit["operating_point"]["minimum_matched_bits"] == sdxl.SDXL_MIN_MATCHED_BITS
    assert audit["operating_point"]["threshold_id"] == sdxl.SDXL_THRESHOLD_ID
    assert audit["negative_screen"]["eligible_images"] >= 2000
    assert audit["negative_screen"]["false_positive_rate"] <= 0.001
    assert audit["negative_screen"]["zero_error_one_sided_95_upper"] <= 0.005
    assert audit["decision_policy"]["evidence_strength"] == "limited"
    assert audit["decision_policy"]["negative_contribution"] == "neutral"
