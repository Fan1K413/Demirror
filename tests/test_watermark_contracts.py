from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from image_trust.ai_likelihood.contracts import AiLikelihoodResult, AiSignal
from image_trust.origin import assess_origin
from image_trust.provenance.contracts import (
    C2paRecord,
    C2paRecordStatus,
    C2paSignatureValidationStatus,
    C2paTrustStatus,
)
from image_trust.watermark.contracts import (
    ImplicitWatermarkAssessment,
    WatermarkAdapterResult,
    WatermarkCoverage,
    WatermarkPayload,
    WatermarkScore,
)
from image_trust.watermark.suite import (
    assess_implicit_watermarks,
    build_offline_watermark_adapters,
)


COVERAGE = WatermarkCoverage(
    ecosystem=["test"],
    min_short_side=256,
    supported_formats=["jpeg", "png", "webp"],
)


def _adapter_result(**updates) -> WatermarkAdapterResult:
    values = {
        "adapter_id": "test_adapter_v1",
        "scheme": "test_scheme",
        "detector_version": "test-1",
        "run_status": "ok",
        "observation": "negative",
        "coverage": COVERAGE,
    }
    values.update(updates)
    return WatermarkAdapterResult(**values)


def _c2pa() -> C2paRecord:
    return C2paRecord(
        config_version="test",
        config_digest="0" * 64,
        original_filename="asset.png",
        status=C2paRecordStatus.NOT_OBSERVED,
        manifest_present=False,
        signature_validation_status=C2paSignatureValidationStatus.NOT_OBSERVED,
        trust_status=C2paTrustStatus.NOT_ASSESSED,
        trust_list_version="not_configured",
    )


def _ai() -> AiLikelihoodResult:
    return AiLikelihoodResult(
        status="available",
        target_definition="test",
        signals=[
            AiSignal(
                name="dda_pixel_detector",
                status="available",
                value=0.1,
                interpretation="test",
            )
        ],
    )


def test_negative_observation_is_forced_to_remain_neutral() -> None:
    with pytest.raises(ValidationError, match="negative watermark observation must remain neutral"):
        _adapter_result(direction="supports_ai", strength="limited")


def test_unverified_identifier_cannot_be_decision_eligible() -> None:
    with pytest.raises(ValidationError, match="unverified identifier cannot affect"):
        _adapter_result(
            observation="positive",
            evidence_class="unverified_identifier",
            direction="supports_ai",
            strength="limited",
            decision_eligible=True,
        )


def test_suite_keeps_unconfigured_and_failed_adapters_distinct(tmp_path: Path) -> None:
    asset = tmp_path / "asset.png"
    Image.new("RGB", (300, 300)).save(asset)

    assert assess_implicit_watermarks(asset).status == "not_configured"

    def broken(_: Path) -> WatermarkAdapterResult:
        raise RuntimeError("test failure")

    result = assess_implicit_watermarks(asset, [broken])
    assert result.status == "unavailable"
    assert result.adapters[0].run_status == "failed"
    assert result.adapters[0].observation == "not_observed"
    assert result.direction == "neutral"
    assert result.decision_eligible is False


def test_default_offline_suite_registers_both_independent_schemes() -> None:
    adapters = build_offline_watermark_adapters()

    assert [adapter.adapter_id for adapter in adapters] == [
        "sdxl_dwt_dct_v1",
        "trustmark_q_onnx_v1",
    ]
    assert len({adapter.scheme for adapter in adapters}) == 2


def test_suite_aggregates_only_explicitly_eligible_positive_results(tmp_path: Path) -> None:
    asset = tmp_path / "asset.png"
    Image.new("RGB", (300, 300)).save(asset)

    def positive(_: Path) -> WatermarkAdapterResult:
        return _adapter_result(
            observation="positive",
            evidence_class="known_open_ai_watermark",
            direction="supports_ai",
            strength="limited",
            decision_eligible=True,
            score=WatermarkScore(
                name="bit_match_rate",
                value=1.0,
                threshold=0.95,
                threshold_id="test-threshold",
            ),
            payload=WatermarkPayload(
                present=True,
                payload_schema="fixed_bits",
                sha256="a" * 64,
                bit_length=48,
            ),
        )

    result = assess_implicit_watermarks(asset, [positive])
    assert result.status == "completed"
    assert result.direction == "supports_ai"
    assert result.strength == "limited"
    assert result.decision_eligible is True


def test_eligible_open_watermark_can_only_produce_limited_possible_ai(tmp_path: Path) -> None:
    asset = tmp_path / "asset.png"
    Image.new("RGB", (300, 300)).save(asset)
    watermark = ImplicitWatermarkAssessment(
        status="completed",
        adapters=[
            _adapter_result(
                observation="positive",
                evidence_class="known_open_ai_watermark",
                direction="supports_ai",
                strength="limited",
                decision_eligible=True,
            )
        ],
        direction="supports_ai",
        strength="limited",
        decision_eligible=True,
    )

    result = assess_origin(asset, _ai(), _c2pa(), watermark_result=watermark)

    assert result.decision == "possible_ai"
    assert result.evidence_strength == "limited"
    assert result.supporting_evidence == ["已知开放 AI 生态水印（有限强度）"]


def test_unverified_identifier_does_not_change_origin_decision(tmp_path: Path) -> None:
    asset = tmp_path / "asset.png"
    Image.new("RGB", (300, 300)).save(asset)
    watermark = ImplicitWatermarkAssessment(
        status="completed",
        adapters=[
            _adapter_result(
                observation="positive",
                evidence_class="unverified_identifier",
            )
        ],
    )

    result = assess_origin(asset, _ai(), _c2pa(), watermark_result=watermark)

    assert result.decision == "no_ai_signal"
    assert result.implicit_watermark.adapters[0].observation == "positive"
