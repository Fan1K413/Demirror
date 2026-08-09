from __future__ import annotations

from pathlib import Path

from PIL import Image

from image_trust.ai_likelihood.contracts import AiLikelihoodResult, AiSignal
from image_trust.origin import assess_origin, inspect_camera_metadata
from image_trust.provenance.contracts import (
    C2paRecord,
    C2paRecordStatus,
    C2paSignatureValidationStatus,
    C2paTrustStatus,
)


def _c2pa(
    *,
    signature: C2paSignatureValidationStatus = C2paSignatureValidationStatus.NOT_OBSERVED,
    source_types: list[str] | None = None,
    trust: C2paTrustStatus = C2paTrustStatus.NOT_ASSESSED,
    trust_list_version: str = "not_configured",
) -> C2paRecord:
    return C2paRecord(
        config_version="test",
        config_digest="0" * 64,
        original_filename="asset.jpg",
        status=C2paRecordStatus.PRESENT if source_types else C2paRecordStatus.NOT_OBSERVED,
        manifest_present=bool(source_types),
        declared_digital_source_types=source_types or [],
        signature_validation_status=signature,
        trust_status=trust,
        trust_list_version=trust_list_version,
    )


def _ai(risk_band: str = "unknown") -> AiLikelihoodResult:
    return AiLikelihoodResult(
        status="available",
        risk_band=risk_band,  # type: ignore[arg-type]
        reliability=0.85 if risk_band == "high" else 0.5,
        reliability_label="high" if risk_band == "high" else "limited",
        target_definition="test",
        signals=[
            AiSignal(
                name="dda_pixel_detector",
                status="available",
                value=0.945 if risk_band == "high" else 0.2,
                interpretation="test",
            )
        ],
    )


def test_coherent_camera_exif_supports_limited_camera_decision_after_ai_check() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")

    metadata = inspect_camera_metadata(source)
    result = assess_origin(source, _ai(), _c2pa())

    assert metadata.status == "coherent"
    assert {"曝光时间", "光圈", "焦距"}.issubset(metadata.physical_capture_fields)
    assert result.decision == "possible_camera"
    assert result.evidence_strength == "limited"
    assert "完整相机 EXIF（可复制或编辑）" in result.supporting_evidence
    assert "exif_metadata_can_be_copied_or_edited" in result.limitations


def test_coherent_exif_does_not_become_camera_evidence_when_ai_check_is_unavailable() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")
    unavailable = AiLikelihoodResult(
        status="unavailable",
        target_definition="test",
        limitations=["detector unavailable"],
    )

    result = assess_origin(source, unavailable, _c2pa())

    assert result.decision == "no_ai_signal"


def test_untrusted_capture_claim_is_not_camera_decision(tmp_path: Path) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)

    result = assess_origin(
        asset,
        _ai(),
        _c2pa(
            signature=C2paSignatureValidationStatus.VALID,
            source_types=["http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"],
        ),
    )

    assert result.decision == "no_ai_signal"
    assert result.verified_c2pa_capture_declaration is True
    assert result.trusted_c2pa_capture_declaration is False
    assert "c2pa_capture_declaration_not_trusted_for_camera_decision" in result.limitations


def test_trusted_full_uri_capture_claim_is_positive_camera_evidence(tmp_path: Path) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)

    result = assess_origin(
        asset,
        _ai(),
        _c2pa(
            signature=C2paSignatureValidationStatus.VALID,
            source_types=["http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"],
            trust=C2paTrustStatus.TRUSTED,
            trust_list_version="c2pa-tl-test-v1",
        ),
    )

    assert result.decision == "possible_camera"
    assert result.verified_c2pa_capture_declaration is True
    assert result.trusted_c2pa_capture_declaration is True


def test_no_ai_signal_does_not_become_camera_evidence_without_positive_capture_evidence(tmp_path: Path) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)

    result = assess_origin(asset, _ai(), _c2pa())

    assert result.decision == "no_ai_signal"
    assert result.camera_metadata.status == "not_observed"


def test_high_ai_signal_overrides_copyable_camera_metadata() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")

    result = assess_origin(source, _ai("high"), _c2pa())

    assert result.decision == "possible_ai"
    assert "高置信 AI 像素检测" in result.supporting_evidence


def test_below_threshold_available_signal_is_not_labeled_high_confidence() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")
    result = _ai("high").model_copy(
        update={
            "signals": [
                AiSignal(
                    name="dda_pixel_detector",
                    status="available",
                    value=0.2,
                    interpretation="test",
                    details={"high_confidence_threshold": 0.94},
                ),
                AiSignal(
                    name="safe_pixel_detector",
                    status="available",
                    value=0.95,
                    interpretation="test",
                    details={"high_confidence_threshold": 0.90},
                ),
            ]
        }
    )

    assessment = assess_origin(source, result, _c2pa())

    assert assessment.supporting_evidence == ["高置信无损纹理检测"]


def test_forensic_clip_high_signal_has_a_specific_evidence_label() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")
    result = _ai("high").model_copy(
        update={
            "signals": [
                AiSignal(
                    name="forensic_clip_detector",
                    status="available",
                    value=0.993,
                    interpretation="test",
                    details={"high_confidence_threshold": 0.9925177097320557},
                )
            ]
        }
    )

    assessment = assess_origin(source, result, _c2pa())

    assert assessment.supporting_evidence == ["高置信耐压缩像素检测"]


def test_forensic_clip_limited_signal_yields_limited_possible_ai() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")
    result = _ai().model_copy(
        update={
            "risk_band": "medium",
            "reliability": 0.65,
            "reliability_label": "limited",
            "signals": [
                AiSignal(
                    name="forensic_clip_detector",
                    status="available",
                    value=0.992,
                    interpretation="test",
                    details={
                        "high_confidence_threshold": 0.9925177097320557,
                        "limited_review_threshold": 0.9919478297233582,
                    },
                )
            ],
        }
    )

    assessment = assess_origin(source, result, _c2pa())

    assert assessment.decision == "possible_ai"
    assert assessment.evidence_strength == "limited"
    assert assessment.supporting_evidence == ["耐压缩像素检测（有限强度）"]


def _fixture(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "p0_f6_real_v2" / "images" / name
