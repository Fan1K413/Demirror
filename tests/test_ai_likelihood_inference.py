from __future__ import annotations

from PIL import Image
import pytest

from image_trust.ai_likelihood.dda import DdaScore, assess_high_confidence_ai
from image_trust.ai_likelihood.community_forensics import CommunityForensicsScore
from image_trust.ai_likelihood.forensic_clip import ForensicClipScore
from image_trust.ai_likelihood.safe import SafeScore
from image_trust.ai_likelihood.nonescape import NonescapeMiniScore
from image_trust.provenance.contracts import (
    C2paRecord,
    C2paRecordStatus,
    C2paSignatureValidationStatus,
    C2paTrustStatus,
)


@pytest.fixture(autouse=True)
def _default_nonescape_score(monkeypatch) -> None:
    """Keep unrelated policy tests free of a real child-model invocation."""

    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_nonescape_mini_isolated",
        lambda *_args, **_kwargs: NonescapeMiniScore(
            0.1,
            "resize_256_center_crop_224_jpeg100_imagenet_normalization",
        ),
    )


def _record(
    *,
    signature: C2paSignatureValidationStatus = C2paSignatureValidationStatus.NOT_OBSERVED,
    source_types: list[str] | None = None,
) -> C2paRecord:
    return C2paRecord(
        config_version="test",
        config_digest="0" * 64,
        original_filename="asset.jpg",
        status=C2paRecordStatus.PRESENT if source_types else C2paRecordStatus.NOT_OBSERVED,
        manifest_present=bool(source_types),
        declared_digital_source_types=source_types or [],
        signature_validation_status=signature,
        trust_status=C2paTrustStatus.NOT_ASSESSED,
        trust_list_version="not_configured",
    )


def test_dda_high_signal_is_conservative_and_auditable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.945, "center_crop_336_clip_normalization"),
    )

    result = assess_high_confidence_ai(tmp_path / "asset.png", _record())

    assert result.status == "available"
    assert result.probability is None
    assert result.risk_band == "high"
    assert result.decision_threshold == 0.94
    detector = next(signal for signal in result.signals if signal.name == "dda_pixel_detector")
    assert detector.value == 0.945
    assert detector.details["high_confidence_threshold"] == 0.94


def test_dda_score_below_high_threshold_is_not_camera_evidence(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.2, "center_crop_336_clip_normalization"),
    )

    result = assess_high_confidence_ai(tmp_path / "asset.png", _record())

    assert result.status == "available"
    assert result.risk_band == "unknown"
    assert "dda_no_high_confidence_pixel_signal_is_not_camera_evidence" in result.limitations


def test_safe_high_signal_can_complement_a_low_dda_score(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.2, "center_crop_336_clip_normalization"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_safe_isolated",
        lambda *_args, **_kwargs: SafeScore(0.95, "center_crop_256_rgb_to_tensor_dwt_bior1.3"),
    )

    result = assess_high_confidence_ai(tmp_path / "asset.png", _record())

    assert result.risk_band == "high"
    detector = next(signal for signal in result.signals if signal.name == "safe_pixel_detector")
    assert detector.value == 0.95
    assert detector.details["high_confidence_threshold"] == 0.9


def test_compression_stable_signal_can_complement_other_low_scores(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.2, "center_crop_336_clip_normalization"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_safe_isolated",
        lambda *_args, **_kwargs: SafeScore(0.1, "center_crop_256_rgb_to_tensor_dwt_bior1.3"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_forensic_clip_isolated",
        lambda *_args, **_kwargs: ForensicClipScore(
            0.993,
            "resize_256x256_clip_normalization_temperature_0.594889",
        ),
    )

    result = assess_high_confidence_ai(tmp_path / "asset.png", _record())

    assert result.risk_band == "high"
    detector = next(signal for signal in result.signals if signal.name == "forensic_clip_detector")
    assert detector.value == 0.993
    assert detector.details["high_confidence_threshold"] == 0.9925177097320557


def test_recall_oriented_forensic_threshold_emits_only_limited_ai_signal(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.2, "center_crop_336_clip_normalization"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_safe_isolated",
        lambda *_args, **_kwargs: SafeScore(0.1, "center_crop_256_rgb_to_tensor_dwt_bior1.3"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_forensic_clip_isolated",
        lambda *_args, **_kwargs: ForensicClipScore(
            0.992,
            "resize_256x256_clip_normalization_temperature_0.594889",
        ),
    )

    result = assess_high_confidence_ai(tmp_path / "asset.png", _record())

    assert result.risk_band == "medium"
    assert result.reliability_label == "limited"
    detector = next(signal for signal in result.signals if signal.name == "forensic_clip_detector")
    assert detector.details["limited_review_threshold"] == 0.9919478297233582


def test_community_forensics_high_signal_is_auditable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.2, "center_crop_336_clip_normalization"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_community_forensics_isolated",
        lambda *_args, **_kwargs: CommunityForensicsScore(
            0.9,
            "resize_shorter_256_center_crop_224_imagenet_normalization",
        ),
    )

    result = assess_high_confidence_ai(tmp_path / "asset.png", _record())

    assert result.risk_band == "high"
    detector = next(signal for signal in result.signals if signal.name == "community_forensics_detector")
    assert detector.value == 0.9
    assert detector.details["high_confidence_threshold"] == 0.8866265416145325


def test_nonescape_high_signal_can_complement_other_low_scores(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.2, "center_crop_336_clip_normalization"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_nonescape_mini_isolated",
        lambda *_args, **_kwargs: NonescapeMiniScore(
            0.94,
            "resize_256_center_crop_224_jpeg100_imagenet_normalization",
        ),
    )

    result = assess_high_confidence_ai(tmp_path / "asset.png", _record())

    assert result.risk_band == "high"
    detector = next(signal for signal in result.signals if signal.name == "nonescape_mini_detector")
    assert detector.value == 0.94
    assert detector.details["high_confidence_threshold"] == 0.9260923266410828


def test_static_webp_pixel_high_signal_is_limited_review_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.97, "center_crop_336_clip_normalization"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_community_forensics_isolated",
        lambda *_args, **_kwargs: CommunityForensicsScore(
            0.9,
            "resize_shorter_256_center_crop_224_imagenet_normalization",
        ),
    )
    input_path = tmp_path / "renamed-input.bin"
    Image.new("RGB", (8, 8), "white").save(input_path, format="WEBP", quality=80)

    result = assess_high_confidence_ai(input_path, _record())

    assert result.risk_band == "medium"
    assert result.reliability_label == "limited"
    assert "static_webp_pixel_high_scores_are_limited_review_only_without_format_calibration" in result.limitations
    detector = next(signal for signal in result.signals if signal.name == "community_forensics_detector")
    assert detector.details["high_confidence_eligible"] is False


def test_community_forensics_limited_signal_is_not_camera_evidence(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_dda_isolated",
        lambda *_args, **_kwargs: DdaScore(0.2, "center_crop_336_clip_normalization"),
    )
    monkeypatch.setattr(
        "image_trust.ai_likelihood.dda.score_community_forensics_isolated",
        lambda *_args, **_kwargs: CommunityForensicsScore(
            0.6,
            "resize_shorter_256_center_crop_224_imagenet_normalization",
        ),
    )

    result = assess_high_confidence_ai(tmp_path / "asset.png", _record())

    assert result.risk_band == "medium"
    assert "no_high_confidence_pixel_signal_is_not_camera_evidence" in result.limitations
    detector = next(signal for signal in result.signals if signal.name == "community_forensics_detector")
    assert detector.details["limited_review_threshold"] == 0.5


def test_verified_ai_c2pa_declaration_has_an_explicit_provenance_path(tmp_path) -> None:
    result = assess_high_confidence_ai(
        tmp_path / "asset.jpg",
        _record(
            signature=C2paSignatureValidationStatus.VALID,
            source_types=["trainedAlgorithmicMedia"],
        ),
    )

    assert result.status == "available"
    assert result.probability == 0.995
    assert result.model_version == "verified-c2pa-policy-v1"


def test_verified_ai_c2pa_uri_is_normalized_before_policy_evaluation(tmp_path) -> None:
    result = assess_high_confidence_ai(
        tmp_path / "asset.jpg",
        _record(
            signature=C2paSignatureValidationStatus.VALID,
            source_types=["http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"],
        ),
    )

    assert result.risk_band == "high"
    assert result.model_version == "verified-c2pa-policy-v1"
