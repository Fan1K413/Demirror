from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from image_trust.ai_likelihood.contracts import AiLikelihoodResult, AiSignal
from image_trust.geometry_ai.contracts import GeometryRelationshipResult
from image_trust.origin import (
    ORIGIN_SCORE_POLICY,
    SCORE_POLICY_VERSION,
    _score_summary,
    assess_origin,
    inspect_camera_metadata,
)
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


def test_coherent_camera_exif_yields_a_negative_score_and_possible_photo() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")

    metadata = inspect_camera_metadata(source)
    result = assess_origin(source, _ai(), _c2pa())

    assert metadata.status == "coherent"
    assert {"曝光时间", "光圈", "焦距"}.issubset(metadata.physical_capture_fields)
    assert result.decision == "possible_photo"
    assert result.evidence_strength == "limited"
    assert result.ai_score == -15
    assert result.summary == "可能为实拍"
    assert result.score_components["metadata"].points == -15
    assert result.supporting_evidence == ["完整相机信息"]
    assert "exif_metadata_can_be_copied_or_edited" in result.limitations


def test_coherent_exif_yields_possible_photo_when_ai_check_is_unavailable() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")
    unavailable = AiLikelihoodResult(
        status="unavailable",
        target_definition="test",
        limitations=["detector unavailable"],
    )

    result = assess_origin(source, unavailable, _c2pa())

    assert result.decision == "possible_photo"
    assert result.ai_score == -15
    assert result.summary == "可能为实拍"


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


def test_trusted_full_uri_capture_claim_yields_a_negative_score_and_possible_photo(tmp_path: Path) -> None:
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

    assert result.decision == "possible_photo"
    assert result.ai_score == -20
    assert result.summary == "可能为实拍"
    assert result.score_components["c2pa_capture"].points == -20
    assert result.supporting_evidence == ["可信数字拍摄来源链"]
    assert result.verified_c2pa_capture_declaration is True
    assert result.trusted_c2pa_capture_declaration is True


def test_no_ai_signal_does_not_become_camera_evidence_without_positive_capture_evidence(tmp_path: Path) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)

    result = assess_origin(asset, _ai(), _c2pa())

    assert result.decision == "no_ai_signal"
    assert result.summary == "未检出 AI 信号"
    assert result.camera_metadata.status == "not_observed"


def test_high_ai_signal_overrides_copyable_camera_metadata() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")

    result = assess_origin(source, _ai("high"), _c2pa())

    assert result.decision == "possible_ai"
    assert result.ai_score == 45
    assert result.summary == "可能为 AI"
    assert result.score_components["dda"].points == 60
    assert result.score_components["metadata"].points == -15
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
    assert assessment.ai_score == 35
    assert assessment.score_components["forensic"].points == 50
    assert assessment.supporting_evidence == ["耐压缩像素检测（有限强度）"]


def test_community_forensics_high_signal_has_a_specific_evidence_label() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")
    result = _ai("high").model_copy(
        update={
            "signals": [
                AiSignal(
                    name="community_forensics_detector",
                    status="available",
                    value=0.9,
                    interpretation="test",
                    details={"high_confidence_threshold": 0.8866265416145325},
                )
            ]
        }
    )

    assessment = assess_origin(source, result, _c2pa())

    assert assessment.supporting_evidence == ["高置信跨生成器像素检测"]


def test_community_forensics_limited_signal_yields_limited_possible_ai() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")
    result = _ai().model_copy(
        update={
            "risk_band": "medium",
            "reliability": 0.65,
            "reliability_label": "limited",
            "signals": [
                AiSignal(
                    name="community_forensics_detector",
                    status="available",
                    value=0.6,
                    interpretation="test",
                    details={
                        "high_confidence_threshold": 0.8866265416145325,
                        "limited_review_threshold": 0.5,
                    },
                )
            ],
        }
    )

    assessment = assess_origin(source, result, _c2pa())

    assert assessment.decision == "possible_ai"
    assert assessment.evidence_strength == "limited"
    assert assessment.ai_score == 35
    assert assessment.score_components["community"].points == 50
    assert assessment.supporting_evidence == ["跨生成器像素检测（有限强度）"]


def test_correlated_pixel_detectors_share_a_twenty_point_corroboration_budget(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)
    ai_result = AiLikelihoodResult(
        status="available",
        risk_band="high",
        reliability=0.9,
        reliability_label="high",
        target_definition="test",
        signals=[
            AiSignal(
                name="safe_pixel_detector",
                status="available",
                value=0.95,
                interpretation="test",
                details={"high_confidence_threshold": 0.90},
            ),
            AiSignal(
                name="community_forensics_detector",
                status="available",
                value=0.90,
                interpretation="test",
                details={"high_confidence_threshold": 0.88},
            ),
            AiSignal(
                name="nonescape_mini_detector",
                status="available",
                value=0.90,
                interpretation="test",
                details={"high_confidence_threshold": 0.88},
            ),
        ],
    )


def _limited_open_watermark() -> ImplicitWatermarkAssessment:
    return ImplicitWatermarkAssessment(
        status="completed",
        adapters=[
            WatermarkAdapterResult(
                adapter_id="test-open-watermark",
                scheme="test",
                detector_version="test",
                run_status="ok",
                observation="positive",
                evidence_class="known_open_ai_watermark",
                direction="supports_ai",
                strength="limited",
                decision_eligible=True,
                coverage=WatermarkCoverage(),
            )
        ],
        direction="supports_ai",
        strength="limited",
        decision_eligible=True,
    )
    geometry = GeometryRelationshipResult(
        status="available",
        probability=0.74,
        risk_band="high",
        applicability=1.0,
        line_count=24,
        decision_threshold=0.61,
        strong_threshold=0.70,
        model_version="test",
        summary="test",
    )

    assessment = assess_origin(asset, ai_result, _c2pa(), geometry_result=geometry)

    assert assessment.ai_score == 100
    assert assessment.score_components["community"].points == 60
    assert assessment.score_components["safe"].points == 10
    assert assessment.score_components["nonescape"].points == 10
    assert assessment.score_components["p0"].points == 20
    assert sum(component.points for component in assessment.score_components.values()) == 100
    assert assessment.summary == "大概率为 AI"
    assert assessment.schema_version == "origin-assessment-v3"
    assert assessment.score_policy_version == SCORE_POLICY_VERSION


def test_verified_c2pa_declaration_splits_ai_score_across_declaration_and_signature(tmp_path: Path) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)
    result = AiLikelihoodResult(
        status="available",
        risk_band="high",
        reliability=1.0,
        reliability_label="high",
        target_definition="test",
        signals=[
            AiSignal(
                name="verified_c2pa",
                status="available",
                value=1.0,
                interpretation="test",
            )
        ],
    )

    assessment = assess_origin(asset, result, _c2pa())

    assert assessment.decision == "possible_ai"
    assert assessment.ai_score == 100
    assert assessment.summary == "大概率为 AI"
    assert assessment.score_components["c2pa_declaration"].points == 70
    assert assessment.score_components["c2pa_signature"].points == 30
    assert all(assessment.score_components[key].points == 0 for key in ("dda", "safe", "forensic", "community", "nonescape"))
    assert assessment.supporting_evidence == ["已验证的 AI 来源声明"]


def test_global_bound_does_not_reduce_semantic_evidence_strength(tmp_path: Path) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)
    result = AiLikelihoodResult(
        status="available",
        risk_band="high",
        reliability=1.0,
        reliability_label="high",
        target_definition="test",
        signals=[
            AiSignal(
                name="community_forensics_detector",
                status="available",
                value=0.90,
                interpretation="test",
                details={"high_confidence_threshold": 0.88},
            )
        ],
    )

    assessment = assess_origin(
        asset,
        result,
        _c2pa(),
        watermark_result=_limited_open_watermark(),
    )

    assert assessment.ai_score == 100
    assert assessment.score_components["community"].points == 50
    assert assessment.score_components["watermark"].points == 50
    assert assessment.evidence_strength == "high"
    assert assessment.supporting_evidence == [
        "高置信跨生成器像素检测",
        "已知开放 AI 生态水印（有限强度）",
    ]


def test_zeroed_pixel_component_is_not_listed_as_supporting_evidence(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)
    result = AiLikelihoodResult(
        status="available",
        risk_band="high",
        reliability=1.0,
        reliability_label="high",
        target_definition="test",
        signals=[
            AiSignal(
                name="safe_pixel_detector",
                status="available",
                value=0.95,
                interpretation="test",
                details={"high_confidence_threshold": 0.90},
            ),
            AiSignal(
                name="community_forensics_detector",
                status="available",
                value=0.90,
                interpretation="test",
                details={"high_confidence_threshold": 0.88},
            ),
        ],
    )

    assessment = assess_origin(
        asset,
        result,
        _c2pa(),
        watermark_result=_limited_open_watermark(),
    )

    assert assessment.ai_score == 100
    assert assessment.score_components["safe"].points == 0
    assert assessment.score_components["community"].points == 50
    assert assessment.score_components["watermark"].points == 50
    assert assessment.supporting_evidence == [
        "高置信跨生成器像素检测",
        "已知开放 AI 生态水印（有限强度）",
    ]


def test_verified_source_prevents_weaker_visual_cards_from_exceeding_the_ring(tmp_path: Path) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)
    result = AiLikelihoodResult(
        status="available",
        risk_band="high",
        reliability=1.0,
        reliability_label="high",
        target_definition="test",
        signals=[
            AiSignal(
                name="verified_c2pa",
                status="available",
                value=1.0,
                interpretation="test",
            ),
            AiSignal(
                name="safe_pixel_detector",
                status="available",
                value=0.95,
                interpretation="test",
                details={"high_confidence_threshold": 0.90},
            ),
            AiSignal(
                name="community_forensics_detector",
                status="available",
                value=0.90,
                interpretation="test",
                details={"high_confidence_threshold": 0.88},
            ),
            AiSignal(
                name="nonescape_mini_detector",
                status="available",
                value=0.90,
                interpretation="test",
                details={"high_confidence_threshold": 0.88},
            ),
        ],
    )

    assessment = assess_origin(asset, result, _c2pa())

    assert assessment.ai_score == 100
    assert assessment.score_components["c2pa_declaration"].points == 70
    assert assessment.score_components["c2pa_signature"].points == 30
    assert all(
        assessment.score_components[key].points == 0
        for key in ("dda", "safe", "forensic", "community", "nonescape", "p0")
    )
    assert sum(component.points for component in assessment.score_components.values()) == 100


def test_origin_score_policy_audit_matches_runtime() -> None:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    audit_path = models_dir / "origin_assessment_policy_v3.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    prior_audit = json.loads(
        (models_dir / "origin_assessment_policy_v2.json").read_text(encoding="utf-8")
    )

    assert audit["policy_version"] == SCORE_POLICY_VERSION
    assert audit["supersedes"] == prior_audit["policy_version"]
    assert prior_audit["status"] == "superseded"
    assert prior_audit["superseded_by"] == SCORE_POLICY_VERSION
    assert audit["origin_result_schema"] == "origin-assessment-v3"
    assert audit["runtime_parameters"] == ORIGIN_SCORE_POLICY.audit_parameters()
    assert "always equals ai_score" in audit["validation"]["required_invariant"]


def test_subthreshold_positive_score_uses_small_ai_probability_label(tmp_path: Path) -> None:
    asset = tmp_path / "metadata_free.png"
    Image.new("RGB", (32, 32)).save(asset)
    result = AiLikelihoodResult(
        status="available",
        target_definition="test",
        signals=[
            AiSignal(
                name="safe_pixel_detector",
                status="available",
                value=0.95,
                interpretation="test",
                details={"high_confidence_threshold": 0.90, "high_confidence_eligible": False},
            )
        ],
    )

    assessment = assess_origin(asset, result, _c2pa())

    assert assessment.decision == "no_ai_signal"
    assert assessment.ai_score == 30
    assert assessment.summary == "小概率为 AI"


def test_registered_strong_geometry_tier_is_bounded_but_visible_in_ai_score(tmp_path: Path) -> None:
    asset = tmp_path / "geometry.png"
    Image.new("RGB", (32, 32)).save(asset)
    geometry = GeometryRelationshipResult(
        status="available",
        probability=0.74,
        risk_band="high",
        applicability=1.0,
        line_count=24,
        decision_threshold=0.61,
        strong_threshold=0.70,
        model_version="test",
        summary="test",
    )

    assessment = assess_origin(asset, _ai(), _c2pa(), geometry_result=geometry)

    assert assessment.decision == "no_ai_signal"
    assert assessment.ai_score == 20
    assert assessment.summary == "未检出 AI 信号"
    assert assessment.score_components["p0"].points == 20


def test_registered_limited_geometry_tier_only_supports_other_evidence(tmp_path: Path) -> None:
    asset = tmp_path / "geometry.png"
    Image.new("RGB", (32, 32)).save(asset)
    geometry = GeometryRelationshipResult(
        status="available",
        probability=0.65,
        risk_band="medium",
        applicability=1.0,
        line_count=24,
        decision_threshold=0.61,
        strong_threshold=0.70,
        model_version="test",
        summary="test",
    )

    assessment = assess_origin(asset, _ai(), _c2pa(), geometry_result=geometry)

    assert assessment.decision == "no_ai_signal"
    assert assessment.ai_score == 10
    assert assessment.score_components["p0"].points == 10


def test_combined_capture_and_exif_evidence_uses_large_photo_probability_label() -> None:
    source = _fixture("f6_01_railway_perspective.jpg")

    assessment = assess_origin(
        source,
        _ai(),
        _c2pa(
            signature=C2paSignatureValidationStatus.VALID,
            source_types=["http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"],
            trust=C2paTrustStatus.TRUSTED,
            trust_list_version="c2pa-tl-test-v1",
        ),
    )

    assert assessment.decision == "possible_photo"
    assert assessment.ai_score == -35
    assert assessment.summary == "可能为实拍"


def test_score_summary_requires_more_than_twenty_points_for_a_small_ai_probability_label() -> None:
    assert _score_summary(65) == "大概率为 AI"
    assert _score_summary(35) == "可能为 AI"
    assert _score_summary(21) == "小概率为 AI"
    assert _score_summary(20) == "未检出 AI 信号"
    assert _score_summary(1) == "未检出 AI 信号"
    assert _score_summary(0) == "未检出 AI 信号"
    assert _score_summary(-1) == "可能为实拍"
    assert _score_summary(-100) == "可能为实拍"


def _fixture(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "p0_f6_real_v2" / "images" / name
