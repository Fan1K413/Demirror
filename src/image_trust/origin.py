"""Conservative two-band origin assessment for local image reviews.

The result is deliberately categorical.  It may report a possible camera
capture only from positive capture evidence; a missing AI signal is never
silently converted into a camera-origin conclusion.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Literal

from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from image_trust.ai_likelihood.contracts import AiLikelihoodResult, AiSignal
from image_trust.camera.contracts import CameraConsistencyObservation, CameraExperimentResult
from image_trust.geometry_ai.contracts import GeometryRelationshipResult
from image_trust.provenance.contracts import (
    C2paRecord,
    C2paSignatureValidationStatus,
    C2paTrustStatus,
)
from image_trust.watermark.contracts import ImplicitWatermarkAssessment


_CAPTURE_SOURCE_TYPES = {"digitalcapture", "computationalcapture"}
_PHYSICAL_EXIF_TAGS = {
    "曝光时间": 33434,
    "光圈": 33437,
    "ISO": 34855,
    "焦距": 37386,
}


class CameraMetadataEvidence(BaseModel):
    """Read-only EXIF evidence; it can be copied or edited and is not proof."""

    model_config = ConfigDict(frozen=True)

    status: Literal["coherent", "partial", "not_observed", "unavailable"]
    camera_make: str | None = None
    camera_model: str | None = None
    captured_at_local: str | None = None
    physical_capture_fields: list[str] = Field(default_factory=list)
    software: str | None = None
    limitations: list[str] = Field(default_factory=list)


class AiScoreComponent(BaseModel):
    """One transparent contribution to the bounded AI signal score."""

    model_config = ConfigDict(frozen=True)

    points: int = Field(ge=-100, le=100)
    state: Literal["positive", "negative", "neutral", "not_detected"]
    explanation: str


class OriginAssessment(BaseModel):
    """Auditable three-band review result with a signed, non-probabilistic score."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "origin-assessment-v2"
    decision: Literal["possible_ai", "possible_photo", "no_ai_signal"]
    evidence_strength: Literal["high", "limited"]
    ai_score: int = Field(ge=-100, le=100)
    score_components: dict[str, AiScoreComponent] = Field(default_factory=dict)
    summary: str
    explanation: str
    supporting_evidence: list[str] = Field(default_factory=list)
    camera_metadata: CameraMetadataEvidence
    verified_c2pa_capture_declaration: bool = False
    trusted_c2pa_capture_declaration: bool = False
    declared_c2pa_source_types: list[str] = Field(default_factory=list)
    camera_consistency: Literal["measured_not_calibrated", "not_observed", "not_run"]
    implicit_watermark: ImplicitWatermarkAssessment = Field(
        default_factory=ImplicitWatermarkAssessment.not_configured
    )
    limitations: list[str] = Field(default_factory=list)


def inspect_camera_metadata(input_path: Path) -> CameraMetadataEvidence:
    """Read the small, camera-related EXIF subset needed for an auditable rule."""

    try:
        with Image.open(input_path) as image:
            exif = image.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif) if exif else {}
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return CameraMetadataEvidence(
            status="unavailable",
            limitations=[f"camera_metadata_unavailable:{type(error).__name__}"],
        )

    make = _text(exif.get(271)) if exif else None
    model = _text(exif.get(272)) if exif else None
    captured_at = _valid_capture_time(
        _text(exif_ifd.get(36867)) or _text(exif.get(306) if exif else None)
    )
    software = _text(exif.get(305)) if exif else None
    physical_fields = sorted(
        name for name, tag in _PHYSICAL_EXIF_TAGS.items() if _positive_number(exif_ifd.get(tag))
    )
    identifiers_complete = bool(make and model and captured_at)
    if identifiers_complete and len(physical_fields) >= 3:
        status: Literal["coherent", "partial", "not_observed", "unavailable"] = "coherent"
    elif make or model or captured_at or physical_fields:
        status = "partial"
    else:
        status = "not_observed"
    limitations = ["exif_metadata_can_be_copied_or_edited"] if status != "not_observed" else []
    return CameraMetadataEvidence(
        status=status,
        camera_make=make,
        camera_model=model,
        captured_at_local=captured_at,
        physical_capture_fields=physical_fields,
        software=software,
        limitations=limitations,
    )


def assess_origin(
    input_path: Path,
    ai_result: AiLikelihoodResult,
    c2pa_record: C2paRecord,
    camera_result: CameraExperimentResult | None = None,
    watermark_result: ImplicitWatermarkAssessment | None = None,
    geometry_result: GeometryRelationshipResult | None = None,
) -> OriginAssessment:
    """Combine registered AI signals and limited camera counter-evidence.

    ``ai_score`` is a signed, bounded evidence score, not a probability that
    an image is AI-generated. Positive AI evidence raises it; trusted capture
    records and coherent camera metadata lower it. Geometry and camera-parameter
    measurements remain visible for review but receive zero points until a source
    calibration exists.  The registered geometry relationship model is an
    exception: its contribution is explicitly capped because it is a weak,
    generator-limited auxiliary signal rather than provenance evidence.
    """

    metadata = inspect_camera_metadata(input_path)
    watermark = watermark_result or ImplicitWatermarkAssessment.not_configured()
    source_types = sorted(set(c2pa_record.declared_digital_source_types))
    verified_capture = (
        c2pa_record.signature_validation_status is C2paSignatureValidationStatus.VALID
        and bool({_source_type_slug(value) for value in source_types} & _CAPTURE_SOURCE_TYPES)
    )
    trusted_capture = (
        verified_capture
        and c2pa_record.trust_status is C2paTrustStatus.TRUSTED
        and c2pa_record.trust_list_version != "not_configured"
    )
    camera_consistency = _camera_consistency_status(camera_result)
    limitations = [
        *metadata.limitations,
        *watermark.limitations,
    ]
    if camera_consistency != "not_run":
        limitations.append("camera_consistency_not_calibrated_for_origin_decision")
    if geometry_result is not None:
        limitations.extend(geometry_result.limitations)
    if verified_capture and not trusted_capture:
        limitations.append("c2pa_capture_declaration_not_trusted_for_camera_decision")

    score_components = _score_components(
        ai_result,
        watermark,
        metadata,
        trusted_capture=trusted_capture,
        geometry_result=geometry_result,
    )
    ai_score = max(-100, min(100, sum(component.points for component in score_components.values())))
    positive_components = [
        component for component in score_components.values() if component.points > 0
    ]
    possible_ai = ai_score >= 35
    high_strength = any(component.points >= 60 for component in positive_components)
    ai_labels = _ai_evidence_labels(ai_result) if any(
        score_components[key].points > 0
        for key in _PIXEL_COMPONENT_KEYS
    ) else []
    watermark_labels = _watermark_evidence_labels(watermark) if score_components["watermark"].points else []
    if possible_ai:
        return OriginAssessment(
            decision="possible_ai",
            evidence_strength="high" if high_strength else "limited",
            ai_score=ai_score,
            score_components=score_components,
            summary=_score_summary(ai_score),
            explanation="已检出可计分的 AI 信号。AI 分数按各项加减分汇总。",
            supporting_evidence=ai_labels + watermark_labels,
            camera_metadata=metadata,
            verified_c2pa_capture_declaration=verified_capture,
            trusted_c2pa_capture_declaration=trusted_capture,
            declared_c2pa_source_types=source_types,
            camera_consistency=camera_consistency,
            implicit_watermark=watermark,
            limitations=sorted(set(limitations)),
        )
    if ai_score < 0:
        return OriginAssessment(
            decision="possible_photo",
            evidence_strength="limited",
            ai_score=ai_score,
            score_components=score_components,
            summary=_score_summary(ai_score),
            explanation="检测到可计分的拍摄来源或相机线索，AI 分数为负。",
            supporting_evidence=_photo_evidence_labels(
                metadata,
                trusted_capture=trusted_capture,
            ),
            camera_metadata=metadata,
            verified_c2pa_capture_declaration=verified_capture,
            trusted_c2pa_capture_declaration=trusted_capture,
            declared_c2pa_source_types=source_types,
            camera_consistency=camera_consistency,
            implicit_watermark=watermark,
            limitations=sorted(set(limitations)),
        )
    return OriginAssessment(
        decision="no_ai_signal",
        evidence_strength="limited",
        ai_score=ai_score,
        score_components=score_components,
        summary=_score_summary(ai_score),
        explanation="没有出现足以达到 AI 判断分界的可计分 AI 信号。",
        supporting_evidence=["未达到 AI 信号分界"],
        camera_metadata=metadata,
        verified_c2pa_capture_declaration=verified_capture,
        trusted_c2pa_capture_declaration=trusted_capture,
        declared_c2pa_source_types=source_types,
        camera_consistency=camera_consistency,
        implicit_watermark=watermark,
        limitations=sorted(set(limitations)),
    )


def _score_components(
    ai_result: AiLikelihoodResult,
    watermark: ImplicitWatermarkAssessment,
    metadata: CameraMetadataEvidence,
    *,
    trusted_capture: bool,
    geometry_result: GeometryRelationshipResult | None,
) -> dict[str, AiScoreComponent]:
    """Return the registered card-level score contributions for one review."""

    verified_ai_declaration = any(
        signal.name == "verified_c2pa" and signal.status == "available"
        for signal in ai_result.signals
    )
    if verified_ai_declaration:
        c2pa_declaration = AiScoreComponent(points=70, state="positive", explanation="C2PA 明确声明生成式内容")
        c2pa_signature = AiScoreComponent(points=30, state="positive", explanation="C2PA 签名已通过本地验证")
    else:
        c2pa_declaration = AiScoreComponent(points=0, state="not_detected", explanation="未检出已验证的生成式来源声明")
        c2pa_signature = AiScoreComponent(points=0, state="neutral", explanation="签名状态本身不指向 AI")
    if trusted_capture:
        c2pa_capture = AiScoreComponent(points=-20, state="negative", explanation="可信数字拍摄来源链")
    else:
        c2pa_capture = AiScoreComponent(points=0, state="neutral", explanation="未形成可信数字拍摄来源线索")

    if watermark.decision_eligible and watermark.strength == "strong":
        watermark_component = AiScoreComponent(points=80, state="positive", explanation="强度较高的 AI 隐式水印")
    elif watermark.decision_eligible:
        watermark_component = AiScoreComponent(points=50, state="positive", explanation="有限强度 AI 隐式水印")
    elif watermark.status == "completed":
        watermark_component = AiScoreComponent(points=0, state="not_detected", explanation="未检出可计分 AI 隐式水印")
    else:
        watermark_component = AiScoreComponent(points=0, state="neutral", explanation="水印检测未形成可用结论")

    if metadata.status == "coherent":
        metadata_component = AiScoreComponent(points=-15, state="negative", explanation="相机信息完整但可被复制或编辑")
    elif metadata.status == "partial":
        metadata_component = AiScoreComponent(points=-5, state="negative", explanation="检测到部分相机信息")
    else:
        metadata_component = AiScoreComponent(points=0, state="neutral", explanation="未检测到可用相机信息")

    return {
        **_pixel_score_components(ai_result),
        "c2pa_declaration": c2pa_declaration,
        "c2pa_signature": c2pa_signature,
        "c2pa_capture": c2pa_capture,
        "metadata": metadata_component,
        "p0": _score_geometry_relationship(geometry_result),
        "camera": AiScoreComponent(
            points=0,
            state="neutral",
            explanation="相机参数一致性尚未形成可复现的来源区分能力，仅供复核",
        ),
        "watermark": watermark_component,
    }


def _score_geometry_relationship(
    result: GeometryRelationshipResult | None,
) -> AiScoreComponent:
    """Map the registered, bounded geometry tiers to transparent score points."""

    if result is None:
        return AiScoreComponent(points=0, state="neutral", explanation="几何关系模型未运行")
    if result.status == "not_applicable":
        return AiScoreComponent(points=0, state="neutral", explanation="可用直线关系不足，未形成几何分数")
    if result.status != "available" or result.probability is None:
        return AiScoreComponent(points=0, state="neutral", explanation="几何关系模型未形成可用结果")
    if result.risk_band == "high":
        return AiScoreComponent(
            points=25,
            state="positive",
            explanation="线段关系达到经独立留出集校准的较强几何 AI 线索阈值",
        )
    if result.risk_band == "medium":
        return AiScoreComponent(
            points=10,
            state="positive",
            explanation="线段关系达到辅助几何 AI 线索阈值；该项不单独给出 AI 结论",
        )
    return AiScoreComponent(points=0, state="not_detected", explanation="未检出达到计分阈值的几何 AI 线索")


_PIXEL_COMPONENT_KEYS = (
    "dda",
    "safe",
    "forensic",
    "community",
    "nonescape",
)
_PIXEL_SCORE_POLICY: dict[str, tuple[str, int, int]] = {
    "dda_pixel_detector": ("dda", 60, 45),
    "safe_pixel_detector": ("safe", 50, 30),
    "forensic_clip_detector": ("forensic", 60, 50),
    "community_forensics_detector": ("community", 60, 50),
    "nonescape_mini_detector": ("nonescape", 50, 35),
}


def _pixel_score_components(ai_result: AiLikelihoodResult) -> dict[str, AiScoreComponent]:
    signals = {signal.name: signal for signal in ai_result.signals}
    components: dict[str, AiScoreComponent] = {}
    pixel_signal_count = sum(name in _PIXEL_SCORE_POLICY for name in signals)
    for signal_name, (key, high_points, limited_points) in _PIXEL_SCORE_POLICY.items():
        signal = signals.get(signal_name)
        components[key] = _score_pixel_signal(
            signal,
            ai_result,
            high_points=high_points,
            limited_points=limited_points,
            pixel_signal_count=pixel_signal_count,
        )
    return components


def _score_pixel_signal(
    signal: AiSignal | None,
    ai_result: AiLikelihoodResult,
    *,
    high_points: int,
    limited_points: int,
    pixel_signal_count: int,
) -> AiScoreComponent:
    if ai_result.status != "available":
        return AiScoreComponent(points=0, state="neutral", explanation="像素检测未形成可用结果")
    if signal is None or signal.status in {"not_run", "unavailable", "failed"}:
        return AiScoreComponent(points=0, state="neutral", explanation="该像素检测未形成可用结果")
    if signal.status != "available" or signal.value is None:
        return AiScoreComponent(points=0, state="not_detected", explanation="未检出可计分 AI 像素信号")
    high_threshold = _number(signal.details.get("high_confidence_threshold"))
    limited_threshold = _number(signal.details.get("limited_review_threshold"))
    high_eligible = signal.details.get("high_confidence_eligible") is not False
    if high_threshold is not None and signal.value >= high_threshold:
        if high_eligible:
            return AiScoreComponent(points=high_points, state="positive", explanation="达到高强度 AI 像素阈值")
        return AiScoreComponent(points=limited_points, state="positive", explanation="达到阈值，但当前格式只作有限强度复核")
    if limited_threshold is not None and signal.value >= limited_threshold:
        return AiScoreComponent(points=limited_points, state="positive", explanation="达到有限强度 AI 像素阈值")
    if high_threshold is None and pixel_signal_count == 1 and ai_result.risk_band == "high":
        return AiScoreComponent(points=high_points, state="positive", explanation="达到高强度 AI 像素阈值")
    if high_threshold is None and pixel_signal_count == 1 and ai_result.risk_band == "medium":
        return AiScoreComponent(points=limited_points, state="positive", explanation="达到有限强度 AI 像素阈值")
    return AiScoreComponent(points=0, state="not_detected", explanation="未检出可计分 AI 像素信号")


def _photo_evidence_labels(
    metadata: CameraMetadataEvidence,
    *,
    trusted_capture: bool,
) -> list[str]:
    labels: list[str] = []
    if trusted_capture:
        labels.append("可信数字拍摄来源链")
    if metadata.status == "coherent":
        labels.append("完整相机信息")
    elif metadata.status == "partial":
        labels.append("部分相机信息")
    return labels or ["可计分拍摄线索"]


def _score_summary(ai_score: int) -> str:
    """Return an intentionally probabilistic, human-facing label for a signed score."""

    if ai_score >= 65:
        return "大概率为 AI"
    if ai_score >= 35:
        return "可能为 AI"
    if ai_score > 20:
        return "小概率为 AI"
    if ai_score >= 0:
        return "未检出 AI 信号"
    return "可能为实拍"


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _watermark_evidence_labels(result: ImplicitWatermarkAssessment) -> list[str]:
    labels: list[str] = []
    for adapter in result.adapters:
        if not adapter.decision_eligible:
            continue
        if adapter.evidence_class == "verified_provider_ai":
            labels.append("已验证的供应商 AI 隐式水印")
        elif adapter.evidence_class == "known_open_ai_watermark":
            labels.append("已知开放 AI 生态水印（有限强度）")
    return labels


def _camera_consistency_status(
    camera_result: CameraExperimentResult | None,
) -> Literal["measured_not_calibrated", "not_observed", "not_run"]:
    if camera_result is None:
        return "not_run"
    if camera_result.e_cam.observation is CameraConsistencyObservation.MEASURED:
        return "measured_not_calibrated"
    return "not_observed"


def _ai_evidence_labels(result: AiLikelihoodResult) -> list[str]:
    labels: list[str] = []
    for signal in result.signals:
        if signal.status != "available":
            continue
        if signal.name == "verified_c2pa":
            labels.append("已验证的 AI 来源声明")
        elif signal.name == "dda_pixel_detector":
            threshold = float(signal.details.get("high_confidence_threshold", 0.94))
            if signal.value is not None and signal.value >= threshold:
                labels.append("高置信 AI 像素检测")
        elif signal.name == "safe_pixel_detector":
            threshold = float(signal.details.get("high_confidence_threshold", 0.90))
            if signal.value is not None and signal.value >= threshold:
                labels.append("高置信无损纹理检测")
        elif signal.name == "forensic_clip_detector":
            threshold = float(signal.details.get("high_confidence_threshold", 0.9925177097320557))
            if signal.value is not None and signal.value >= threshold:
                labels.append("高置信耐压缩像素检测")
            else:
                limited_threshold = float(signal.details.get("limited_review_threshold", 1.0))
                if signal.value is not None and signal.value >= limited_threshold:
                    labels.append("耐压缩像素检测（有限强度）")
        elif signal.name == "community_forensics_detector":
            threshold = float(signal.details.get("high_confidence_threshold", 0.8866265416145325))
            if signal.value is not None and signal.value >= threshold:
                labels.append("高置信跨生成器像素检测")
            else:
                limited_threshold = float(signal.details.get("limited_review_threshold", 1.0))
                if signal.value is not None and signal.value >= limited_threshold:
                    labels.append("跨生成器像素检测（有限强度）")
    return labels or ["高置信 AI 信号"]


def _source_type_slug(value: str) -> str:
    candidate = value.strip().rstrip("/")
    candidate = candidate.rsplit("/", 1)[-1]
    candidate = candidate.rsplit(":", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", candidate.lower())


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _valid_capture_time(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    return value


def _positive_number(value: object) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return False
