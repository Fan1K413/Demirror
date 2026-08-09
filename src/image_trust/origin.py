"""Conservative three-band origin assessment for local image reviews.

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

from image_trust.ai_likelihood.contracts import AiLikelihoodResult
from image_trust.camera.contracts import CameraConsistencyObservation, CameraExperimentResult
from image_trust.provenance.contracts import (
    C2paRecord,
    C2paSignatureValidationStatus,
    C2paTrustStatus,
)


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


class OriginAssessment(BaseModel):
    """Auditable, three-band review result rather than a provenance verdict."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "origin-assessment-v1"
    decision: Literal["possible_ai", "no_ai_signal", "possible_camera"]
    evidence_strength: Literal["high", "limited"]
    summary: str
    explanation: str
    supporting_evidence: list[str] = Field(default_factory=list)
    camera_metadata: CameraMetadataEvidence
    verified_c2pa_capture_declaration: bool = False
    trusted_c2pa_capture_declaration: bool = False
    declared_c2pa_source_types: list[str] = Field(default_factory=list)
    camera_consistency: Literal["measured_not_calibrated", "not_observed", "not_run"]
    implicit_watermark: Literal["not_configured"] = "not_configured"
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
) -> OriginAssessment:
    """Combine direct AI and direct capture signals without negative inference.

    The P1 camera measurement is surfaced but excluded: it has no registered
    source-direction calibration in this project.  Coherent EXIF may support a
    *limited* possible-camera review result only after the AI detector has
    completed without a high-confidence signal.  It remains explicitly
    spoofable and is always overridden by positive AI evidence.
    """

    metadata = inspect_camera_metadata(input_path)
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
        "implicit_watermark_detector_not_configured",
    ]
    if camera_consistency != "not_run":
        limitations.append("camera_consistency_not_calibrated_for_origin_decision")
    if verified_capture and not trusted_capture:
        limitations.append("c2pa_capture_declaration_not_trusted_for_camera_decision")

    if ai_result.risk_band in {"high", "medium"}:
        return OriginAssessment(
            decision="possible_ai",
            evidence_strength=(
                "high"
                if ai_result.risk_band == "high" and ai_result.reliability_label == "high"
                else "limited"
            ),
            summary="可能为 AI",
            explanation=(
                "发现了高置信 AI 像素信号或已验证的 AI 来源声明。"
                if ai_result.risk_band == "high"
                else "耐压缩像素检测达到偏向 AI 召回的有限强度复核阈值；该档允许更高误报，因此需要人工复核。"
            ),
            supporting_evidence=_ai_evidence_labels(ai_result),
            camera_metadata=metadata,
            verified_c2pa_capture_declaration=verified_capture,
            trusted_c2pa_capture_declaration=trusted_capture,
            declared_c2pa_source_types=source_types,
            camera_consistency=camera_consistency,
            limitations=sorted(set(limitations)),
        )
    if trusted_capture:
        return OriginAssessment(
            decision="possible_camera",
            evidence_strength="high",
            summary="可能为实拍",
            explanation="可信 C2PA 来源链支持为数字拍摄，且没有高置信 AI 信号；这不证明画面内容从未编辑。",
            supporting_evidence=["可信 C2PA 数字拍摄来源链"],
            camera_metadata=metadata,
            verified_c2pa_capture_declaration=verified_capture,
            trusted_c2pa_capture_declaration=trusted_capture,
            declared_c2pa_source_types=source_types,
            camera_consistency=camera_consistency,
            limitations=sorted(set(limitations)),
        )
    if ai_result.status == "available" and metadata.status == "coherent":
        supporting = ["完整相机 EXIF（可复制或编辑）"]
        if (
            camera_result is not None
            and camera_result.full_image.status.value == "ok"
        ):
            supporting.append("相机参数估计已完成（仅辅助复核）")
        return OriginAssessment(
            decision="possible_camera",
            evidence_strength="limited",
            summary="可能为实拍",
            explanation="AI 像素检测未达到高置信标准，且图片包含完整的相机、拍摄时间和物理拍摄参数。EXIF 可以被复制或编辑，因此这里只给出有限强度的实拍可能性。",
            supporting_evidence=supporting,
            camera_metadata=metadata,
            verified_c2pa_capture_declaration=verified_capture,
            trusted_c2pa_capture_declaration=trusted_capture,
            declared_c2pa_source_types=source_types,
            camera_consistency=camera_consistency,
            limitations=sorted(set(limitations)),
        )
    return OriginAssessment(
        decision="no_ai_signal",
        evidence_strength="limited",
        summary="未检出 AI 信号",
        explanation="没有出现高置信 AI 信号，也没有可信的数字拍摄来源链。这不是相机来源结论。",
        supporting_evidence=["AI 像素检测未达到高置信标准"],
        camera_metadata=metadata,
        verified_c2pa_capture_declaration=verified_capture,
        trusted_c2pa_capture_declaration=trusted_capture,
        declared_c2pa_source_types=source_types,
        camera_consistency=camera_consistency,
        limitations=sorted(set(limitations)),
    )


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
