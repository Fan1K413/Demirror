"""Portable runtime inference for the geometry relationship model."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from image_trust.geometry_ai.contracts import GeometryRelationshipModel, GeometryRelationshipResult
from image_trust.geometry_ai.features import FEATURE_SCHEMA_VERSION, extract_image_relationship_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "geometry_relationship_v1.json"


def assess_geometry_ai(
    input_path: Path,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> GeometryRelationshipResult:
    """Return a calibrated geometry-only AI signal or an explicit non-result."""

    model = _load_model(model_path)
    if model is None:
        return GeometryRelationshipResult(
            status="unavailable",
            summary="几何关系模型尚未安装。",
            limitations=["geometry_relationship_model_unavailable"],
        )
    if model.feature_schema_version != FEATURE_SCHEMA_VERSION:
        return GeometryRelationshipResult(
            status="unavailable",
            summary="几何关系模型与当前特征版本不兼容。",
            model_version=model.model_version,
            limitations=["geometry_relationship_feature_schema_mismatch"],
        )
    try:
        features, line_count, _ = extract_image_relationship_features(input_path)
    except (OSError, ValueError) as error:
        return GeometryRelationshipResult(
            status="failed",
            summary="几何关系提取失败。",
            model_version=model.model_version,
            limitations=[str(error)],
        )
    applicability = min(1.0, line_count / max(model.minimum_line_count * 3, 1))
    if line_count < model.minimum_line_count:
        return GeometryRelationshipResult(
            status="not_applicable",
            applicability=applicability,
            line_count=line_count,
            decision_threshold=model.ai_threshold,
            strong_threshold=model.strong_ai_threshold,
            model_version=model.model_version,
            summary="画面中可用直线关系不足，几何分支不作判断。",
            evaluation=model.evaluation,
            limitations=["geometry_relationship_low_line_support"],
        )
    try:
        vector = np.asarray([features[name] for name in model.feature_names], dtype=np.float64)
    except KeyError:
        return GeometryRelationshipResult(
            status="unavailable",
            applicability=applicability,
            line_count=line_count,
            model_version=model.model_version,
            summary="当前几何特征与模型不匹配。",
            limitations=["geometry_relationship_feature_name_mismatch"],
        )
    standardized = (vector - np.asarray(model.standardizer_mean)) / np.asarray(model.standardizer_scale)
    hidden = standardized
    for layer in model.layers:
        hidden = hidden @ np.asarray(layer.weights) + np.asarray(layer.bias)
        if layer.activation == "relu":
            hidden = np.maximum(hidden, 0.0)
    raw_logit = float(hidden.reshape(-1)[0])
    probability = _sigmoid(model.platt_coefficient * raw_logit + model.platt_intercept)
    if probability >= model.strong_ai_threshold:
        risk_band = "high"
        summary = "多处线段关系难以由一致透视结构解释，几何信号强烈支持 AI。"
    elif probability >= model.ai_threshold:
        risk_band = "medium"
        summary = "线段空间与方向关系更接近校准集中的 AI 生成图。"
    else:
        risk_band = "low"
        summary = "几何关系未达到 AI 信号阈值；这不等于实拍证明。"
    findings = _findings(features)
    return GeometryRelationshipResult(
        status="available",
        probability=probability,
        risk_band=risk_band,
        applicability=applicability,
        line_count=line_count,
        decision_threshold=model.ai_threshold,
        strong_threshold=model.strong_ai_threshold,
        model_version=model.model_version,
        summary=summary,
        findings=findings,
        evaluation=model.evaluation,
        limitations=list(model.limitations),
    )


def _findings(features: dict[str, float]) -> list[str]:
    findings: list[str] = []
    if features.get("intersection_concentration", 0.0) < 0.08:
        findings.append("候选消失方向较分散")
    if features.get("orientation_top2_weight", 0.0) < 0.38:
        findings.append("主要方向族的支持较弱")
    if features.get("near_parallel_pair_ratio", 0.0) < 0.10:
        findings.append("局部平行关系较少或较碎")
    if features.get("spatial_weight_entropy", 0.0) > 0.88:
        findings.append("结构线分布较分散")
    return findings[:3]


@lru_cache(maxsize=4)
def _load_model(path: Path) -> GeometryRelationshipModel | None:
    try:
        return GeometryRelationshipModel.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
