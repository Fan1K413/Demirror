"""Runtime inference for the compact, auditable P2 model artifact."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from image_trust.p2.contracts import P2ExperimentalConfidence, P2ModelArtifact
from image_trust.schemas import AnalysisResult


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "p2_projective_geometry_pilot_v1.json"
)


def infer_experimental_ai_confidence(
    p0_result: AnalysisResult,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> P2ExperimentalConfidence:
    """Return a P2 probability, never adding a score to the P0 result.

    The response is deliberately unavailable when the signed-in local project
    has no verified artifact.  This avoids silently substituting a heuristic
    for a calibrated model.
    """

    artifact = _load_artifact(model_path)
    if artifact is None:
        return P2ExperimentalConfidence(
            status="unavailable",
            target_definition="No local P2 model artifact is installed.",
            limitations=["p2_model_artifact_not_available"],
        )
    values = extract_geometry_features(p0_result)
    vector = np.asarray([values[name] for name in artifact.feature_names], dtype=float)
    mean = np.asarray(artifact.standardizer_mean, dtype=float)
    scale = np.asarray(artifact.standardizer_scale, dtype=float)
    coefficients = np.asarray(artifact.base_coefficients, dtype=float)
    raw_logit = float(np.dot((vector - mean) / scale, coefficients) + artifact.base_intercept)
    probability = _sigmoid(artifact.platt_coefficient * raw_logit + artifact.platt_intercept)
    limitations = list(artifact.limitations)
    if p0_result.evidence.applicability < 0.45:
        limitations.append("p2_low_geometry_applicability")
    return P2ExperimentalConfidence(
        status="available",
        calibrated_probability=probability,
        target_definition=artifact.target_definition,
        model_version=artifact.model_version,
        evaluation=artifact.evaluation,
        limitations=sorted(set(limitations)),
    )


def extract_geometry_features(p0_result: AnalysisResult) -> dict[str, float]:
    """Map P0's descriptive output to the frozen P2 numeric feature schema."""

    evidence = p0_result.evidence
    features: Mapping[str, object] = evidence.features
    curve = _mapping(features.get("curve_suppression"))
    residual = _finite(features.get("vp_residual_deg"), default=10.0)
    return {
        "applicability": evidence.applicability,
        "coverage": evidence.coverage,
        "line_count_log": min(1.0, math.log1p(_finite(features.get("line_count"))) / math.log1p(750.0)),
        "total_length_normalized": _finite(features.get("total_length_normalized")),
        "spatial_coverage": _finite(features.get("spatial_coverage")),
        "spatial_entropy": _finite(features.get("spatial_entropy")),
        "direction_entropy": _finite(features.get("direction_entropy")),
        "occupied_cells_normalized": min(1.0, _finite(features.get("occupied_cells")) / 64.0),
        "vp_family_count_normalized": min(1.0, _finite(features.get("vp_family_count")) / 4.0),
        "vp_inlier_ratio": _finite(features.get("vp_inlier_ratio")),
        "vp_residual_deg_normalized": min(1.0, residual / 10.0),
        "vp_residual_observed": 1.0 if features.get("vp_residual_deg") is not None else 0.0,
        "family_stability": _finite(features.get("family_stability")),
        "parallel_family_count_normalized": min(1.0, _finite(features.get("parallel_family_count")) / 6.0),
        "local_family_count_normalized": min(1.0, _finite(features.get("local_family_count")) / 8.0),
        "anomalous_line_count_normalized": min(1.0, _sequence_length(features.get("anomalous_lines")) / 25.0),
        "curve_suppressed_line_count_normalized": min(1.0, _finite(curve.get("suppressed_line_count")) / 100.0),
    }


@lru_cache(maxsize=4)
def _load_artifact(path: Path) -> P2ModelArtifact | None:
    try:
        return P2ModelArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence_length(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _finite(value: object, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
