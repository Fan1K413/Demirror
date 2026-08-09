from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

from image_trust.p2.contracts import P2ModelArtifact
from image_trust.p2.inference import _load_artifact, infer_experimental_ai_confidence
from image_trust.schemas import AnalysisResult


def _p0_result() -> AnalysisResult:
    return cast(
        AnalysisResult,
        SimpleNamespace(
            evidence=SimpleNamespace(
                applicability=0.75,
                coverage=0.50,
                features={
                    "line_count": 120,
                    "vp_residual_deg": None,
                    "curve_suppression": {"suppressed_line_count": 5},
                },
            )
        ),
    )


def test_p2_returns_unavailable_without_a_model(tmp_path) -> None:
    result = infer_experimental_ai_confidence(_p0_result(), tmp_path / "missing.json")
    assert result.status == "unavailable"
    assert result.calibrated_probability is None


def test_p2_uses_a_serialized_calibrated_model(tmp_path) -> None:
    artifact = P2ModelArtifact(
        model_version="test-model",
        feature_names=["applicability", "coverage"],
        standardizer_mean=[0.0, 0.0],
        standardizer_scale=[1.0, 1.0],
        base_coefficients=[1.0, 1.0],
        base_intercept=0.0,
        platt_coefficient=1.0,
        platt_intercept=0.0,
        target_definition="test target",
        calibration_dataset={},
        evaluation={"held_out_test_roc_auc": 0.5},
    )
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact.model_dump()), encoding="utf-8")
    _load_artifact.cache_clear()
    result = infer_experimental_ai_confidence(_p0_result(), path)
    assert result.status == "available"
    assert result.model_version == "test-model"
    assert result.calibrated_probability is not None
    assert result.calibrated_probability > 0.5
