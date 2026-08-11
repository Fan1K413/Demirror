from __future__ import annotations

from image_trust.geometry_ai.measurement_types import GeometryMeasurementV2Result
from image_trust.geometry_ai.origin_v2 import (
    FEATURE_NAMES,
    GeometryOriginV2Model,
    extract_geometry_origin_features,
    predict_geometry_origin_v2,
)


def _measurement() -> GeometryMeasurementV2Result:
    return GeometryMeasurementV2Result.model_validate(
        {
            "status": "measurable",
            "summary": "fixture",
            "canonical_size": [1000, 800],
            "applicability": 0.9,
            "gates": [
                {
                    "gate_id": "support",
                    "passed": True,
                    "observed": 12,
                    "threshold": 8,
                    "description": "fixture",
                }
            ],
            "merged_lines": [
                {
                    "line_id": "m1",
                    "x1": 0,
                    "y1": 10,
                    "x2": 100,
                    "y2": 10,
                    "length_px": 100,
                    "length_normalized": 0.1,
                    "angle_rad": 0,
                    "source_line_ids": ["a", "b"],
                    "source_scale_ids": ["global", "tile-0"],
                    "cross_scale_stability": 0.8,
                },
                {
                    "line_id": "m2",
                    "x1": 0,
                    "y1": 20,
                    "x2": 80,
                    "y2": 20,
                    "length_px": 80,
                    "length_normalized": 0.08,
                    "angle_rad": 0,
                    "source_line_ids": ["c"],
                    "source_scale_ids": ["global"],
                    "cross_scale_stability": 0.3,
                },
            ],
            "regions": [
                {
                    "region_id": "r1",
                    "canonical_box": {"x": 0, "y": 0, "width": 200, "height": 100},
                    "cell_ids": ["c1"],
                    "line_ids": ["m1", "m2"],
                    "line_count": 2,
                    "normalized_line_support": 0.18,
                    "orientation_entropy": 0.2,
                    "status": "usable",
                }
            ],
            "families": [
                {
                    "family_id": "f1",
                    "region_id": "r1",
                    "kind": "parallel",
                    "member_line_ids": ["m1", "m2"],
                    "direction_rad": 0,
                    "weighted_inlier_ratio": 0.75,
                    "residual_p50_deg": 1,
                    "residual_p90_deg": 2,
                    "bootstrap_stability": 0.8,
                    "stable": True,
                }
            ],
            "checks": [
                {
                    "check_id": "G1",
                    "title": "parallel",
                    "status": "available",
                    "anomaly_score": 0.7,
                    "findings": [
                        {
                            "finding_id": "g1-1",
                            "check_id": "G1",
                            "severity": 0.6,
                            "description": "fixture",
                        },
                        {
                            "finding_id": "g1-2",
                            "check_id": "G1",
                            "severity": 0.8,
                            "description": "fixture",
                        },
                    ],
                },
                {"check_id": "G2", "title": "vp", "status": "not_applicable"},
                {"check_id": "G3", "title": "spacing", "status": "available", "anomaly_score": 0.1},
                {"check_id": "G4", "title": "continuity", "status": "failed"},
                {"check_id": "G5", "title": "camera", "status": "not_run"},
            ],
        }
    )


def _candidate_model() -> GeometryOriginV2Model:
    return GeometryOriginV2Model(
        model_version="fixture-candidate",
        feature_names=list(FEATURE_NAMES),
        standardizer_mean=[0.0] * len(FEATURE_NAMES),
        standardizer_scale=[1.0] * len(FEATURE_NAMES),
        coefficients=[0.0] * len(FEATURE_NAMES),
        intercept=0.0,
        decision_threshold=0.5,
        deployment_eligible=False,
        replacement_gate={"eligible": False},
    )


def test_feature_extraction_includes_g1_g5_availability_and_structure_statistics() -> None:
    features = extract_geometry_origin_features(_measurement())

    assert tuple(features) == FEATURE_NAMES
    assert features["merged_line_count_log1p"] > 0
    assert features["stable_merged_line_ratio"] == 0.5
    assert features["usable_region_ratio"] == 1.0
    assert features["stable_family_ratio"] == 1.0
    assert features["g1_status_available"] == 1.0
    assert features["g1_anomaly_score"] == 0.7
    assert features["g1_finding_count_log1p"] > 1.0
    assert features["g1_max_finding_severity"] == 0.8
    assert features["g4_status_failed"] == 1.0
    assert features["g5_status_not_run"] == 1.0


def test_candidate_model_is_blocked_by_default_but_can_be_scored_offline() -> None:
    model = _candidate_model()

    blocked = predict_geometry_origin_v2(_measurement(), model)
    offline = predict_geometry_origin_v2(
        _measurement(),
        model,
        allow_ineligible_candidate=True,
    )

    assert blocked.status == "candidate_only"
    assert blocked.probability is None
    assert blocked.limitations == ["geometry_origin_v2_replacement_gate_not_passed"]
    assert offline.status == "candidate_only"
    assert offline.probability == 0.5
    assert offline.predicted_ai is True
    assert list(offline.contributions) == sorted(FEATURE_NAMES)


def test_model_cannot_claim_deployment_eligibility_without_passing_gate() -> None:
    data = _candidate_model().model_dump()
    data["deployment_eligible"] = True

    try:
        GeometryOriginV2Model.model_validate(data)
    except ValueError as error:
        assert "passing replacement gate" in str(error)
    else:
        raise AssertionError("deployment eligibility must be fail-closed")
