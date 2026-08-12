from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from image_trust.geometry_ai.deterministic_surfaces import (
    assess_deterministic_surface_baseline,
)
from image_trust.geometry_ai.relation_annotations import GeometryRelationAnnotation
from image_trust.geometry_ai.surface_agent_diagnostic import (
    AgentSurfaceComparisonReport,
    assess_agent_surface_conditioned_g1_g4,
    build_agent_diagnostic_replay_authorization,
    compare_deterministic_surfaces_with_agent_annotations,
)
from image_trust.geometry_ai.surface_comparison import SurfaceComparisonConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "surface_agent_diagnostic_fixtures",
    REPOSITORY_ROOT / "tests/test_geometry_surface_conditioned.py",
)
assert SPEC is not None and SPEC.loader is not None
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixtures
SPEC.loader.exec_module(fixtures)


def _config(expected_packet_count: int = 6) -> SurfaceComparisonConfig:
    return SurfaceComparisonConfig(
        expected_packet_count=expected_packet_count,
        terminal_annotation_ratio_minimum=1.0,
        completed_packet_count_minimum=1,
        assessable_family_count_minimum=1,
        comparable_family_fraction_minimum=1.0,
        active_line_assignment_fraction_minimum=1.0,
        macro_same_surface_pair_retention_minimum=1.0,
        macro_different_surface_pair_separation_minimum=1.0,
        split_family_recall_minimum=1.0,
        non_split_family_specificity_minimum=0.0,
    )


def _cohort(packet, annotation):
    packets = {}
    annotations = {}
    baselines = {}
    for index in range(6):
        reviewer_id = f"grr-agent-{index}"
        current_packet = packet.model_copy(update={"reviewer_id": reviewer_id})
        current_annotation = annotation.model_copy(update={"reviewer_id": reviewer_id})
        packets[reviewer_id] = current_packet
        annotations[reviewer_id] = current_annotation
        baselines[reviewer_id] = assess_deterministic_surface_baseline(
            fixtures.fixtures.fixtures._image(),
            current_packet,
        )
    return packets, annotations, baselines


def _bound_comparison() -> AgentSurfaceComparisonReport:
    packet = fixtures.fixtures._packet_with_coherent_control()
    annotation = fixtures.fixtures._split_annotation_with_coherent_control()
    packets, annotations, baselines = _cohort(packet, annotation)
    report = compare_deterministic_surfaces_with_agent_annotations(
        packets,
        annotations,
        baselines,
        config=_config(),
    )
    return AgentSurfaceComparisonReport.model_validate(
        {
            **report.model_dump(mode="json"),
            "diagnostic_protocol_sha256": "1" * 64,
            "agent_annotation_audit_sha256": "2" * 64,
            "baseline_audit_sha256": "3" * 64,
            "review_manifest_sha256": "4" * 64,
        }
    )


def test_agent_comparison_is_explicitly_diagnostic_only() -> None:
    report = _bound_comparison()

    assert report.completed_packet_count == 6
    assert report.counterfactual_human_thresholds_passed is True
    assert report.ai_assisted_annotations_used is True
    assert report.human_annotations_used is False
    assert report.posthoc_source_key_opened is False
    assert report.origin_scoring_authorized is False
    assert report.decision == "diagnostic_only_human_confirmation_still_required"


def test_agent_comparison_rejects_pending_annotations() -> None:
    packet = fixtures.fixtures._packet_with_coherent_control()
    annotation = fixtures.fixtures._split_annotation_with_coherent_control()
    packets, annotations, baselines = _cohort(packet, annotation)
    first = sorted(annotations)[0]
    annotations[first] = GeometryRelationAnnotation(
        reviewer_id=first,
        proposed_family_reviews=[
            review.model_copy(update={"verdict": "pending", "surface_ids": []})
            for review in annotation.proposed_family_reviews
        ],
    )

    with pytest.raises(ValueError, match="all be terminal"):
        compare_deterministic_surfaces_with_agent_annotations(
            packets,
            annotations,
            baselines,
            config=_config(),
        )


def test_agent_authorization_never_claims_human_or_origin_authority() -> None:
    report = _bound_comparison().model_dump(mode="json")
    report["counterfactual_human_gates"] = {"counterfactual": False}
    report["counterfactual_human_thresholds_passed"] = False

    authorization = build_agent_diagnostic_replay_authorization(
        report,
        comparison_report_sha256="5" * 64,
        g1_g4_measurement_sha256="6" * 64,
    )

    assert authorization.ai_assisted_annotations_used is True
    assert authorization.human_annotations_used is False
    assert authorization.origin_scoring_authorized is False
    assert len(authorization.reviewer_ids) == 6

    report["human_annotations_used"] = True
    with pytest.raises(ValueError, match="diagnostic field"):
        build_agent_diagnostic_replay_authorization(
            report,
            comparison_report_sha256="5" * 64,
            g1_g4_measurement_sha256="6" * 64,
        )


def test_agent_surface_replay_wraps_frozen_g1_g4_with_ai_semantics() -> None:
    packet = fixtures.fixtures._packet_with_coherent_control()
    baseline = assess_deterministic_surface_baseline(
        fixtures.fixtures.fixtures._image(),
        packet,
    )
    measurement = fixtures._measurement(packet)
    rgb = np.asarray(fixtures.fixtures.fixtures._image().convert("RGB"))
    comparison = _bound_comparison().model_dump(mode="json")
    authorization = build_agent_diagnostic_replay_authorization(
        comparison,
        comparison_report_sha256="5" * 64,
        g1_g4_measurement_sha256="6" * 64,
    ).model_copy(update={"reviewer_ids": [packet.reviewer_id]})

    diagnostic = assess_agent_surface_conditioned_g1_g4(
        rgb,
        measurement,
        packet,
        baseline,
        authorization,
    )

    assert diagnostic.ai_assisted_annotations_used is True
    assert diagnostic.human_annotations_used is False
    assert diagnostic.origin_scoring_authorized is False
    assert [check.check_id for check in diagnostic.result.checks] == [
        "G1",
        "G2",
        "G3",
        "G4",
    ]


def test_agent_diagnostic_protocol_preserves_human_gate() -> None:
    import json

    protocol_path = (
        REPOSITORY_ROOT
        / "research/records/2026-08-12/geometry/"
        "geometry_surface_agent_diagnostic_protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert protocol["decision"] == (
        "diagnostic_only_human_confirmation_still_required"
    )
    assert protocol["ai_assisted_annotations_used"] is True
    assert protocol["human_annotations_used"] is False
    assert protocol["origin_scoring_authorized"] is False
    assert protocol["web_integration_authorized"] is False
