from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from image_trust.geometry_ai.deterministic_surfaces import (
    assess_deterministic_surface_baseline,
)
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    NormalizedPoint,
    ProposedFamilyReview,
    SemanticSurface,
)
from image_trust.geometry_ai.surface_comparison import (
    DEFAULT_SURFACE_COMPARISON_CONFIG,
    REQUIRED_QUALITY_GATES,
    HumanRelationQualityReceipt,
    SurfaceComparisonConfig,
    compare_deterministic_surfaces_with_human,
    extract_human_quality_receipt,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "surface_comparison_fixtures",
    REPOSITORY_ROOT / "tests/test_geometry_deterministic_surfaces.py",
)
assert SPEC is not None and SPEC.loader is not None
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixtures
SPEC.loader.exec_module(fixtures)


def _polygon(x0: float, x1: float) -> list[NormalizedPoint]:
    return [
        NormalizedPoint(x=x0, y=0.05),
        NormalizedPoint(x=x1, y=0.05),
        NormalizedPoint(x=x1, y=0.95),
        NormalizedPoint(x=x0, y=0.95),
    ]


def _split_annotation():
    return GeometryRelationAnnotation(
        reviewer_id="grr-deterministic-test",
        status="completed",
        surfaces=[
            SemanticSurface(
                surface_id="left",
                surface_kind="facade",
                polygon_normalized=_polygon(0.0, 0.5),
                line_ids=["m0001", "m0002"],
                visibility="clear",
            ),
            SemanticSurface(
                surface_id="right",
                surface_kind="facade",
                polygon_normalized=_polygon(0.5, 1.0),
                line_ids=["m0003", "m0004"],
                visibility="clear",
            ),
        ],
        proposed_family_reviews=[
            ProposedFamilyReview(
                proposed_family_id="global-p01",
                verdict="split_across_surfaces",
                surface_ids=["left", "right"],
            )
        ],
    )


def _packet_with_coherent_control():
    packet = fixtures._packet()
    control = packet.family_proposals[0].model_copy(
        update={
            "family_id": "global-p02",
            "member_line_ids": ["m0001", "m0002"],
        }
    )
    return packet.model_copy(
        update={"family_proposals": [*packet.family_proposals, control]}
    )


def _split_annotation_with_coherent_control():
    annotation = _split_annotation()
    return annotation.model_copy(
        update={
            "proposed_family_reviews": [
                *annotation.proposed_family_reviews,
                ProposedFamilyReview(
                    proposed_family_id="global-p02",
                    verdict="coherent_within_surface",
                    surface_ids=["left"],
                ),
            ]
        }
    )


def _coherent_annotation():
    return GeometryRelationAnnotation(
        reviewer_id="grr-deterministic-test",
        status="completed",
        surfaces=[
            SemanticSurface(
                surface_id="whole",
                surface_kind="facade",
                polygon_normalized=_polygon(0.0, 1.0),
                line_ids=["m0001", "m0002", "m0003", "m0004"],
                visibility="clear",
            )
        ],
        proposed_family_reviews=[
            ProposedFamilyReview(
                proposed_family_id="global-p01",
                verdict="coherent_within_surface",
                surface_ids=["whole"],
            )
        ],
    )


def _receipt(packet_count: int = 6):
    return HumanRelationQualityReceipt(
        input_audit_sha256="a" * 64,
        packet_count=packet_count,
        unique_source_count=3,
        completed_unique_count=3,
        hidden_duplicate_group_count=3,
        completed_duplicate_pair_count=3,
        family_decision_agreement=1.0,
        surface_line_pair_agreement=1.0,
        gates={gate: True for gate in REQUIRED_QUALITY_GATES},
        passed=True,
    )


def _config(expected_packet_count: int = 6, **updates):
    values = {
        "expected_packet_count": expected_packet_count,
        "terminal_annotation_ratio_minimum": 1.0,
        "completed_packet_count_minimum": 1,
        "assessable_family_count_minimum": 1,
        "comparable_family_fraction_minimum": 1.0,
        "active_line_assignment_fraction_minimum": 1.0,
        "macro_same_surface_pair_retention_minimum": 1.0,
        "macro_different_surface_pair_separation_minimum": 1.0,
        "split_family_recall_minimum": 1.0,
        "non_split_family_specificity_minimum": 0.0,
    }
    values.update(updates)
    return SurfaceComparisonConfig(**values)


def _cohort(packet, annotation):
    packets = {}
    annotations = {}
    baselines = {}
    for index in range(6):
        reviewer_id = f"grr-comparison-{index}"
        current_packet = packet.model_copy(update={"reviewer_id": reviewer_id})
        current_annotation = annotation.model_copy(update={"reviewer_id": reviewer_id})
        packets[reviewer_id] = current_packet
        annotations[reviewer_id] = current_annotation
        baselines[reviewer_id] = assess_deterministic_surface_baseline(
            fixtures._image(), current_packet
        )
    return packets, annotations, baselines


def test_split_annotation_matches_two_deterministic_surfaces() -> None:
    packet = _packet_with_coherent_control()
    packets, annotations, baselines = _cohort(
        packet,
        _split_annotation_with_coherent_control(),
    )

    report = compare_deterministic_surfaces_with_human(
        packets,
        annotations,
        baselines,
        quality_receipt=_receipt(),
        config=_config(),
    )

    assert report.status == "complete"
    assert report.passed is True
    assert report.decision == "eligible_for_surface_conditioned_g1_g4_replay_only"
    assert report.family_counts.true_positive == 6
    assert report.family_counts.true_negative == 6
    assert report.pair_counts.true_positive == 18
    assert report.pair_counts.true_negative == 24
    assert report.active_line_assignment_fraction == 1.0
    assert report.source_key_opened is False
    assert report.origin_scoring_authorized is False


def test_coherent_human_surface_exposes_deterministic_oversplitting() -> None:
    packet = fixtures._packet()
    packets, annotations, baselines = _cohort(packet, _coherent_annotation())

    report = compare_deterministic_surfaces_with_human(
        packets,
        annotations,
        baselines,
        quality_receipt=_receipt(),
        config=_config(
            macro_same_surface_pair_retention_minimum=0.70,
            macro_different_surface_pair_separation_minimum=0.0,
            split_family_recall_minimum=0.0,
            non_split_family_specificity_minimum=0.70,
        ),
    )

    assert report.passed is False
    assert report.family_counts.false_positive == 6
    assert report.pair_counts.true_positive == 12
    assert report.pair_counts.false_negative == 24
    assert report.macro_same_surface_pair_retention == pytest.approx(1 / 3)
    assert report.decision == "keep_as_review_visualization_without_tuning"


def test_pending_annotation_publishes_no_partial_metrics() -> None:
    packet = fixtures._packet()
    baseline = assess_deterministic_surface_baseline(fixtures._image(), packet)
    pending = GeometryRelationAnnotation(
        reviewer_id=packet.reviewer_id,
        proposed_family_reviews=[
            ProposedFamilyReview(proposed_family_id="global-p01")
        ],
    )

    report = compare_deterministic_surfaces_with_human(
        {packet.reviewer_id: packet},
        {packet.reviewer_id: pending},
        {packet.reviewer_id: baseline},
        quality_receipt=None,
        config=_config(expected_packet_count=1),
    )

    assert report.status == "waiting_for_human_annotations"
    assert report.records == []
    assert report.assessable_family_count == 0
    assert report.human_annotation_hash_closure_sha256 is None


def test_quality_receipt_strips_all_source_summary_fields() -> None:
    pilot = {
        "schema_version": "geometry-semantic-relation-pilot-audit-v1",
        "status": "complete",
        "packet_count": 36,
        "unique_source_count": 32,
        "source_key_opened": True,
        "origin_scoring_authorized": False,
        "quality": {
            "completed_unique_count": 30,
            "hidden_duplicate_group_count": 4,
            "completed_duplicate_pair_count": 4,
            "family_decision_agreement": 0.9,
            "surface_line_pair_agreement": 0.85,
            "gates": {gate: True for gate in REQUIRED_QUALITY_GATES},
            "passed": True,
        },
        "posthoc_source_summary": {
            "by_declared_label": {"generated": {"count": 16}},
            "by_archive": {"secret": {"count": 3}},
        },
    }

    receipt = extract_human_quality_receipt(pilot, input_audit_sha256="b" * 64)
    payload = receipt.model_dump(mode="json")

    assert receipt.passed is True
    assert payload["source_details_transferred"] is False
    assert "posthoc_source_summary" not in payload
    assert "label" not in str(payload).lower()
    assert "archive" not in str(payload).lower()


def test_quality_receipt_and_default_protocol_are_fail_closed() -> None:
    with pytest.raises(ValidationError, match="gates do not match"):
        HumanRelationQualityReceipt(
            input_audit_sha256="c" * 64,
            packet_count=36,
            unique_source_count=32,
            completed_unique_count=30,
            hidden_duplicate_group_count=4,
            completed_duplicate_pair_count=4,
            family_decision_agreement=0.9,
            surface_line_pair_agreement=0.9,
            gates={"unexpected": True},
            passed=True,
        )

    assert DEFAULT_SURFACE_COMPARISON_CONFIG.model_dump() == {
        "expected_packet_count": 36,
        "terminal_annotation_ratio_minimum": 1.0,
        "completed_packet_count_minimum": 24,
        "assessable_family_count_minimum": 48,
        "comparable_family_fraction_minimum": 0.75,
        "active_line_assignment_fraction_minimum": 0.70,
        "macro_same_surface_pair_retention_minimum": 0.70,
        "macro_different_surface_pair_separation_minimum": 0.70,
        "split_family_recall_minimum": 0.60,
        "non_split_family_specificity_minimum": 0.70,
    }
