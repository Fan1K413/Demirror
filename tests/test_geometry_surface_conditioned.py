from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from image_trust.geometry_ai.deterministic_surfaces import (
    assess_deterministic_surface_baseline,
)
from image_trust.geometry_ai.measurement_types import (
    GeometryFamilyV2,
    GeometryMeasurementV2Result,
    MergedGeometryLineV2,
)
from image_trust.geometry_ai.surface_conditioned import (
    SurfaceConditionedReplayAuthorization,
    assess_surface_conditioned_g1_g4,
    build_surface_replay_authorization,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "surface_conditioned_fixtures",
    REPOSITORY_ROOT / "tests/test_geometry_surface_comparison.py",
)
assert SPEC is not None and SPEC.loader is not None
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixtures
SPEC.loader.exec_module(fixtures)


def _measurement(packet):
    width, height = packet.canonical_size
    diagonal = math.hypot(width, height)
    merged = []
    for line in packet.lines:
        x1, y1 = line.start.x * width, line.start.y * height
        x2, y2 = line.end.x * width, line.end.y * height
        length = math.hypot(x2 - x1, y2 - y1)
        merged.append(
            MergedGeometryLineV2(
                line_id=line.line_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                length_px=length,
                length_normalized=length / diagonal,
                angle_rad=math.atan2(y2 - y1, x2 - x1) % math.pi,
                source_line_ids=[line.line_id],
                source_scale_ids=["global"],
                cross_scale_stability=line.cross_scale_stability,
            )
        )
    families = [
        GeometryFamilyV2(
            family_id=family.family_id,
            region_id=family.region_id,
            kind=family.kind,
            member_line_ids=family.member_line_ids,
            direction_rad=0.0,
            weighted_inlier_ratio=0.95,
            residual_p50_deg=0.2,
            residual_p90_deg=0.5,
            bootstrap_stability=0.9,
            stable=True,
        )
        for family in packet.family_proposals
    ]
    return GeometryMeasurementV2Result(
        status="measurable",
        summary="fixture",
        canonical_size=packet.canonical_size,
        applicability=1.0,
        merged_lines=merged,
        families=families,
    )


def _authorization(reviewer_id: str):
    return SurfaceConditionedReplayAuthorization(
        continuation_protocol_sha256="1" * 64,
        comparison_report_sha256="2" * 64,
        baseline_audit_sha256="3" * 64,
        human_annotation_hash_closure_sha256="4" * 64,
        quality_receipt_sha256="5" * 64,
        g1_g4_measurement_sha256="6" * 64,
        reviewer_ids=[reviewer_id],
    )


def test_surface_conditioned_replay_contains_only_g1_through_g4() -> None:
    packet = fixtures._packet_with_coherent_control()
    baseline = assess_deterministic_surface_baseline(fixtures.fixtures._image(), packet)
    measurement = _measurement(packet)
    rgb = np.asarray(fixtures.fixtures._image().convert("RGB"))

    result = assess_surface_conditioned_g1_g4(
        rgb,
        measurement,
        packet,
        baseline,
        _authorization(packet.reviewer_id),
    )

    assert result.status == "available"
    assert [check.check_id for check in result.checks] == ["G1", "G2", "G3", "G4"]
    assert all(check.origin_eligible is False for check in result.checks)
    assert len(result.conditioned_regions) == 2
    assert len(result.conditioned_families) == 3
    assert result.human_annotations_used is False
    assert result.origin_scoring_authorized is False


def test_replay_fails_closed_on_identity_or_authorization_drift() -> None:
    packet = fixtures._packet_with_coherent_control()
    baseline = assess_deterministic_surface_baseline(fixtures.fixtures._image(), packet)
    measurement = _measurement(packet)
    rgb = np.asarray(fixtures.fixtures._image().convert("RGB"))

    with pytest.raises(ValueError, match="not authorized"):
        assess_surface_conditioned_g1_g4(
            rgb,
            measurement,
            packet,
            baseline,
            _authorization("another-reviewer"),
        )
    with pytest.raises(ValueError, match="image size"):
        assess_surface_conditioned_g1_g4(
            rgb[:64, :64],
            measurement,
            packet,
            baseline,
            _authorization(packet.reviewer_id),
        )


def test_not_measurable_input_publishes_no_partial_replay() -> None:
    packet = fixtures._packet_with_coherent_control()
    baseline = assess_deterministic_surface_baseline(fixtures.fixtures._image(), packet)
    measurement = _measurement(packet).model_copy(update={"status": "not_applicable"})
    rgb = np.asarray(fixtures.fixtures._image().convert("RGB"))

    result = assess_surface_conditioned_g1_g4(
        rgb,
        measurement,
        packet,
        baseline,
        _authorization(packet.reviewer_id),
    )

    assert result.status == "not_applicable"
    assert result.checks == []
    assert result.conditioned_regions == []


def test_authorization_requires_complete_passing_source_neutral_comparison() -> None:
    comparison = {
        "schema_version": "geometry-surface-human-comparison-v1",
        "status": "complete",
        "passed": True,
        "decision": "eligible_for_surface_conditioned_g1_g4_replay_only",
        "gates": {"all": True},
        "source_key_opened": False,
        "source_labels_used": False,
        "ai_assisted_annotations_used": False,
        "origin_scoring_authorized": False,
        "web_integration_authorized": False,
        "replay_reviewer_ids": ["grr-a", "grr-b"],
        "continuation_protocol_sha256": "1" * 64,
        "baseline_audit_sha256": "2" * 64,
        "human_annotation_hash_closure_sha256": "3" * 64,
        "quality_receipt_sha256": "4" * 64,
    }

    authorization = build_surface_replay_authorization(
        comparison,
        comparison_report_sha256="5" * 64,
        g1_g4_measurement_sha256="6" * 64,
    )
    assert authorization.reviewer_ids == ["grr-a", "grr-b"]

    comparison["source_labels_used"] = True
    with pytest.raises(ValueError, match="source-neutral"):
        build_surface_replay_authorization(
            comparison,
            comparison_report_sha256="5" * 64,
            g1_g4_measurement_sha256="6" * 64,
        )
