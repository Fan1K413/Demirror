from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from image_trust.geometry_ai.measurement_types import (
    CanonicalBox,
    GeometryCheckV2,
    GeometryFamilyV2,
    GeometryFindingV2,
    GeometryMeasurementV2Result,
    MergedGeometryLineV2,
    StructureRegionV2,
)
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    NormalizedPoint,
    ProposedFamilyReview,
    SemanticSurface,
    build_review_packet,
    surface_pair_signature,
    validate_annotation_against_packet,
    write_relation_review_overlays,
)


def _line(line_id: str, y: float) -> MergedGeometryLineV2:
    return MergedGeometryLineV2(
        line_id=line_id,
        x1=10.0,
        y1=y,
        x2=90.0,
        y2=y,
        length_px=80.0,
        length_normalized=0.8,
        angle_rad=0.0,
        source_line_ids=[f"global-{line_id}"],
        source_scale_ids=["global", "tile-0-0"],
        cross_scale_stability=0.85,
    )


def _measurement() -> GeometryMeasurementV2Result:
    lines = [_line("m0001", 20), _line("m0002", 30), _line("m0003", 65), _line("m0004", 75)]
    return GeometryMeasurementV2Result(
        status="measurable",
        summary="measurement",
        canonical_size=(100, 100),
        applicability=1.0,
        merged_lines=lines,
        regions=[
            StructureRegionV2(
                region_id="r001",
                canonical_box=CanonicalBox(x=0, y=0, width=100, height=100),
                cell_ids=["c0-0"],
                line_ids=[line.line_id for line in lines],
                line_count=4,
                normalized_line_support=3.2,
                orientation_entropy=0.0,
                status="usable",
            )
        ],
        families=[
            GeometryFamilyV2(
                family_id="global-p01",
                region_id="global",
                kind="parallel",
                member_line_ids=["m0001", "m0002"],
                direction_rad=0.0,
                weighted_inlier_ratio=0.9,
                residual_p50_deg=0.2,
                residual_p90_deg=0.5,
                bootstrap_stability=0.9,
                stable=True,
            ),
            GeometryFamilyV2(
                family_id="r001-p01",
                region_id="r001",
                kind="parallel",
                member_line_ids=["m0003", "m0004"],
                direction_rad=0.0,
                weighted_inlier_ratio=0.9,
                residual_p50_deg=0.2,
                residual_p90_deg=0.5,
                bootstrap_stability=0.9,
                stable=True,
            ),
        ],
    )


def _surface(surface_id: str, line_ids: list[str], y0: float) -> SemanticSurface:
    return SemanticSurface(
        surface_id=surface_id,
        surface_kind="roof",
        polygon_normalized=[
            NormalizedPoint(x=0.05, y=y0),
            NormalizedPoint(x=0.95, y=y0),
            NormalizedPoint(x=0.95, y=min(0.98, y0 + 0.25)),
            NormalizedPoint(x=0.05, y=min(0.98, y0 + 0.25)),
        ],
        line_ids=line_ids,
        visibility="clear",
    )


def test_review_packet_is_source_blind_and_separates_global_from_local_families() -> None:
    packet, annotation = build_review_packet("grr-test", _measurement())

    serialized = packet.model_dump_json()
    assert packet.source_label_visibility == "forbidden"
    assert "sample_id" not in serialized
    assert "generator" not in serialized
    assert [family.overlay_asset for family in packet.family_proposals] == [
        "global_families_overlay",
        "local_families_overlay",
    ]
    assert [family.detail_overlay for family in packet.family_proposals] == [
        "family_details/global-p01.png",
        "family_details/r001-p01.png",
    ]
    assert all(family.priority_reason == "stable_control" for family in packet.family_proposals)
    assert [review.proposed_family_id for review in annotation.proposed_family_reviews] == [
        "global-p01",
        "r001-p01",
    ]


def test_review_packet_caps_each_scope_and_prioritizes_check_findings() -> None:
    measurement = _measurement()
    families = []
    for scope, region_id in (("global", "global"), ("local", "r001")):
        for index in range(6):
            families.append(
                GeometryFamilyV2(
                    family_id=f"{scope}-p{index + 1:02d}",
                    region_id=region_id,
                    kind="parallel",
                    member_line_ids=["m0001", "m0002"],
                    direction_rad=0.0,
                    weighted_inlier_ratio=0.9,
                    residual_p50_deg=0.2,
                    residual_p90_deg=0.5,
                    bootstrap_stability=0.9,
                    stable=True,
                )
            )
    measurement = measurement.model_copy(
        update={
            "families": families,
            "checks": [
                GeometryCheckV2(
                    check_id="G1",
                    title="candidate",
                    status="available",
                    anomaly_score=0.8,
                    findings=[
                        GeometryFindingV2(
                            finding_id="finding-1",
                            check_id="G1",
                            family_ids=["global-p06"],
                            severity=0.8,
                            description="candidate",
                        )
                    ],
                )
            ],
        }
    )

    packet, _ = build_review_packet("grr-test", measurement)

    assert len(packet.family_proposals) == 8
    assert sum(family.region_id == "global" for family in packet.family_proposals) == 4
    assert sum(family.region_id != "global" for family in packet.family_proposals) == 4
    assert packet.family_proposals[0].family_id == "global-p06"
    assert packet.family_proposals[0].priority_reason == "check_finding"


def test_completed_annotation_validates_surface_aware_family_assignments() -> None:
    packet, _ = build_review_packet("grr-test", _measurement())
    annotation = GeometryRelationAnnotation(
        reviewer_id="grr-test",
        status="completed",
        surfaces=[
            _surface("s-roof-a", ["m0001", "m0002"], 0.10),
            _surface("s-roof-b", ["m0003", "m0004"], 0.60),
        ],
        proposed_family_reviews=[
            ProposedFamilyReview(
                proposed_family_id="global-p01",
                verdict="coherent_within_surface",
                surface_ids=["s-roof-a"],
            ),
            ProposedFamilyReview(
                proposed_family_id="r001-p01",
                verdict="coherent_within_surface",
                surface_ids=["s-roof-b"],
            ),
        ],
    )

    validate_annotation_against_packet(packet, annotation)


def test_annotation_rejects_source_fields_and_dangling_line_references() -> None:
    with pytest.raises(ValidationError):
        GeometryRelationAnnotation.model_validate(
            {
                "reviewer_id": "grr-test",
                "source_label": "ai_generated",
            }
        )

    packet, _ = build_review_packet("grr-test", _measurement())
    annotation = GeometryRelationAnnotation(
        reviewer_id="grr-test",
        status="completed",
        surfaces=[_surface("s-roof", ["unknown-line"], 0.10)],
        proposed_family_reviews=[
            ProposedFamilyReview(
                proposed_family_id=family.family_id,
                verdict="unassessable",
            )
            for family in packet.family_proposals
        ],
    )
    with pytest.raises(ValueError, match="unknown values"):
        validate_annotation_against_packet(packet, annotation)


def test_surface_pair_signature_is_invariant_to_surface_names() -> None:
    first = GeometryRelationAnnotation(
        reviewer_id="first",
        status="completed",
        surfaces=[
            _surface("white-circle", ["m0001", "m0002"], 0.10),
            _surface("black-circle", ["m0003", "m0004"], 0.60),
        ],
    )
    second = GeometryRelationAnnotation(
        reviewer_id="second",
        status="completed",
        surfaces=[
            _surface("surface-9", ["m0001", "m0002"], 0.10),
            _surface("surface-2", ["m0003", "m0004"], 0.60),
        ],
    )

    assert surface_pair_signature(first) == surface_pair_signature(second)
    assert surface_pair_signature(first)[("m0001", "m0002")] is True
    assert surface_pair_signature(first)[("m0001", "m0003")] is False


def test_relation_review_overlays_are_written_without_changing_measurement(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    measurement = _measurement()

    write_relation_review_overlays(image_path, measurement, tmp_path)

    assert (tmp_path / "line_ids_overlay.png").is_file()
    assert (tmp_path / "global_families_overlay.png").is_file()
    assert (tmp_path / "family_details" / "global-p01.png").is_file()
    assert (tmp_path / "family_details" / "r001-p01.png").is_file()
    assert measurement.merged_lines[0].line_id == "m0001"
