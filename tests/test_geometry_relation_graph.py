from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
    NormalizedBox,
    NormalizedPoint,
    ProposedFamilyReview,
    ReviewAssets,
    ReviewFamilyProposal,
    ReviewLine,
    ReviewRegionProposal,
    SemanticSurface,
)
from image_trust.geometry_ai.relation_graph import (
    GeometryRelationGraph,
    build_relation_graph,
    export_relation_graph_diagnostics,
)
from image_trust.geometry_ai.relation_validation import (
    validate_annotation_semantic_closure,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "geometry_relation_graph_exporter",
    REPOSITORY_ROOT / "scripts/export_geometry_relation_graph.py",
)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def _packet(image_asset: str = "image.png") -> GeometryRelationReviewPacket:
    line_coordinates = {
        "m0001": ((0.10, 0.20), (0.90, 0.20)),
        "m0002": ((0.10, 0.35), (0.90, 0.35)),
        "m0003": ((0.10, 0.65), (0.90, 0.65)),
        "m0004": ((0.10, 0.80), (0.90, 0.80)),
    }
    return GeometryRelationReviewPacket(
        purpose="source-neutral relation review",
        reviewer_id="grr-test",
        canonical_size=(100, 100),
        assets=ReviewAssets(
            image=image_asset,
            line_ids_overlay="line_ids_overlay.png",
            regions_overlay="regions_overlay.png",
            local_families_overlay="local_families_overlay.png",
            global_families_overlay="global_families_overlay.png",
            consistency_overlay="consistency_overlay.png",
            repeat_spacing_overlay="repeat_spacing_overlay.png",
            measurement="geometry_measurement_v2.json",
        ),
        lines=[
            ReviewLine(
                line_id=line_id,
                start=NormalizedPoint(x=start[0], y=start[1]),
                end=NormalizedPoint(x=end[0], y=end[1]),
                cross_scale_stability=0.85,
            )
            for line_id, (start, end) in line_coordinates.items()
        ],
        region_proposals=[
            ReviewRegionProposal(
                region_id="r001",
                box=NormalizedBox(x=0.0, y=0.0, width=1.0, height=1.0),
                line_ids=list(line_coordinates),
                status="usable",
            )
        ],
        family_proposals=[
            ReviewFamilyProposal(
                family_id="global-p01",
                region_id="global",
                kind="parallel",
                member_line_ids=list(line_coordinates),
                overlay_asset="global_families_overlay",
                detail_overlay="family_details/global-p01.png",
                overlay_color_rgb=(58, 119, 175),
                priority_reason="stable_control",
                maximum_finding_severity=0.0,
            )
        ],
        instructions=["review geometry only"],
    )


def _surface(
    surface_id: str,
    line_ids: list[str],
    y0: float,
) -> SemanticSurface:
    return SemanticSurface(
        surface_id=surface_id,
        surface_kind="roof",
        polygon_normalized=[
            NormalizedPoint(x=0.05, y=y0),
            NormalizedPoint(x=0.95, y=y0),
            NormalizedPoint(x=0.95, y=min(0.98, y0 + 0.30)),
            NormalizedPoint(x=0.05, y=min(0.98, y0 + 0.30)),
        ],
        line_ids=line_ids,
        visibility="clear",
    )


def _split_annotation() -> GeometryRelationAnnotation:
    return GeometryRelationAnnotation(
        reviewer_id="grr-test",
        status="completed",
        surfaces=[
            _surface("surface-a", ["m0001", "m0002"], 0.10),
            _surface("surface-b", ["m0002", "m0003"], 0.55),
        ],
        proposed_family_reviews=[
            ProposedFamilyReview(
                proposed_family_id="global-p01",
                verdict="split_across_surfaces",
                surface_ids=["surface-a", "surface-b"],
                outlier_line_ids=["m0004"],
            )
        ],
    )


def test_relation_graph_splits_families_and_preserves_shared_boundaries() -> None:
    graph = build_relation_graph(_packet(), _split_annotation())

    assert graph.source_labels_used is False
    assert graph.origin_scoring_authorized is False
    assert [family.member_line_ids for family in graph.surface_conditioned_families] == [
        ["m0001", "m0002"],
        ["m0002", "m0003"],
    ]
    shared_edges = [
        edge for edge in graph.line_on_surface if edge.line_id == "m0002"
    ]
    assert len(shared_edges) == 2
    assert all(edge.role == "shared_boundary_candidate" for edge in shared_edges)
    pairs = {
        (edge.first_line_id, edge.second_line_id): edge.shared_surface_ids
        for edge in graph.line_pair_same_surface
    }
    assert pairs == {
        ("m0001", "m0002"): ["surface-a"],
        ("m0002", "m0003"): ["surface-b"],
    }
    assert graph.excluded_family_members[0].model_dump() == {
        "original_family_id": "global-p01",
        "line_id": "m0004",
        "reason": "explicit_outlier",
    }


def test_relation_graph_is_deterministic_and_has_closed_family_edges() -> None:
    first = build_relation_graph(_packet(), _split_annotation())
    second = build_relation_graph(_packet(), _split_annotation())

    assert first.model_dump() == second.model_dump()
    assert {
        edge.conditioned_family_id for edge in first.family_on_surface
    } == {
        family.conditioned_family_id
        for family in first.surface_conditioned_families
    }
    assert all(
        family.usability == "usable"
        for family in first.surface_conditioned_families
    )


def test_unassessable_review_preserves_all_members_as_unresolved() -> None:
    annotation = GeometryRelationAnnotation(
        reviewer_id="grr-test",
        status="unassessable",
        assessability_reason="no stable visible surface",
        proposed_family_reviews=[
            ProposedFamilyReview(
                proposed_family_id="global-p01",
                verdict="unassessable",
            )
        ],
    )

    graph = build_relation_graph(_packet(), annotation)

    assert graph.surfaces == []
    assert graph.surface_conditioned_families == []
    assert [item.line_id for item in graph.excluded_family_members] == [
        "m0001",
        "m0002",
        "m0003",
        "m0004",
    ]
    assert {
        item.reason for item in graph.excluded_family_members
    } == {"review_unassessable"}


def test_graph_rejects_pending_or_semantically_open_annotation() -> None:
    pending = GeometryRelationAnnotation(
        reviewer_id="grr-test",
        proposed_family_reviews=[
            ProposedFamilyReview(proposed_family_id="global-p01")
        ],
    )
    with pytest.raises(ValueError, match="finalized"):
        build_relation_graph(_packet(), pending)

    annotation = _split_annotation().model_copy(
        update={
            "surfaces": [
                _surface("surface-a", ["m0001"], 0.10),
                _surface("surface-b", ["m0002"], 0.55),
            ]
        }
    )
    with pytest.raises(ValueError, match="unexplained by its reviewed surfaces"):
        validate_annotation_semantic_closure(_packet(), annotation)


def test_graph_contract_rejects_dropped_membership_edge() -> None:
    graph = build_relation_graph(_packet(), _split_annotation())
    payload = graph.model_dump(mode="json")
    payload["line_on_surface"].pop()

    with pytest.raises(ValidationError, match="do not close"):
        GeometryRelationGraph.model_validate(payload)

    payload = graph.model_dump(mode="json")
    payload["line_on_surface"][0]["role"] = "shared_boundary_candidate"
    with pytest.raises(ValidationError, match="membership degree"):
        GeometryRelationGraph.model_validate(payload)

    payload = graph.model_dump(mode="json")
    payload["lines"][0]["region_ids"] = []
    with pytest.raises(ValidationError, match="region memberships"):
        GeometryRelationGraph.model_validate(payload)

    payload = graph.model_dump(mode="json")
    payload["original_families"][0]["region_id"] = "missing-region"
    with pytest.raises(ValidationError, match="unknown region"):
        GeometryRelationGraph.model_validate(payload)


def test_semantic_closure_rejects_duplicate_outlier_members() -> None:
    annotation = _split_annotation().model_copy(
        update={
            "proposed_family_reviews": [
                ProposedFamilyReview(
                    proposed_family_id="global-p01",
                    verdict="split_across_surfaces",
                    surface_ids=["surface-a", "surface-b"],
                    outlier_line_ids=["m0004", "m0004"],
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="outlier_line_ids must be unique"):
        validate_annotation_semantic_closure(_packet(), annotation)


def test_diagnostic_export_writes_graph_and_two_aligned_overlays(tmp_path: Path) -> None:
    graph, manifest = export_relation_graph_diagnostics(
        Image.new("RGB", (100, 100), "white"),
        _packet(),
        _split_annotation(),
        tmp_path,
    )

    assert manifest.conditioned_family_count == 2
    assert manifest.region_count == 1
    assert (tmp_path / manifest.graph_json).is_file()
    assert (tmp_path / manifest.surface_membership_overlay).is_file()
    assert (tmp_path / manifest.family_comparison_overlay).is_file()
    assert (tmp_path / "geometry_relation_graph_artifacts.json").is_file()
    with Image.open(tmp_path / manifest.surface_membership_overlay) as overlay:
        assert overlay.size == (100, 100)
    with Image.open(tmp_path / manifest.family_comparison_overlay) as comparison:
        assert comparison.size == (202, 100)
    payload = json.loads((tmp_path / manifest.graph_json).read_text(encoding="utf-8"))
    assert payload["source_labels_used"] is False
    assert payload["origin_scoring_authorized"] is False
    assert payload["schema_version"] == graph.schema_version


def test_diagnostic_export_fails_before_writing_on_image_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="does not match"):
        export_relation_graph_diagnostics(
            Image.new("RGB", (50, 50), "white"),
            _packet(),
            _split_annotation(),
            output,
        )
    assert not output.exists()


def test_cli_exporter_reads_only_declared_packet_image(tmp_path: Path) -> None:
    packet_dir = tmp_path / "blind" / "packets" / "grr-test"
    packet_dir.mkdir(parents=True)
    packet_path = packet_dir / "review_packet.json"
    annotation_path = tmp_path / "agent_annotations" / "grr-test.json"
    annotation_path.parent.mkdir()
    packet_path.write_text(_packet().model_dump_json(indent=2), encoding="utf-8")
    annotation_path.write_text(
        _split_annotation().model_dump_json(indent=2), encoding="utf-8"
    )
    Image.new("RGB", (100, 100), "white").save(packet_dir / "image.png")

    manifest = exporter.export_packet_graph(
        packet_path,
        annotation_path,
        tmp_path / "graphs",
    )

    assert manifest.graph_json == "geometry_relation_graph.json"
    assert (tmp_path / "graphs" / manifest.graph_json).is_file()


def test_cli_exporter_rejects_traversal_and_posthoc_paths(tmp_path: Path) -> None:
    packet_dir = tmp_path / "blind" / "packets" / "grr-test"
    packet_dir.mkdir(parents=True)
    escaped_packet = _packet("../image.png")
    packet_path = packet_dir / "review_packet.json"
    packet_path.write_text(escaped_packet.model_dump_json(), encoding="utf-8")
    annotation_path = packet_dir / "annotation.json"
    annotation_path.write_text(_split_annotation().model_dump_json(), encoding="utf-8")
    Image.new("RGB", (100, 100), "white").save(packet_dir.parent / "image.png")

    with pytest.raises(ValueError, match="escapes"):
        exporter.export_packet_graph(packet_path, annotation_path, tmp_path / "graphs")

    posthoc = tmp_path / "posthoc"
    posthoc.mkdir()
    with pytest.raises(ValueError, match="posthoc"):
        exporter.export_packet_graph(packet_path, annotation_path, posthoc)
