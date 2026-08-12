from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from image_trust.geometry_ai.deterministic_surfaces import (
    DEFAULT_DETERMINISTIC_SURFACE_CONFIG,
    DeterministicSurfaceBaselineResult,
    DeterministicSurfaceConfig,
    assess_deterministic_surface_baseline,
    export_deterministic_surface_diagnostics,
)
from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationReviewPacket,
    NormalizedBox,
    NormalizedPoint,
    ReviewAssets,
    ReviewFamilyProposal,
    ReviewLine,
    ReviewRegionProposal,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "research/records/2026-08-12/geometry/geometry_deterministic_surface_baseline_protocol_v1.json"
)
SPEC = importlib.util.spec_from_file_location(
    "geometry_surface_baseline_runner",
    REPOSITORY_ROOT / "scripts/run_geometry_surface_baseline.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _image() -> Image.Image:
    image = Image.new("RGB", (128, 128), (200, 70, 55))
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 0, 127, 127), fill=(45, 105, 190))
    return image


def _line(
    line_id: str,
    start: tuple[float, float],
    end: tuple[float, float],
    stability: float = 0.90,
) -> ReviewLine:
    return ReviewLine(
        line_id=line_id,
        start=NormalizedPoint(x=start[0], y=start[1]),
        end=NormalizedPoint(x=end[0], y=end[1]),
        cross_scale_stability=stability,
    )


def _packet(
    *,
    image_asset: str = "image.png",
    include_boundary: bool = False,
    low_stability_last: bool = False,
) -> GeometryRelationReviewPacket:
    lines = [
        _line("m0001", (0.08, 0.25), (0.42, 0.25)),
        _line("m0002", (0.08, 0.35), (0.42, 0.35)),
        _line("m0003", (0.58, 0.65), (0.92, 0.65)),
        _line(
            "m0004",
            (0.58, 0.75),
            (0.92, 0.75),
            0.30 if low_stability_last else 0.90,
        ),
    ]
    if include_boundary:
        lines.append(_line("m0005", (0.50, 0.10), (0.50, 0.90)))
    line_ids = [line.line_id for line in lines]
    return GeometryRelationReviewPacket(
        purpose="source-neutral deterministic fixture",
        reviewer_id="grr-deterministic-test",
        canonical_size=(128, 128),
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
        lines=lines,
        region_proposals=[
            ReviewRegionProposal(
                region_id="r001",
                box=NormalizedBox(x=0.0, y=0.0, width=1.0, height=1.0),
                line_ids=line_ids,
                status="usable",
            )
        ],
        family_proposals=[
            ReviewFamilyProposal(
                family_id="global-p01",
                region_id="global",
                kind="parallel",
                member_line_ids=["m0001", "m0002", "m0003", "m0004"],
                overlay_asset="global_families_overlay",
                detail_overlay="family_details/global-p01.png",
                overlay_color_rgb=(58, 119, 175),
                priority_reason="stable_control",
                maximum_finding_severity=0.0,
            )
        ],
        instructions=["geometry only"],
    )


def test_default_configuration_matches_the_pre_registered_protocol() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    registered = protocol["fixed_configuration"]
    observed = json.loads(
        DEFAULT_DETERMINISTIC_SURFACE_CONFIG.model_dump_json()
    )

    maximum_packet_lines = observed.pop("maximum_packet_lines")
    assert observed == registered
    assert (
        maximum_packet_lines
        == protocol["resource_boundary"]["maximum_packet_lines"]
    )
    assert protocol["origin_scoring_authorized"] is False


def test_baseline_splits_one_direction_family_across_two_appearance_surfaces() -> None:
    result = assess_deterministic_surface_baseline(_image(), _packet())

    assert result.status == "available"
    assert result.source_labels_used is False
    assert result.origin_scoring_authorized is False
    assert result.web_integration_authorized is False
    assert len(result.appearance_components) == 2
    assert len(result.surface_candidates) == 2
    partition = result.family_partitions[0]
    assert partition.partition_status == "split_candidate"
    assert [subfamily.member_line_ids for subfamily in partition.surface_subfamilies] == [
        ["m0001", "m0002"],
        ["m0003", "m0004"],
    ]
    assert partition.unassigned_line_ids == []
    assert "annotation" not in inspect.signature(
        assess_deterministic_surface_baseline
    ).parameters


def test_boundary_line_can_vote_for_two_appearance_components() -> None:
    result = assess_deterministic_surface_baseline(
        _image(),
        _packet(include_boundary=True),
    )

    boundary = next(
        assignment
        for assignment in result.line_assignments
        if assignment.line_id == "m0005"
    )
    assert len(boundary.appearance_component_ids) == 2


def test_low_stability_member_is_explicitly_unassigned() -> None:
    result = assess_deterministic_surface_baseline(
        _image(),
        _packet(low_stability_last=True),
    )

    assignment = next(
        item for item in result.line_assignments if item.line_id == "m0004"
    )
    assert assignment.exclusion_reason == "low_cross_scale_stability"
    assert result.family_partitions[0].unassigned_line_ids == ["m0003", "m0004"]
    assert result.family_partitions[0].partition_status == "single_surface_candidate"


def test_resource_cap_returns_no_partial_relationship_result() -> None:
    config = DEFAULT_DETERMINISTIC_SURFACE_CONFIG.model_copy(
        update={"maximum_packet_lines": 3}
    )

    result = assess_deterministic_surface_baseline(_image(), _packet(), config)

    assert result.status == "not_applicable"
    assert result.line_assignments == []
    assert result.surface_candidates == []
    assert result.family_partitions == []
    assert result.accepted_line_link_count == 0


def test_baseline_is_deterministic_and_export_is_byte_stable(tmp_path: Path) -> None:
    first_result, first_manifest = export_deterministic_surface_diagnostics(
        _image(),
        _packet(),
        tmp_path / "first",
    )
    second_result, second_manifest = export_deterministic_surface_diagnostics(
        _image(),
        _packet(),
        tmp_path / "second",
    )

    assert first_result.model_dump() == second_result.model_dump()
    assert first_manifest == second_manifest
    assert (
        tmp_path / "first" / first_manifest.result_json
    ).read_bytes() == (
        tmp_path / "second" / second_manifest.result_json
    ).read_bytes()
    with Image.open(
        tmp_path / "first" / first_manifest.surface_candidates_overlay
    ) as overlay:
        assert overlay.size == (128, 128)
    with Image.open(
        tmp_path / "first" / first_manifest.family_partitions_overlay
    ) as comparison:
        assert comparison.size == (258, 128)
    assert not list(tmp_path.rglob("*.tmp"))


def test_result_contract_rejects_dropped_family_membership() -> None:
    result = assess_deterministic_surface_baseline(_image(), _packet())
    payload = result.model_dump(mode="json")
    payload["family_partitions"][0]["surface_subfamilies"][0][
        "member_line_ids"
    ].pop()
    payload["family_partitions"][0]["surface_subfamilies"][0][
        "usability"
    ] = "insufficient_members"

    with pytest.raises(ValidationError, match="member closure"):
        DeterministicSurfaceBaselineResult.model_validate(payload)


def test_result_contract_rejects_missing_or_inconsistent_exclusion_reason() -> None:
    result = assess_deterministic_surface_baseline(
        _image(),
        _packet(low_stability_last=True),
    )
    payload = result.model_dump(mode="json")
    assignment = next(
        item for item in payload["line_assignments"] if item["line_id"] == "m0004"
    )
    assignment["exclusion_reason"] = None

    with pytest.raises(ValidationError, match="exclusion reason"):
        DeterministicSurfaceBaselineResult.model_validate(payload)


def test_image_mismatch_and_packet_reference_drift_fail_closed() -> None:
    with pytest.raises(ValueError, match="does not match"):
        assess_deterministic_surface_baseline(
            Image.new("RGB", (64, 64), "white"),
            _packet(),
        )

    packet = _packet().model_copy(
        update={
            "region_proposals": [
                ReviewRegionProposal(
                    region_id="r001",
                    box=NormalizedBox(x=0.0, y=0.0, width=1.0, height=1.0),
                    line_ids=["unknown-line"],
                    status="usable",
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="unknown values"):
        assess_deterministic_surface_baseline(_image(), packet)


def test_cli_reads_only_the_packet_declared_image(tmp_path: Path) -> None:
    packet_dir = tmp_path / "blind" / "packets" / "grr-test"
    packet_dir.mkdir(parents=True)
    packet_path = packet_dir / "review_packet.json"
    packet_path.write_text(_packet().model_dump_json(indent=2), encoding="utf-8")
    _image().save(packet_dir / "image.png")

    manifest = runner.run_packet_baseline(packet_path, tmp_path / "baseline")

    assert manifest.surface_candidate_count == 2
    assert (tmp_path / "baseline" / manifest.result_json).is_file()


def test_cli_rejects_packet_asset_traversal_and_posthoc_output(tmp_path: Path) -> None:
    packet_dir = tmp_path / "blind" / "packets" / "grr-test"
    packet_dir.mkdir(parents=True)
    packet_path = packet_dir / "review_packet.json"
    packet_path.write_text(
        _packet(image_asset="../image.png").model_dump_json(),
        encoding="utf-8",
    )
    _image().save(packet_dir.parent / "image.png")

    with pytest.raises(ValueError, match="escapes"):
        runner.run_packet_baseline(packet_path, tmp_path / "baseline")

    posthoc = tmp_path / "posthoc"
    posthoc.mkdir()
    with pytest.raises(ValueError, match="posthoc"):
        runner.run_packet_baseline(packet_path, posthoc)


def test_configuration_rejects_unregistered_weight_or_sample_shapes() -> None:
    with pytest.raises(ValidationError, match="sum to one"):
        DeterministicSurfaceConfig(
            line_affinity_weights={
                "midpoint": 0.5,
                "endpoint": 0.5,
                "axis_relation": 0.5,
                "appearance": 0.5,
                "shared_region": 0.5,
                "cross_scale_stability": 0.5,
            }
        )
    with pytest.raises(ValidationError, match="unique and ascending"):
        DeterministicSurfaceConfig(line_side_sample_positions=(0.5, 0.2))
