from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from image_trust.geometry_ai.consistency_v2 import (
    merge_multiscale_lines,
    propose_structure_regions,
)
from image_trust.geometry_ai.measurement_v2 import assess_geometry_measurement_v2
from image_trust.geometry_ai.measurement_types import (
    CanonicalBox,
    GeometryLineV2,
    GeometryScaleV2,
    MergedGeometryLineV2,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _save_structured_image(path: Path) -> None:
    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    for x in range(100, 1150, 140):
        draw.line((x, 60, x, 840), fill="black", width=5)
    for y in range(120, 850, 120):
        draw.line((60, y, 1140, y), fill="black", width=5)
    image.save(path)


def test_measurement_v2_returns_canonical_two_scale_line_evidence(tmp_path: Path) -> None:
    input_path = tmp_path / "structured.png"
    _save_structured_image(input_path)

    result = assess_geometry_measurement_v2(input_path)

    assert result.status == "measurable"
    assert result.canonical_size == (1200, 900)
    assert result.global_scale is not None
    assert result.global_scale.analysis_size == (960, 720)
    assert len(result.local_scales) == 9
    assert all(gate.passed for gate in result.gates)
    assert result.global_scale.line_count >= 12
    assert all(0.0 <= line.x1 <= 1200.0 for line in result.global_scale.lines)
    assert all(0.0 <= line.y2 <= 900.0 for line in result.global_scale.lines)
    assert "geometry_measurement_v2_has_no_source_or_ai_decision" in result.limitations
    assert {check.check_id for check in result.checks} == {"G1", "G2", "G3", "G4", "G5"}


def test_measurement_v2_writes_review_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "structured.png"
    output_dir = tmp_path / "measurement"
    _save_structured_image(input_path)

    result = assess_geometry_measurement_v2(input_path, output_dir=output_dir)

    assert result.artifacts.result_json == "geometry_measurement_v2.json"
    for name in (
        "geometry_measurement_v2.json",
        "regions_overlay.png",
        "families_overlay.png",
        "consistency_overlay.png",
        "repeat_spacing_overlay.png",
    ):
        assert (output_dir / name).is_file()


def test_measurement_v2_rejects_a_low_structure_image(tmp_path: Path) -> None:
    input_path = tmp_path / "blank.png"
    Image.new("RGB", (800, 600), "white").save(input_path)

    result = assess_geometry_measurement_v2(input_path)

    assert result.status == "not_applicable"
    assert result.global_scale is not None
    assert result.global_scale.line_count == 0
    assert not all(gate.passed for gate in result.gates)
    assert "geometry_measurement_v2_insufficient_structural_support" in result.limitations


def test_cross_scale_duplicate_lines_merge_with_stability() -> None:
    crop = CanonicalBox(x=0, y=0, width=400, height=300)
    global_line = GeometryLineV2(
        line_id="global-l000",
        x1=20,
        y1=80,
        x2=360,
        y2=80,
        length_px=340,
        length_normalized=0.68,
    )
    local_line = GeometryLineV2(
        line_id="tile-0-0-l000",
        x1=22,
        y1=81,
        x2=358,
        y2=81,
        length_px=336,
        length_normalized=0.672,
    )
    global_scale = GeometryScaleV2(
        scale_id="global",
        scope="global",
        canonical_crop=crop,
        analysis_size=(400, 300),
        line_count=1,
        normalized_total_length=0.68,
        lines=[global_line],
    )
    local_scale = GeometryScaleV2(
        scale_id="tile-0-0",
        scope="local_tile",
        canonical_crop=crop,
        analysis_size=(400, 300),
        line_count=1,
        normalized_total_length=0.672,
        lines=[local_line],
    )

    merged = merge_multiscale_lines(global_scale, [local_scale], (400, 300))

    assert len(merged) == 1
    assert merged[0].source_scale_ids == ["global", "tile-0-0"]
    assert merged[0].cross_scale_stability >= 0.85


def test_separated_structures_remain_separate_regions() -> None:
    lines: list[MergedGeometryLineV2] = []
    for group, x_offset in (("a", 15), ("b", 315)):
        for index, y in enumerate((10, 20, 30, 40)):
            lines.append(
                MergedGeometryLineV2(
                    line_id=f"{group}{index}",
                    x1=x_offset,
                    y1=y,
                    x2=x_offset + 65,
                    y2=y,
                    length_px=65,
                    length_normalized=0.13,
                    angle_rad=0.0,
                    source_line_ids=[f"{group}{index}"],
                    source_scale_ids=["global", f"tile-{group}"],
                    cross_scale_stability=0.85,
                )
            )

    regions = propose_structure_regions(lines, (400, 200))

    assert len(regions) == 2
    assert all(region.line_count == 4 for region in regions)


def test_controlled_parallel_outlier_is_separated_from_clean_control() -> None:
    fixtures = REPOSITORY_ROOT / "data" / "p0_geometry_anomaly_v1" / "fixtures"

    clean = assess_geometry_measurement_v2(fixtures / "fixture-parallel_family-clean-01.png")
    anomaly = assess_geometry_measurement_v2(
        fixtures / "fixture-parallel_family-anomaly-01.png"
    )
    clean_g1 = next(check for check in clean.checks if check.check_id == "G1")
    anomaly_g1 = next(check for check in anomaly.checks if check.check_id == "G1")

    assert clean_g1.anomaly_score == 0.0
    assert anomaly_g1.anomaly_score is not None
    assert anomaly_g1.anomaly_score >= 0.50
    assert any(len(finding.family_ids) == 2 for finding in anomaly_g1.findings)


def test_measurement_v2_publishes_completed_checks_in_order(tmp_path: Path) -> None:
    input_path = tmp_path / "structured.png"
    _save_structured_image(input_path)
    snapshots: list[list[str]] = []
    events: list[str] = []

    result = assess_geometry_measurement_v2(
        input_path,
        check_started_callback=lambda check_id: events.append(f"start:{check_id}"),
        check_callback=lambda checks: (
            snapshots.append([check.check_id for check in checks]),
            events.append(f"done:{checks[-1].check_id}"),
        ),
    )

    assert snapshots == [
        ["G1"],
        ["G1", "G2"],
        ["G1", "G2", "G3"],
        ["G1", "G2", "G3", "G4"],
    ]
    assert events == [
        "start:G1", "done:G1",
        "start:G2", "done:G2",
        "start:G3", "done:G3",
        "start:G4", "done:G4",
    ]
    assert [check.check_id for check in result.checks] == ["G1", "G2", "G3", "G4", "G5"]
