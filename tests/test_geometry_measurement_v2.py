from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from image_trust.geometry_ai.measurement_v2 import assess_geometry_measurement_v2


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


def test_measurement_v2_rejects_a_low_structure_image(tmp_path: Path) -> None:
    input_path = tmp_path / "blank.png"
    Image.new("RGB", (800, 600), "white").save(input_path)

    result = assess_geometry_measurement_v2(input_path)

    assert result.status == "not_applicable"
    assert result.global_scale is not None
    assert result.global_scale.line_count == 0
    assert not all(gate.passed for gate in result.gates)
    assert "geometry_measurement_v2_insufficient_structural_support" in result.limitations
