"""Review overlays for geometry-v2 regions, families, and consistency checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_trust.geometry_ai.measurement_types import (
    GeometryCheckV2,
    GeometryFamilyV2,
    MergedGeometryLineV2,
    StructureRegionV2,
)


COLORS = (
    (58, 119, 175),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (230, 159, 0),
    (86, 180, 233),
    (0, 114, 178),
    (240, 228, 66),
)


def write_geometry_v2_overlays(
    canonical_rgb: np.ndarray,
    lines: list[MergedGeometryLineV2],
    regions: list[StructureRegionV2],
    families: list[GeometryFamilyV2],
    checks: list[GeometryCheckV2],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_regions(canonical_rgb, regions, output_dir / "regions_overlay.png")
    _write_families(canonical_rgb, lines, families, output_dir / "families_overlay.png")
    _write_consistency(
        canonical_rgb,
        lines,
        regions,
        checks,
        output_dir / "consistency_overlay.png",
    )
    _write_spacing(
        canonical_rgb,
        lines,
        families,
        checks,
        output_dir / "repeat_spacing_overlay.png",
    )


def _write_regions(
    rgb: np.ndarray, regions: list[StructureRegionV2], path: Path
) -> None:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    for index, region in enumerate(regions):
        box = region.canonical_box
        color = COLORS[index % len(COLORS)]
        draw.rectangle(
            (box.x, box.y, box.x + box.width - 1, box.y + box.height - 1),
            outline=color,
            width=3,
        )
        draw.rectangle((box.x, box.y, box.x + 88, box.y + 18), fill=(0, 0, 0))
        draw.text((box.x + 4, box.y + 3), f"{region.region_id}  L={region.line_count}", fill=color)
    _banner(draw, image.width, "GEOMETRY V2 STRUCTURE REGIONS - MEASUREMENT ONLY")
    image.save(path)


def _write_families(
    rgb: np.ndarray,
    lines: list[MergedGeometryLineV2],
    families: list[GeometryFamilyV2],
    path: Path,
) -> None:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    line_by_id = {line.line_id: line for line in lines}
    stable = [family for family in families if family.stable and family.region_id != "global"]
    for index, family in enumerate(stable):
        color = COLORS[index % len(COLORS)]
        for line_id in family.member_line_ids:
            line = line_by_id.get(line_id)
            if line is not None:
                draw.line((line.x1, line.y1, line.x2, line.y2), fill=color, width=3)
        if family.vanishing_point is not None:
            point = family.vanishing_point
            if 0 <= point.x < image.width and 0 <= point.y < image.height:
                draw.ellipse((point.x - 5, point.y - 5, point.x + 5, point.y + 5), outline=color, width=2)
    _banner(draw, image.width, "GEOMETRY V2 LOCAL FAMILIES - COLORS ARE REGION-SCOPED")
    image.save(path)


def _write_consistency(
    rgb: np.ndarray,
    lines: list[MergedGeometryLineV2],
    regions: list[StructureRegionV2],
    checks: list[GeometryCheckV2],
    path: Path,
) -> None:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    line_by_id = {line.line_id: line for line in lines}
    region_by_id = {region.region_id: region for region in regions}
    findings = [finding for check in checks if check.check_id in {"G1", "G2", "G4"} for finding in check.findings]
    for finding in findings:
        for region_id in finding.region_ids:
            region = region_by_id.get(region_id)
            if region is not None:
                box = region.canonical_box
                draw.rectangle(
                    (box.x, box.y, box.x + box.width - 1, box.y + box.height - 1),
                    outline=(240, 160, 40),
                    width=2,
                )
        for line_id in finding.line_ids:
            line = line_by_id.get(line_id)
            if line is not None:
                draw.line((line.x1, line.y1, line.x2, line.y2), fill=(215, 50, 50), width=4)
    _banner(
        draw,
        image.width,
        f"GEOMETRY V2 CONSISTENCY CANDIDATES: {len(findings)} - NOT SOURCE EVIDENCE",
    )
    image.save(path)


def _write_spacing(
    rgb: np.ndarray,
    lines: list[MergedGeometryLineV2],
    families: list[GeometryFamilyV2],
    checks: list[GeometryCheckV2],
    path: Path,
) -> None:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    line_by_id = {line.line_id: line for line in lines}
    family_by_id = {family.family_id: family for family in families}
    findings = [finding for check in checks if check.check_id == "G3" for finding in check.findings]
    for finding in findings:
        for family_id in finding.family_ids:
            family = family_by_id.get(family_id)
            if family is None:
                continue
            for line_id in family.member_line_ids:
                line = line_by_id.get(line_id)
                if line is not None:
                    draw.line((line.x1, line.y1, line.x2, line.y2), fill=(190, 80, 210), width=4)
    _banner(draw, image.width, f"GEOMETRY V2 REPEAT-SPACING CANDIDATES: {len(findings)}")
    image.save(path)


def _banner(draw: ImageDraw.ImageDraw, width: int, text: str) -> None:
    draw.rectangle((0, 0, width, 24), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
