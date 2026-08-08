"""PNG overlays for line coverage and uncalibrated anomaly candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_trust.schemas import AnomalyCandidate, LineRecord, OverlayConfig, VPFamily


FAMILY_COLORS: tuple[tuple[int, int, int], ...] = (
    (70, 150, 255),   # blue
    (255, 185, 70),   # amber
    (180, 105, 245),  # violet
    (65, 205, 165),   # teal
    (235, 95, 180),   # magenta
    (170, 215, 75),   # lime
    (70, 220, 235),   # cyan
    (245, 145, 75),   # orange
    (150, 235, 150),  # mint
    (190, 145, 75),   # ochre
    (150, 190, 255),  # periwinkle
    (225, 135, 235),  # orchid
)


def write_overlays(
    canonical_rgb: np.ndarray,
    lines: list[LineRecord],
    families: list[VPFamily],
    anomalies: list[AnomalyCandidate],
    config: OverlayConfig,
    lines_path: Path,
    anomalies_path: Path,
) -> None:
    line_image = Image.fromarray(canonical_rgb, mode="RGB")
    line_draw = ImageDraw.Draw(line_image)
    for line in lines:
        color = (40, 210, 100) if line.selected else (135, 135, 135)
        _draw_line(line_draw, line, color, config)
    _draw_legend(
        line_draw,
        line_image.width,
        [f"SELECTED LINES: {len(lines)}", "GREEN = SELECTED LINE SEGMENT"],
    )
    line_image.save(lines_path)

    anomaly_image = Image.fromarray(canonical_rgb, mode="RGB")
    anomaly_draw = ImageDraw.Draw(anomaly_image)
    stable_families = [family for family in families if family.stable]
    family_colors = {
        family.family_id: FAMILY_COLORS[index % len(FAMILY_COLORS)]
        for index, family in enumerate(stable_families)
    }
    line_colors = {
        line_id: family_colors[family.family_id]
        for family in stable_families
        for line_id in family.member_line_ids
    }
    # ``lines_overlay`` is the complete coverage diagnostic.  This review
    # overlay deliberately omits unassigned lines so incidental texture (for
    # example, the small straight pieces that LSD finds on an arch) does not
    # visually compete with the actual family/candidate decision.
    for line in lines:
        color = line_colors.get(line.line_id)
        if color is not None:
            _draw_line(anomaly_draw, line, color, config)
    line_index = {line.line_id: line for line in lines}
    for anomaly in anomalies:
        line = line_index.get(anomaly.line_id)
        if line is not None:
            _draw_line(anomaly_draw, line, (245, 70, 70), config)
    _draw_vanishing_points(anomaly_draw, anomaly_image, stable_families, family_colors)
    banner = (
        "GLOBAL/LOCAL GEOMETRIC FAMILIES + CANDIDATES - UNCALIBRATED"
        if anomalies
        else "GLOBAL/LOCAL GEOMETRIC FAMILIES - NO EXPLAINABLE ANOMALY CANDIDATE"
    )
    legend = [banner, "COLORED = GLOBAL VP OR LOCAL DIRECTION | RED = REVIEW CANDIDATE"]
    legend.extend(
        (_family_legend(family), family_colors[family.family_id])
        for family in stable_families
    )
    if not stable_families:
        legend.append(("NO STABLE REVIEW FAMILY", None))
    else:
        legend = [legend[0], (legend[1], None), *legend[2:]]
    _draw_legend(anomaly_draw, anomaly_image.width, legend)
    anomaly_image.save(anomalies_path)


def _draw_line(
    draw: ImageDraw.ImageDraw,
    line: LineRecord,
    color: tuple[int, int, int],
    config: OverlayConfig,
) -> None:
    draw.line(
        [(line.p1.x, line.p1.y), (line.p2.x, line.p2.y)],
        fill=color,
        width=config.line_width,
    )
    if config.draw_line_ids:
        draw.text((line.p1.x + 2, line.p1.y + 2), line.line_id, fill=color)


def _draw_vanishing_points(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    families: list[VPFamily],
    family_colors: dict[str, tuple[int, int, int]],
) -> None:
    for family in families:
        if family.vp is None:
            continue
        x, y = family.vp.x, family.vp.y
        if not (0 <= x < image.width and 0 <= y < image.height):
            continue
        radius = 5
        color = family_colors[family.family_id]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.text((x + radius + 2, y + radius + 2), family.family_id, fill=color)


def _family_legend(family: VPFamily) -> str:
    if family.scope == "global_vp":
        scope = "GLOBAL"
    elif family.scope == "local_image_plane_parallel":
        scope = "LOCAL DIRECTION"
    elif family.scope == "local_vp":
        scope = "LOCAL VP"
    else:
        scope = "PARALLEL"
    if family.vp_type == "finite" and family.vp is not None:
        return f"{family.family_id}: {scope} FINITE VP ({family.vp.x:.0f}, {family.vp.y:.0f})"
    if family.direction_analysis is not None:
        return f"{family.family_id}: {scope} DIRECTION {family.direction_analysis * 180.0 / np.pi:.1f} DEG"
    return f"{family.family_id}: {scope} VP TYPE UNAVAILABLE"


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    image_width: int,
    lines: list[str | tuple[str, tuple[int, int, int] | None]],
) -> None:
    line_height = 13
    height = 8 + line_height * len(lines)
    draw.rectangle((0, 0, image_width, height), fill=(0, 0, 0))
    for index, item in enumerate(lines):
        text, color = item if isinstance(item, tuple) else (item, None)
        y = 5 + index * line_height
        if color is not None:
            draw.rectangle((8, y + 2, 17, y + 11), fill=color)
            draw.text((22, y), text, fill=(255, 255, 255))
        else:
            draw.text((8, y), text, fill=(255, 255, 255))
