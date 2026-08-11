"""Local region, line-family, and source-neutral consistency measurements."""

from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

from image_trust.geometry.vanishing_points import fit_parallel_families, fit_vanishing_families
from image_trust.geometry_ai.measurement_types import (
    CanonicalBox,
    GeometryCheckV2,
    GeometryFamilyV2,
    GeometryFindingV2,
    GeometryLineV2,
    GeometryPointV2,
    GeometryScaleV2,
    MergedGeometryLineV2,
    StructureRegionV2,
)
from image_trust.schemas import LineRecord, Point, VPFamily, VanishingPointConfig
from image_trust.utils.coordinates import CoordinateTransform


REGION_GRID_SIZE = 4
MIN_REGION_LINES = 4
MERGE_ANGLE_DEG = 2.5


def merge_multiscale_lines(
    global_scale: GeometryScaleV2,
    local_scales: list[GeometryScaleV2],
    canonical_size: tuple[int, int],
) -> list[MergedGeometryLineV2]:
    """Merge only overlapping, collinear detections from different scales."""

    width, height = canonical_size
    diagonal = math.hypot(width, height)
    candidates = [*global_scale.lines, *(line for scale in local_scales for line in scale.lines)]
    scale_by_line = {
        line.line_id: scale.scale_id
        for scale in [global_scale, *local_scales]
        for line in scale.lines
    }
    candidates.sort(key=lambda line: (-line.length_px, line.line_id))
    groups: list[dict[str, object]] = []
    for line in candidates:
        matched: dict[str, object] | None = None
        for group in groups:
            representative = group["representative"]
            assert isinstance(representative, GeometryLineV2)
            if _same_line_support(representative, line, diagonal):
                matched = group
                break
        if matched is None:
            groups.append(
                {
                    "representative": line,
                    "source_line_ids": [line.line_id],
                    "source_scale_ids": [scale_by_line[line.line_id]],
                }
            )
        else:
            source_line_ids = matched["source_line_ids"]
            source_scale_ids = matched["source_scale_ids"]
            assert isinstance(source_line_ids, list)
            assert isinstance(source_scale_ids, list)
            source_line_ids.append(line.line_id)
            source_scale_ids.append(scale_by_line[line.line_id])

    merged: list[MergedGeometryLineV2] = []
    for index, group in enumerate(groups):
        line = group["representative"]
        source_line_ids = sorted(set(group["source_line_ids"]))
        source_scale_ids = sorted(set(group["source_scale_ids"]))
        assert isinstance(line, GeometryLineV2)
        global_observed = "global" in source_scale_ids
        local_count = sum(scale_id.startswith("tile-") for scale_id in source_scale_ids)
        stability = min(1.0, 0.35 + (0.35 if global_observed else 0.0) + 0.15 * local_count)
        merged.append(
            MergedGeometryLineV2(
                line_id=f"m{index:04d}",
                x1=line.x1,
                y1=line.y1,
                x2=line.x2,
                y2=line.y2,
                length_px=line.length_px,
                length_normalized=line.length_px / max(diagonal, 1.0),
                angle_rad=_line_angle(line),
                source_line_ids=source_line_ids,
                source_scale_ids=source_scale_ids,
                cross_scale_stability=stability,
            )
        )
    return merged


def propose_structure_regions(
    lines: list[MergedGeometryLineV2],
    canonical_size: tuple[int, int],
) -> list[StructureRegionV2]:
    """Build deterministic regions from spatial cells joined by continuous lines."""

    width, height = canonical_size
    diagonal = math.hypot(width, height)
    cell_lines: dict[tuple[int, int], set[str]] = defaultdict(set)
    line_by_id = {line.line_id: line for line in lines}
    for line in lines:
        for cell in _cells_touched(line, canonical_size):
            cell_lines[cell].add(line.line_id)
    usable_cells = {
        cell for cell, line_ids in cell_lines.items() if len(line_ids) >= MIN_REGION_LINES
    }
    parents = {cell: cell for cell in usable_cells}

    def find(cell: tuple[int, int]) -> tuple[int, int]:
        while parents[cell] != cell:
            parents[cell] = parents[parents[cell]]
            cell = parents[cell]
        return cell

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    for row, column in sorted(usable_cells):
        for neighbor in ((row + 1, column), (row, column + 1)):
            if neighbor not in usable_cells:
                continue
            shared = cell_lines[(row, column)] & cell_lines[neighbor]
            continuous = len(shared) >= 2 or any(
                line_by_id[line_id].cross_scale_stability >= 0.70
                and line_by_id[line_id].length_px >= 0.12 * diagonal
                for line_id in shared
            )
            if continuous and _orientation_similarity(
                cell_lines[(row, column)], cell_lines[neighbor], line_by_id
            ) >= 0.65:
                union((row, column), neighbor)

    components: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for cell in sorted(usable_cells):
        components[find(cell)].append(cell)
    regions: list[StructureRegionV2] = []
    for index, cells in enumerate(sorted(components.values(), key=lambda item: item[0])):
        line_ids = sorted({line_id for cell in cells for line_id in cell_lines[cell]})
        selected = [line_by_id[line_id] for line_id in line_ids]
        x0 = min(int(width * column / REGION_GRID_SIZE) for _, column in cells)
        y0 = min(int(height * row / REGION_GRID_SIZE) for row, _ in cells)
        x1 = max(int(math.ceil(width * (column + 1) / REGION_GRID_SIZE)) for _, column in cells)
        y1 = max(int(math.ceil(height * (row + 1) / REGION_GRID_SIZE)) for row, _ in cells)
        support = sum(line.length_normalized * line.cross_scale_stability for line in selected)
        regions.append(
            StructureRegionV2(
                region_id=f"r{index + 1:03d}",
                canonical_box=CanonicalBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                cell_ids=[f"c{row}-{column}" for row, column in cells],
                line_ids=line_ids,
                line_count=len(line_ids),
                normalized_line_support=support,
                orientation_entropy=_orientation_entropy(selected),
                status="usable" if len(line_ids) >= MIN_REGION_LINES else "insufficient_support",
            )
        )
    return regions


def fit_region_families(
    lines: list[MergedGeometryLineV2],
    regions: list[StructureRegionV2],
    canonical_size: tuple[int, int],
    *,
    seed: int,
) -> list[GeometryFamilyV2]:
    """Fit one global reference and independent families inside each region."""

    records = _line_records(lines)
    by_id = {line.line_id: line for line in records}
    transform = CoordinateTransform(
        encoded_size=canonical_size,
        canonical_size=canonical_size,
        analysis_size=canonical_size,
        exif_orientation=1,
        orientation_applied=False,
    )
    config = VanishingPointConfig(
        min_family_lines=4,
        min_family_weight=40.0,
        local_min_family_weight=30.0,
        bootstrap_rounds=6,
    )
    families: list[GeometryFamilyV2] = []
    families.extend(
        _convert_families(
            "global",
            fit_parallel_families(records, canonical_size, transform, config, seed),
            "parallel",
        )
    )
    families.extend(
        _convert_vp_families(
            "global",
            fit_vanishing_families(records, canonical_size, transform, config, seed ^ 0xA51),
        )
    )
    for index, region in enumerate(regions):
        region_lines = [by_id[line_id] for line_id in region.line_ids if line_id in by_id]
        if len(region_lines) < MIN_REGION_LINES:
            continue
        region_seed = seed ^ ((index + 1) * 0x9E3779B1)
        families.extend(
            _convert_families(
                region.region_id,
                fit_parallel_families(
                    region_lines, canonical_size, transform, config, region_seed
                ),
                "parallel",
            )
        )
        families.extend(
            _convert_vp_families(
                region.region_id,
                fit_vanishing_families(
                    region_lines, canonical_size, transform, config, region_seed ^ 0xB43
                ),
            )
        )
    return families


def measure_consistency_checks(
    lines: list[MergedGeometryLineV2],
    regions: list[StructureRegionV2],
    families: list[GeometryFamilyV2],
    canonical_rgb: np.ndarray,
) -> list[GeometryCheckV2]:
    return [
        _measure_g1(lines, regions, families),
        _measure_g2(regions, families),
        _measure_g3(lines, regions, families),
        _measure_g4(lines, regions, canonical_rgb),
        GeometryCheckV2(
            check_id="G5",
            title="多裁切相机与透视场一致性",
            status="not_run",
            limitations=["geometry_g5_camera_measurements_not_attached"],
        ),
    ]


def _measure_g1(
    lines: list[MergedGeometryLineV2],
    regions: list[StructureRegionV2],
    families: list[GeometryFamilyV2],
) -> GeometryCheckV2:
    line_by_id = {line.line_id: line for line in lines}
    findings: list[GeometryFindingV2] = []
    comparable = 0
    for region in regions:
        all_stable_region_families = [
            family
            for family in families
            if family.region_id == region.region_id and family.stable
        ]
        region_families = [
            family
            for family in all_stable_region_families
            if family.region_id == region.region_id
            and family.kind == "parallel"
            and family.direction_rad is not None
        ]
        if not region_families:
            continue
        comparable += 1
        findings.extend(
            _competing_parallel_family_findings(
                region,
                region_families,
                line_by_id,
                finding_offset=len(findings),
            )
        )
        assigned = {
            line_id
            for family in all_stable_region_families
            for line_id in family.member_line_ids
        }
        for line_id in region.line_ids:
            line = line_by_id.get(line_id)
            if (
                line is None
                or line_id in assigned
                or line.cross_scale_stability < 0.65
                or line.length_normalized < 0.045
            ):
                continue
            nearest = min(
                region_families,
                key=lambda family: _axis_difference(line.angle_rad, float(family.direction_rad)),
            )
            family_lines = [
                line_by_id[member_id]
                for member_id in nearest.member_line_ids
                if member_id in line_by_id
            ]
            if not family_lines or _line_to_family_distance(line, family_lines) > 0.10:
                continue
            residual = math.degrees(
                _axis_difference(line.angle_rad, float(nearest.direction_rad))
            )
            gate = max(3.0, nearest.residual_p90_deg + 2.0)
            if gate < residual <= 28.0:
                severity = min(1.0, (residual - gate) / 4.0)
                severity *= line.cross_scale_stability
                severity *= min(1.0, line.length_normalized / 0.15)
                findings.append(
                    GeometryFindingV2(
                        finding_id=f"g1-{len(findings) + 1:03d}",
                        check_id="G1",
                        region_ids=[region.region_id],
                        family_ids=[nearest.family_id],
                        line_ids=[line_id],
                        severity=severity,
                        measured_value=residual,
                        reference_value=gate,
                        description="跨尺度稳定线段偏离附近局部平行族。",
                    )
                )
    if comparable == 0:
        return GeometryCheckV2(
            check_id="G1",
            title="局部平行族离散度",
            status="not_applicable",
            limitations=["geometry_g1_no_stable_local_parallel_family"],
        )
    return GeometryCheckV2(
        check_id="G1",
        title="局部平行族离散度",
        status="available",
        anomaly_score=max((finding.severity for finding in findings), default=0.0),
        measurements={"comparable_regions": comparable, "candidate_count": len(findings)},
        findings=findings,
        limitations=["geometry_g1_measurement_not_source_evidence"],
    )


def _competing_parallel_family_findings(
    region: StructureRegionV2,
    families: list[GeometryFamilyV2],
    line_by_id: dict[str, MergedGeometryLineV2],
    *,
    finding_offset: int,
) -> list[GeometryFindingV2]:
    """Find a narrow secondary family crossing a repeated dominant family.

    A second direction is not itself anomalous: roofs, streets, and facades
    routinely contain several valid line families.  This check therefore only
    compares families inside the same connected structure region and requires
    all of the following: a dominant family spanning at least four distinct
    normal positions, a secondary family concentrated at no more than two
    positions, clearly weaker support, and overlapping spatial extent.  The
    last condition keeps adjacent structures (for example two roof groups)
    independent even when their directions differ.
    """

    findings: list[GeometryFindingV2] = []
    region_diagonal = math.hypot(
        region.canonical_box.width,
        region.canonical_box.height,
    )
    for first_index, first in enumerate(families):
        if first.direction_rad is None:
            continue
        first_lines = [line_by_id[line_id] for line_id in first.member_line_ids if line_id in line_by_id]
        if not first_lines:
            continue
        for second in families[first_index + 1 :]:
            if second.direction_rad is None:
                continue
            second_lines = [
                line_by_id[line_id]
                for line_id in second.member_line_ids
                if line_id in line_by_id
            ]
            if not second_lines:
                continue
            difference_deg = math.degrees(
                _axis_difference(float(first.direction_rad), float(second.direction_rad))
            )
            if not 4.0 <= difference_deg <= 28.0:
                continue
            first_support = _family_support(first_lines)
            second_support = _family_support(second_lines)
            if first_support >= second_support:
                dominant, dominant_lines = first, first_lines
                minor, minor_lines = second, second_lines
                support_ratio = second_support / max(first_support, 1e-9)
            else:
                dominant, dominant_lines = second, second_lines
                minor, minor_lines = first, first_lines
                support_ratio = first_support / max(second_support, 1e-9)
            if support_ratio > 0.65 or dominant.direction_rad is None:
                continue
            minimum_gap = max(2.0, 0.008 * region_diagonal)
            dominant_positions = _family_unique_normal_positions(
                dominant_lines,
                float(dominant.direction_rad),
                minimum_gap,
            )
            minor_positions = _family_unique_normal_positions(
                minor_lines,
                float(minor.direction_rad),
                minimum_gap,
            )
            if len(dominant_positions) < 4 or len(minor_positions) > 2:
                continue
            spatial_overlap = _family_spatial_overlap(
                dominant_lines,
                minor_lines,
                float(dominant.direction_rad),
                region_diagonal,
            )
            if spatial_overlap < 0.60:
                continue
            severity = min(1.0, max(0.0, (difference_deg - 3.0) / 4.0))
            severity *= 0.75 + 0.25 * (1.0 - support_ratio)
            severity *= 0.80 + 0.20 * spatial_overlap
            findings.append(
                GeometryFindingV2(
                    finding_id=f"g1-{finding_offset + len(findings) + 1:03d}",
                    check_id="G1",
                    region_ids=[region.region_id],
                    family_ids=[dominant.family_id, minor.family_id],
                    line_ids=minor.member_line_ids,
                    severity=min(1.0, severity),
                    measured_value=difference_deg,
                    reference_value=4.0,
                    description="同一结构区内的窄支撑次级线族偏离重复主线族。",
                )
            )
    return findings


def _measure_g2(
    regions: list[StructureRegionV2],
    families: list[GeometryFamilyV2],
) -> GeometryCheckV2:
    global_families = [family for family in families if family.region_id == "global" and family.stable]
    findings: list[GeometryFindingV2] = []
    comparisons = 0
    for region in regions:
        center = (
            region.canonical_box.x + region.canonical_box.width / 2.0,
            region.canonical_box.y + region.canonical_box.height / 2.0,
        )
        for local in [
            family for family in families if family.region_id == region.region_id and family.stable
        ]:
            candidates = [
                family
                for family in global_families
                if family.kind == local.kind
                and len(set(local.member_line_ids) & set(family.member_line_ids))
                >= max(3, math.ceil(0.50 * len(local.member_line_ids)))
                and _set_jaccard(local.member_line_ids, family.member_line_ids) >= 0.65
            ]
            if not candidates:
                continue
            global_family = max(
                candidates,
                key=lambda family: len(set(local.member_line_ids) & set(family.member_line_ids)),
            )
            local_direction = _family_direction_at(local, center)
            global_direction = _family_direction_at(global_family, center)
            if local_direction is None or global_direction is None:
                continue
            comparisons += 1
            difference = math.degrees(_axis_difference(local_direction, global_direction))
            tolerance = max(5.0, local.residual_p90_deg + global_family.residual_p90_deg + 2.0)
            if difference > tolerance:
                findings.append(
                    GeometryFindingV2(
                        finding_id=f"g2-{len(findings) + 1:03d}",
                        check_id="G2",
                        region_ids=[region.region_id],
                        family_ids=[local.family_id, global_family.family_id],
                        line_ids=sorted(
                            set(local.member_line_ids) & set(global_family.member_line_ids)
                        ),
                        severity=min(1.0, (difference - tolerance) / 20.0),
                        measured_value=difference,
                        reference_value=tolerance,
                        description="同一组支持线的局部与全局消失方向不一致。",
                    )
                )
    if comparisons == 0:
        return GeometryCheckV2(
            check_id="G2",
            title="局部与全局消失方向",
            status="not_applicable",
            limitations=["geometry_g2_no_shared_stable_family_support"],
        )
    return GeometryCheckV2(
        check_id="G2",
        title="局部与全局消失方向",
        status="available",
        anomaly_score=max((finding.severity for finding in findings), default=0.0),
        measurements={"comparison_count": comparisons, "conflict_count": len(findings)},
        findings=findings,
        limitations=["geometry_g2_measurement_not_source_evidence"],
    )


def _measure_g3(
    lines: list[MergedGeometryLineV2],
    regions: list[StructureRegionV2],
    families: list[GeometryFamilyV2],
) -> GeometryCheckV2:
    line_by_id = {line.line_id: line for line in lines}
    findings: list[GeometryFindingV2] = []
    sequences = 0
    for region in regions:
        parallel = [
            family
            for family in families
            if family.region_id == region.region_id
            and family.kind == "parallel"
            and family.stable
            and family.direction_rad is not None
            and len(family.member_line_ids) >= 4
        ]
        if len(parallel) < 2:
            continue
        for family in parallel:
            if not any(
                math.radians(55.0)
                <= _axis_difference(float(family.direction_rad), float(other.direction_rad))
                <= math.radians(90.0)
                for other in parallel
                if other.family_id != family.family_id and other.direction_rad is not None
            ):
                continue
            members = [line_by_id[line_id] for line_id in family.member_line_ids if line_id in line_by_id]
            if len(members) < 4:
                continue
            lengths = np.asarray([line.length_px for line in members], dtype=np.float64)
            if (
                float(np.std(lengths) / max(np.mean(lengths), 1e-9)) > 0.45
                or float(np.mean([line.cross_scale_stability for line in members])) < 0.65
                or family.residual_p90_deg > 2.5
            ):
                continue
            positions = sorted(_normal_position(line, float(family.direction_rad)) for line in members)
            minimum_gap = 0.005 * math.hypot(
                region.canonical_box.width, region.canonical_box.height
            )
            positions = _deduplicate_positions(positions, minimum_gap)
            if len(positions) < 5:
                continue
            gaps = np.diff(np.asarray(positions, dtype=np.float64))
            positive = gaps[gaps > minimum_gap]
            if len(positive) < 4:
                continue
            sequences += 1
            log_gaps = np.log(positive)
            x = np.arange(len(log_gaps), dtype=np.float64)
            fitted = np.polyval(np.polyfit(x, log_gaps, 1), x) if len(log_gaps) > 1 else log_gaps
            ratio = float(np.exp(np.max(np.abs(log_gaps - fitted))))
            if ratio >= 3.0:
                findings.append(
                    GeometryFindingV2(
                        finding_id=f"g3-{len(findings) + 1:03d}",
                        check_id="G3",
                        region_ids=[region.region_id],
                        family_ids=[family.family_id],
                        line_ids=family.member_line_ids,
                        severity=min(1.0, math.log(ratio) / math.log(4.0)),
                        measured_value=ratio,
                        reference_value=3.0,
                        description="重复线间距偏离平滑透视变化。",
                    )
                )
    if sequences == 0:
        return GeometryCheckV2(
            check_id="G3",
            title="重复结构透视间距",
            status="not_applicable",
            limitations=["geometry_g3_no_rectifiable_repeated_structure"],
        )
    return GeometryCheckV2(
        check_id="G3",
        title="重复结构透视间距",
        status="available",
        anomaly_score=max((finding.severity for finding in findings), default=0.0),
        measurements={"sequence_count": sequences, "irregular_sequence_count": len(findings)},
        findings=findings,
        limitations=[
            "geometry_g3_experimental_not_origin_eligible",
            "geometry_g3_spacing_model_does_not_resolve_semantic_occlusion",
        ],
    )


def _measure_g4(
    lines: list[MergedGeometryLineV2],
    regions: list[StructureRegionV2],
    canonical_rgb: np.ndarray,
) -> GeometryCheckV2:
    height, width = canonical_rgb.shape[:2]
    diagonal = math.hypot(width, height)
    edge_map = cv2.Canny(cv2.cvtColor(canonical_rgb, cv2.COLOR_RGB2GRAY), 80, 160)
    line_by_id = {line.line_id: line for line in lines}
    findings: list[GeometryFindingV2] = []
    tested_pairs = 0
    seen_pairs: set[tuple[str, str]] = set()
    for region in regions:
        candidates = [
            line_by_id[line_id]
            for line_id in region.line_ids
            if line_id in line_by_id
            and line_by_id[line_id].cross_scale_stability >= 0.65
            and line_by_id[line_id].length_normalized >= 0.04
        ]
        for first_index, first in enumerate(candidates):
            for second in candidates[first_index + 1 :]:
                key = tuple(sorted((first.line_id, second.line_id)))
                if key in seen_pairs:
                    continue
                endpoints = _nearest_endpoints(first, second)
                distance = math.dist(endpoints[0], endpoints[1])
                if distance > max(6.0, 0.008 * diagonal):
                    continue
                angle = math.degrees(_axis_difference(first.angle_rad, second.angle_rad))
                if not 10.0 <= angle <= 22.0:
                    continue
                tested_pairs += 1
                seen_pairs.add(key)
                edge_support = _edge_bridge_support(edge_map, endpoints[0], endpoints[1])
                if edge_support < 0.70:
                    continue
                severity = min(1.0, (angle - 8.0) / 20.0) * min(
                    first.cross_scale_stability, second.cross_scale_stability
                )
                findings.append(
                    GeometryFindingV2(
                        finding_id=f"g4-{len(findings) + 1:03d}",
                        check_id="G4",
                        region_ids=[region.region_id],
                        line_ids=[first.line_id, second.line_id],
                        severity=severity,
                        measured_value=angle,
                        reference_value=8.0,
                        description="具有边缘连续支持的相邻长线出现方向突变。",
                    )
                )
                if len(findings) >= 12:
                    break
            if len(findings) >= 12:
                break
    if tested_pairs == 0:
        return GeometryCheckV2(
            check_id="G4",
            title="线段连接与断裂关系",
            status="not_applicable",
            limitations=["geometry_g4_no_stable_endpoint_continuation"],
        )
    return GeometryCheckV2(
        check_id="G4",
        title="线段连接与断裂关系",
        status="available",
        anomaly_score=max((finding.severity for finding in findings), default=0.0),
        measurements={"tested_pair_count": tested_pairs, "candidate_count": len(findings)},
        findings=findings,
        limitations=[
            "geometry_g4_experimental_not_origin_eligible",
            "geometry_g4_junction_semantics_not_resolved",
        ],
    )


def _same_line_support(first: GeometryLineV2, second: GeometryLineV2, diagonal: float) -> bool:
    if _axis_difference(_line_angle(first), _line_angle(second)) > math.radians(MERGE_ANGLE_DEG):
        return False
    first_midpoint = np.asarray([(first.x1 + first.x2) / 2.0, (first.y1 + first.y2) / 2.0])
    second_midpoint = np.asarray([(second.x1 + second.x2) / 2.0, (second.y1 + second.y2) / 2.0])
    axis = np.asarray([math.cos(_line_angle(first)), math.sin(_line_angle(first))])
    normal = np.asarray([-axis[1], axis[0]])
    if abs(float(np.dot(second_midpoint - first_midpoint, normal))) > max(3.0, 0.006 * diagonal):
        return False
    first_interval = sorted((float(np.dot([first.x1, first.y1], axis)), float(np.dot([first.x2, first.y2], axis))))
    second_interval = sorted((float(np.dot([second.x1, second.y1], axis)), float(np.dot([second.x2, second.y2], axis))))
    overlap = max(0.0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))
    minimum_length = min(first_interval[1] - first_interval[0], second_interval[1] - second_interval[0])
    return minimum_length > 0.0 and overlap / minimum_length >= 0.60


def _cells_touched(
    line: MergedGeometryLineV2, canonical_size: tuple[int, int]
) -> set[tuple[int, int]]:
    width, height = canonical_size
    count = max(4, min(24, int(line.length_px / max(width, height) * 32)))
    cells: set[tuple[int, int]] = set()
    for fraction in np.linspace(0.0, 1.0, count):
        x = line.x1 + (line.x2 - line.x1) * float(fraction)
        y = line.y1 + (line.y2 - line.y1) * float(fraction)
        column = min(REGION_GRID_SIZE - 1, max(0, int(x / width * REGION_GRID_SIZE)))
        row = min(REGION_GRID_SIZE - 1, max(0, int(y / height * REGION_GRID_SIZE)))
        cells.add((row, column))
    return cells


def _orientation_similarity(
    first_ids: set[str],
    second_ids: set[str],
    line_by_id: dict[str, MergedGeometryLineV2],
) -> float:
    first = _orientation_histogram([line_by_id[line_id] for line_id in first_ids])
    second = _orientation_histogram([line_by_id[line_id] for line_id in second_ids])
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0.0 else 0.0


def _orientation_histogram(lines: list[MergedGeometryLineV2]) -> np.ndarray:
    values = np.asarray([line.angle_rad for line in lines], dtype=np.float64)
    weights = np.asarray(
        [line.length_normalized * line.cross_scale_stability for line in lines],
        dtype=np.float64,
    )
    histogram, _ = np.histogram(values, bins=12, range=(0.0, math.pi), weights=weights)
    return histogram.astype(np.float64)


def _orientation_entropy(lines: list[MergedGeometryLineV2]) -> float:
    histogram = _orientation_histogram(lines)
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.0
    probability = histogram[histogram > 0.0] / total
    return float(-np.sum(probability * np.log(probability)) / math.log(12.0))


def _line_records(lines: list[MergedGeometryLineV2]) -> list[LineRecord]:
    return [
        LineRecord(
            line_id=line.line_id,
            p1_analysis=Point(x=line.x1, y=line.y1),
            p2_analysis=Point(x=line.x2, y=line.y2),
            p1=Point(x=line.x1, y=line.y1),
            p2=Point(x=line.x2, y=line.y2),
            length_analysis=line.length_px,
            length=line.length_px,
            angle_rad=line.angle_rad,
            quality=max(0.05, line.cross_scale_stability),
            backend_features={"cross_scale_stability": line.cross_scale_stability},
            selected=True,
        )
        for line in lines
    ]


def _convert_families(
    region_id: str,
    families: list[VPFamily],
    kind: str,
) -> list[GeometryFamilyV2]:
    converted: list[GeometryFamilyV2] = []
    for index, family in enumerate(families):
        converted.append(
            GeometryFamilyV2(
                family_id=f"{region_id}-p{index + 1:02d}",
                region_id=region_id,
                kind=kind,
                member_line_ids=family.member_line_ids,
                direction_rad=family.direction_analysis,
                weighted_inlier_ratio=family.weighted_inlier_ratio,
                residual_p50_deg=family.residual_quantiles_deg.get("p50", family.weighted_median_residual_deg),
                residual_p90_deg=family.residual_quantiles_deg.get("p90", family.weighted_median_residual_deg),
                bootstrap_stability=family.bootstrap_stability,
                stable=family.stable,
            )
        )
    return converted


def _convert_vp_families(region_id: str, families: list[VPFamily]) -> list[GeometryFamilyV2]:
    converted: list[GeometryFamilyV2] = []
    for index, family in enumerate(families):
        converted.append(
            GeometryFamilyV2(
                family_id=f"{region_id}-v{index + 1:02d}",
                region_id=region_id,
                kind="finite_vp" if family.vp_type == "finite" else "infinite_vp",
                member_line_ids=family.member_line_ids,
                direction_rad=family.direction_analysis,
                vanishing_point=(
                    GeometryPointV2(x=family.vp.x, y=family.vp.y)
                    if family.vp is not None
                    else None
                ),
                weighted_inlier_ratio=family.weighted_inlier_ratio,
                residual_p50_deg=family.residual_quantiles_deg.get("p50", family.weighted_median_residual_deg),
                residual_p90_deg=family.residual_quantiles_deg.get("p90", family.weighted_median_residual_deg),
                bootstrap_stability=family.bootstrap_stability,
                stable=family.stable,
            )
        )
    return converted


def _family_direction_at(
    family: GeometryFamilyV2, center: tuple[float, float]
) -> float | None:
    if family.vanishing_point is not None:
        return math.atan2(
            family.vanishing_point.y - center[1], family.vanishing_point.x - center[0]
        ) % math.pi
    return family.direction_rad


def _normal_position(line: MergedGeometryLineV2, direction: float) -> float:
    midpoint = np.asarray([(line.x1 + line.x2) / 2.0, (line.y1 + line.y2) / 2.0])
    normal = np.asarray([-math.sin(direction), math.cos(direction)])
    return float(np.dot(midpoint, normal))


def _family_support(lines: list[MergedGeometryLineV2]) -> float:
    return float(
        sum(line.length_normalized * line.cross_scale_stability for line in lines)
    )


def _family_unique_normal_positions(
    lines: list[MergedGeometryLineV2],
    direction: float,
    minimum_gap: float,
) -> list[float]:
    return _deduplicate_positions(
        sorted(_normal_position(line, direction) for line in lines),
        minimum_gap,
    )


def _family_spatial_overlap(
    dominant: list[MergedGeometryLineV2],
    minor: list[MergedGeometryLineV2],
    direction: float,
    region_diagonal: float,
) -> float:
    """Return axial overlap when the minor family lies inside the major band."""

    axis = np.asarray([math.cos(direction), math.sin(direction)], dtype=np.float64)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    def projections(
        selected: list[MergedGeometryLineV2],
    ) -> tuple[np.ndarray, np.ndarray]:
        endpoints = np.asarray(
            [
                point
                for line in selected
                for point in ((line.x1, line.y1), (line.x2, line.y2))
            ],
            dtype=np.float64,
        )
        midpoints = np.asarray(
            [[(line.x1 + line.x2) / 2.0, (line.y1 + line.y2) / 2.0] for line in selected],
            dtype=np.float64,
        )
        return endpoints @ axis, midpoints @ normal

    dominant_axis, dominant_normal = projections(dominant)
    minor_axis, minor_normal = projections(minor)
    dominant_interval = (float(np.min(dominant_axis)), float(np.max(dominant_axis)))
    minor_interval = (float(np.min(minor_axis)), float(np.max(minor_axis)))
    overlap = max(
        0.0,
        min(dominant_interval[1], minor_interval[1])
        - max(dominant_interval[0], minor_interval[0]),
    )
    minor_span = max(1.0, minor_interval[1] - minor_interval[0])
    axial_overlap = min(1.0, overlap / minor_span)
    margin = max(4.0, 0.03 * region_diagonal)
    dominant_normal_interval = (
        float(np.min(dominant_normal)) - margin,
        float(np.max(dominant_normal)) + margin,
    )
    minor_center = float(np.median(minor_normal))
    if not dominant_normal_interval[0] <= minor_center <= dominant_normal_interval[1]:
        return 0.0
    return axial_overlap


def _deduplicate_positions(values: list[float], minimum_gap: float) -> list[float]:
    retained: list[float] = []
    for value in values:
        if not retained or value - retained[-1] >= minimum_gap:
            retained.append(value)
    return retained


def _nearest_endpoints(
    first: MergedGeometryLineV2, second: MergedGeometryLineV2
) -> tuple[tuple[float, float], tuple[float, float]]:
    first_points = ((first.x1, first.y1), (first.x2, first.y2))
    second_points = ((second.x1, second.y1), (second.x2, second.y2))
    return min(
        ((first_point, second_point) for first_point in first_points for second_point in second_points),
        key=lambda pair: math.dist(pair[0], pair[1]),
    )


def _line_to_family_distance(
    line: MergedGeometryLineV2,
    family_lines: list[MergedGeometryLineV2],
) -> float:
    """Normalized midpoint distance prevents unrelated structures being compared."""

    midpoint = ((line.x1 + line.x2) / 2.0, (line.y1 + line.y2) / 2.0)
    scale = max(line.length_px, max(member.length_px for member in family_lines), 1.0)
    return min(
        math.dist(
            midpoint,
            ((member.x1 + member.x2) / 2.0, (member.y1 + member.y2) / 2.0),
        )
        / scale
        for member in family_lines
    )


def _edge_bridge_support(
    edge_map: np.ndarray,
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    height, width = edge_map.shape
    distance = max(1.0, math.dist(first, second))
    count = max(3, int(math.ceil(distance)) + 1)
    hits = 0
    for fraction in np.linspace(0.0, 1.0, count):
        x = int(round(first[0] + (second[0] - first[0]) * float(fraction)))
        y = int(round(first[1] + (second[1] - first[1]) * float(fraction)))
        x0, x1 = max(0, x - 1), min(width, x + 2)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        if x0 < x1 and y0 < y1 and np.any(edge_map[y0:y1, x0:x1] > 0):
            hits += 1
    return hits / count


def _line_angle(line: GeometryLineV2) -> float:
    return math.atan2(line.y2 - line.y1, line.x2 - line.x1) % math.pi


def _axis_difference(first: float, second: float) -> float:
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def _set_jaccard(first: list[str], second: list[str]) -> float:
    first_set, second_set = set(first), set(second)
    union = first_set | second_set
    return len(first_set & second_set) / len(union) if union else 0.0
