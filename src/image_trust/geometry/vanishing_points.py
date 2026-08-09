"""Deterministic, explainable multi-vanishing-direction baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from image_trust.schemas import AnomalyCandidate, LineRecord, Point, VPFamily, VanishingPointConfig
from image_trust.utils.coordinates import CoordinateTransform


@dataclass(frozen=True)
class _GeometryLine:
    index: int
    record: LineRecord
    midpoint: np.ndarray
    axis: np.ndarray
    homogeneous: np.ndarray
    weight: float


@dataclass(frozen=True)
class _Candidate:
    kind: str
    finite_point: np.ndarray | None = None
    direction: np.ndarray | None = None


@dataclass(frozen=True)
class _Fit:
    candidate: _Candidate
    members: list[int]
    residuals_deg: np.ndarray
    score: float


def fit_vanishing_families(
    lines: list[LineRecord],
    image_size: tuple[int, int],
    transform: CoordinateTransform,
    config: VanishingPointConfig,
    seed: int,
) -> list[VPFamily]:
    geometry = [_to_geometry(index, line) for index, line in enumerate(lines)]
    remaining = list(range(len(geometry)))
    rng = np.random.default_rng(seed)
    families: list[VPFamily] = []
    total_weight = sum(line.weight for line in geometry)
    while remaining and len(families) < config.max_families:
        fit = _fit_one_family([geometry[index] for index in remaining], config, rng)
        if fit is None:
            break
        if len(fit.members) < config.min_family_lines or fit.score < config.min_family_weight:
            break
        member_records = [geometry[index] for index in fit.members]
        refined = _refine_candidate(member_records, fit.candidate)
        member_records = [
            geometry[index]
            for index in remaining
            if _residual_deg(geometry[index], refined) <= config.inlier_angle_deg
        ]
        if len(member_records) < config.min_family_lines:
            break
        refined_weight = sum(line.weight for line in member_records)
        if refined_weight < config.min_family_weight:
            break
        residuals = np.asarray([_residual_deg(line, refined) for line in member_records])
        weights = np.asarray([line.weight for line in member_records])
        duplicate_index = _duplicate_family_index(
            refined,
            member_records,
            families,
            image_size,
            config,
        )
        if duplicate_index is not None:
            existing = families[duplicate_index]
            existing_ids = set(existing.member_line_ids)
            member_records = [
                line for line in geometry if line.record.line_id in existing_ids
            ] + member_records
            refined = _refine_candidate(member_records, refined)
            residuals = np.asarray([_residual_deg(line, refined) for line in member_records])
            weights = np.asarray([line.weight for line in member_records])
            family_id = existing.family_id
        else:
            family_id = f"vp{len(families) + 1:03d}"
        family = _to_family(
            family_id=family_id,
            candidate=refined,
            member_records=member_records,
            residuals=residuals,
            image_size=image_size,
            transform=transform,
            total_weight=total_weight,
            config=config,
            rng=rng,
        )
        if duplicate_index is None:
            families.append(family)
        else:
            families[duplicate_index] = family
        consumed = {line.index for line in member_records}
        remaining = [index for index in remaining if index not in consumed]
    return families


def fit_parallel_families(
    lines: list[LineRecord],
    image_size: tuple[int, int],
    transform: CoordinateTransform,
    config: VanishingPointConfig,
    seed: int,
) -> list[VPFamily]:
    """Find image-plane parallel families for human review overlays.

    This is intentionally separate from global finite/at-infinity VP fitting.
    A repeated roof, facade, or rail direction can be visually parallel in the
    image even when a global VP hypothesis would group it differently. The
    resulting families are descriptive overlays, never source evidence.
    """
    geometry = [_to_geometry(index, line) for index, line in enumerate(lines)]
    remaining = list(range(len(geometry)))
    total_weight = sum(line.weight for line in geometry)
    rng = np.random.default_rng(seed ^ 0x9E3779B97F4A7C15)
    families: list[VPFamily] = []
    threshold = math.radians(config.parallel_inlier_angle_deg)
    while remaining and len(families) < config.max_parallel_families:
        best_members: list[_GeometryLine] | None = None
        best_score = -1.0
        best_direction: np.ndarray | None = None
        for index in remaining:
            candidate_direction = geometry[index].axis
            members = [
                geometry[member_index]
                for member_index in remaining
                if _axis_angle(geometry[member_index].axis, candidate_direction) <= threshold
            ]
            if len(members) < config.min_family_lines:
                continue
            residuals = np.asarray(
                [_axis_angle(member.axis, candidate_direction) for member in members]
            )
            weights = np.asarray([member.weight for member in members])
            robust = np.maximum(0.0, 1.0 - (residuals / threshold) ** 2)
            score = float(np.sum(weights * robust))
            if score > best_score:
                best_members = members
                best_score = score
                best_direction = candidate_direction
        if (
            best_members is None
            or best_direction is None
            or best_score < config.min_family_weight
        ):
            break
        candidate = _refine_candidate(
            best_members,
            _Candidate(kind="infinite", direction=best_direction),
        )
        members = [
            geometry[index]
            for index in remaining
            if _residual_deg(geometry[index], candidate) <= config.parallel_inlier_angle_deg
        ]
        member_weight = sum(member.weight for member in members)
        if len(members) < config.min_family_lines or member_weight < config.min_family_weight:
            break
        residuals = np.asarray([_residual_deg(member, candidate) for member in members])
        family = _to_family(
            family_id=f"parallel{len(families) + 1:03d}",
            candidate=candidate,
            member_records=members,
            residuals=residuals,
            image_size=image_size,
            transform=transform,
            total_weight=total_weight,
            config=config,
            rng=rng,
            scope="image_plane_parallel",
        )
        families.append(family)
        consumed = {member.index for member in members}
        remaining = [index for index in remaining if index not in consumed]
    return families


def fit_local_vanishing_families(
    lines: list[LineRecord],
    image_size: tuple[int, int],
    transform: CoordinateTransform,
    config: VanishingPointConfig,
    seed: int,
) -> list[VPFamily]:
    """Fit independent VP families inside spatial cells for overlay review.

    Global VP fitting remains the measurement baseline. These local families do
    not replace it: they prevent visually unrelated regions from being painted
    as one family merely because a global hypothesis can explain both.
    """
    width, height = image_size
    grid_size = config.local_family_grid_size
    local_config = config.model_copy(
        update={
            "max_families": config.local_families_per_cell,
            "min_family_weight": config.local_min_family_weight,
        }
    )
    line_by_id = {line.line_id: line for line in lines}
    candidates: list[tuple[VPFamily, float]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            x0 = width * col / grid_size
            x1 = width * (col + 1) / grid_size
            y0 = height * row / grid_size
            y1 = height * (row + 1) / grid_size
            cell_lines = [
                line
                for line in lines
                if x0 <= (line.p1_analysis.x + line.p2_analysis.x) / 2.0 < x1
                and y0 <= (line.p1_analysis.y + line.p2_analysis.y) / 2.0 < y1
            ]
            if len(cell_lines) < local_config.min_family_lines:
                continue
            cell_seed = seed ^ ((row + 1) << 32) ^ (col + 1)
            fitted = fit_vanishing_families(
                cell_lines,
                image_size,
                transform,
                local_config,
                cell_seed,
            )
            window = (float(x0), float(y0), float(x1), float(y1))
            for family in fitted:
                scoped = family.model_copy(
                    update={
                        "scope": "local_vp",
                        "spatial_window_analysis": window,
                    }
                )
                support_weight = sum(
                    line_by_id[line_id].length_analysis
                    * max(line_by_id[line_id].quality, 0.05)
                    for line_id in scoped.member_line_ids
                    if line_id in line_by_id
                )
                candidates.append((scoped, support_weight))
    candidates.sort(key=lambda item: (-item[1], item[0].family_id))
    selected = candidates[: config.max_local_families]
    return [
        family.model_copy(update={"family_id": f"local{index:03d}"})
        for index, (family, _) in enumerate(selected, start=1)
    ]


def fit_local_parallel_families(
    lines: list[LineRecord],
    image_size: tuple[int, int],
    transform: CoordinateTransform,
    config: VanishingPointConfig,
    seed: int,
    excluded_line_ids: set[str] | None = None,
) -> list[VPFamily]:
    """Find spatially local image-plane directions for review overlays.

    A global VP is deliberately strict: it is the P0 measurement model.  For
    visual review, however, a nearby roof or facade may contain a distinct
    repeated direction whose individual LSD fragments are too short to give a
    reliable global finite VP.  This local direction grouping is therefore a
    separate, explicitly descriptive layer.  Lines already explained by a
    stable global VP are removed first so dominant vertical or street-depth
    families cannot crowd out the local structure a reviewer is inspecting.
    """
    width, height = image_size
    grid_size = config.local_family_grid_size
    excluded = excluded_line_ids or set()
    local_config = config.model_copy(
        update={
            "max_parallel_families": config.local_direction_families_per_cell,
            "parallel_inlier_angle_deg": config.local_direction_inlier_angle_deg,
            "min_family_weight": config.local_min_family_weight,
        }
    )
    line_by_id = {line.line_id: line for line in lines}
    candidates: list[tuple[VPFamily, float]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            x0 = width * col / grid_size
            x1 = width * (col + 1) / grid_size
            y0 = height * row / grid_size
            y1 = height * (row + 1) / grid_size
            cell_lines = [
                line
                for line in lines
                if line.line_id not in excluded
                and x0 <= (line.p1_analysis.x + line.p2_analysis.x) / 2.0 < x1
                and y0 <= (line.p1_analysis.y + line.p2_analysis.y) / 2.0 < y1
            ]
            if len(cell_lines) < local_config.min_family_lines:
                continue
            cell_seed = seed ^ 0xA24BAED4963EE407 ^ ((row + 1) << 32) ^ (col + 1)
            fitted = fit_parallel_families(
                cell_lines,
                image_size,
                transform,
                local_config,
                cell_seed,
            )
            window = (float(x0), float(y0), float(x1), float(y1))
            for family in fitted:
                scoped = family.model_copy(
                    update={
                        "scope": "local_image_plane_parallel",
                        "spatial_window_analysis": window,
                    }
                )
                support_weight = sum(
                    line_by_id[line_id].length_analysis
                    * max(line_by_id[line_id].quality, 0.05)
                    for line_id in scoped.member_line_ids
                    if line_id in line_by_id
                )
                candidates.append((scoped, support_weight))
    candidates.sort(key=lambda item: (-item[1], item[0].family_id))
    merged: list[tuple[VPFamily, float]] = []
    for component in _merge_adjacent_local_direction_components(
        [family for family, _ in candidates],
        config.local_direction_inlier_angle_deg,
    ):
        member_ids = {
            line_id for family in component for line_id in family.member_line_ids
        }
        member_records = [
            _to_geometry(index, line)
            for index, line in enumerate(lines)
            if line.line_id in member_ids
        ]
        if len(member_records) < config.min_family_lines:
            continue
        first_direction = component[0].direction_analysis
        if first_direction is None:
            continue
        candidate = _refine_candidate(
            member_records,
            _Candidate(
                kind="infinite",
                direction=np.array(
                    [math.cos(first_direction), math.sin(first_direction)], dtype=float
                ),
            ),
        )
        residuals = np.asarray(
            [_residual_deg(record, candidate) for record in member_records]
        )
        window = _combined_spatial_window(component)
        family = _to_family(
            family_id="localdirection-pending",
            candidate=candidate,
            member_records=member_records,
            residuals=residuals,
            image_size=image_size,
            transform=transform,
            total_weight=sum(record.weight for record in member_records),
            config=local_config,
            rng=np.random.default_rng(seed ^ len(merged)),
            scope="local_image_plane_parallel",
            spatial_window_analysis=window,
        )
        support_weight = sum(record.weight for record in member_records)
        merged.append((family, support_weight))
    merged.sort(key=lambda item: (-item[1], item[0].family_id))
    selected = merged[: config.max_local_families]
    return [
        family.model_copy(update={"family_id": f"localdirection{index:03d}"})
        for index, (family, _) in enumerate(selected, start=1)
    ]


def _merge_adjacent_local_direction_components(
    families: list[VPFamily],
    angle_threshold_deg: float,
) -> list[list[VPFamily]]:
    """Join a local direction only when its support crosses a cell boundary."""
    components: list[list[VPFamily]] = []
    for family in families:
        matching = [
            index
            for index, component in enumerate(components)
            if any(
                _local_directions_are_continuous(
                    family, existing, angle_threshold_deg
                )
                for existing in component
            )
        ]
        if not matching:
            components.append([family])
            continue
        combined = [family]
        for index in reversed(matching):
            combined.extend(components.pop(index))
        components.append(combined)
    return components


def _local_directions_are_continuous(
    first: VPFamily,
    second: VPFamily,
    angle_threshold_deg: float,
) -> bool:
    if first.direction_analysis is None or second.direction_analysis is None:
        return False
    if math.degrees(
        _axis_angle(
            np.array(
                [math.cos(first.direction_analysis), math.sin(first.direction_analysis)]
            ),
            np.array(
                [math.cos(second.direction_analysis), math.sin(second.direction_analysis)]
            ),
        )
    ) > angle_threshold_deg:
        return False
    if first.spatial_window_analysis is None or second.spatial_window_analysis is None:
        return False
    first_x0, first_y0, first_x1, first_y1 = first.spatial_window_analysis
    second_x0, second_y0, second_x1, second_y1 = second.spatial_window_analysis
    return (
        first_x0 <= second_x1 + 1e-6
        and second_x0 <= first_x1 + 1e-6
        and first_y0 <= second_y1 + 1e-6
        and second_y0 <= first_y1 + 1e-6
    )


def _combined_spatial_window(
    families: list[VPFamily],
) -> tuple[float, float, float, float] | None:
    windows = [family.spatial_window_analysis for family in families]
    usable = [window for window in windows if window is not None]
    if not usable:
        return None
    return (
        min(window[0] for window in usable),
        min(window[1] for window in usable),
        max(window[2] for window in usable),
        max(window[3] for window in usable),
    )


def identify_anomaly_candidates(
    lines: list[LineRecord],
    families: list[VPFamily],
    image_size: tuple[int, int],
    config: VanishingPointConfig,
    applicability: float,
    minimum_applicability: float,
    explained_line_ids: set[str] | None = None,
) -> list[AnomalyCandidate]:
    stable_families = [family for family in families if family.stable]
    if applicability < minimum_applicability or not stable_families:
        return []
    member_ids = {
        line_id for family in stable_families for line_id in family.member_line_ids
    }
    if explained_line_ids:
        member_ids.update(explained_line_ids)
    diagonal = math.hypot(*image_size)
    candidates: list[AnomalyCandidate] = []
    for line in lines:
        if line.line_id in member_ids:
            continue
        axis = _axis(line)
        midpoint = _midpoint(line)
        residuals = [
            (_residual_to_serialized_family(axis, midpoint, family), index)
            for index, family in enumerate(stable_families)
        ]
        residual, nearest_index = min(residuals, key=lambda item: item[0])
        family = stable_families[nearest_index]
        residual_component = min(1.0, residual / max(config.inlier_angle_deg * 3.0, 1e-6))
        family_disagreement = 1.0 - family.bootstrap_stability
        local_support = min(1.0, line.length_analysis / max(diagonal * 0.15, 1e-6))
        score = (
            0.55 * residual_component
            + 0.25 * family_disagreement
            + 0.20 * local_support
        )
        candidates.append(
            AnomalyCandidate(
                line_id=line.line_id,
                anomaly_candidate_score=float(max(0.0, min(score, 1.0))),
                nearest_family_id=family.family_id,
                residual_deg=float(residual),
                reason="unassigned_line_deviates_from_stable_vanishing_family",
            )
        )
    return sorted(
        candidates,
        key=lambda item: (-item.anomaly_candidate_score, item.line_id),
    )[:50]


def identify_parallel_anomaly_candidates(
    lines: list[LineRecord],
    families: list[VPFamily],
    image_size: tuple[int, int],
    config: VanishingPointConfig,
    applicability: float,
    minimum_applicability: float,
    explained_line_ids: set[str] | None = None,
) -> list[AnomalyCandidate]:
    """Flag a long unassigned line that conflicts with a nearby parallel family.

    A line with a different direction elsewhere in the picture is ordinary
    scene content, not an anomaly.  This detector therefore considers only
    stable image-plane parallel families and requires the candidate midpoint
    to sit near that family's spatial support.  The result remains a review
    candidate, never a source or AI-origin conclusion.
    """

    stable_families = [
        family
        for family in families
        if family.stable and family.direction_analysis is not None
    ]
    if applicability < minimum_applicability or not stable_families:
        return []
    line_by_id = {line.line_id: line for line in lines}
    member_ids = {
        line_id
        for family in stable_families
        for line_id in family.member_line_ids
    }
    if explained_line_ids:
        member_ids.update(explained_line_ids)
    diagonal = math.hypot(*image_size)
    candidates = _competing_family_candidates(
        lines,
        stable_families,
        image_size,
        config,
        reason="small_parallel_family_conflicts_with_nearby_dominant_family",
    )
    for line in lines:
        if line.line_id in member_ids or line.length_analysis < diagonal * 0.08:
            continue
        midpoint = _midpoint(line)
        comparisons = [
            _parallel_family_comparison(line, midpoint, family, line_by_id, diagonal)
            for family in stable_families
        ]
        usable = [comparison for comparison in comparisons if comparison is not None]
        if not usable:
            continue
        residual, proximity, family = min(usable, key=lambda item: item[0])
        if residual <= config.parallel_inlier_angle_deg:
            continue
        residual_component = min(
            1.0,
            residual / max(config.parallel_inlier_angle_deg * 3.0, 1e-6),
        )
        support_component = min(1.0, line.length_analysis / max(diagonal * 0.15, 1e-6))
        proximity_component = 1.0 - proximity
        score = (
            0.55 * residual_component
            + 0.25 * family.bootstrap_stability
            + 0.20 * min(support_component, proximity_component)
        )
        candidates.append(
            AnomalyCandidate(
                line_id=line.line_id,
                anomaly_candidate_score=float(max(0.0, min(score, 1.0))),
                nearest_family_id=family.family_id,
                residual_deg=float(residual),
                reason="unassigned_line_deviates_from_nearby_parallel_family",
            )
        )
    return sorted(
        candidates,
        key=lambda item: (-item.anomaly_candidate_score, item.line_id),
    )[:50]


def identify_competing_vanishing_family_candidates(
    lines: list[LineRecord],
    families: list[VPFamily],
    image_size: tuple[int, int],
    config: VanishingPointConfig,
    applicability: float,
    minimum_applicability: float,
) -> list[AnomalyCandidate]:
    """Flag a small local VP family that conflicts with a dominant nearby VP.

    LSD often represents one visible structural stroke with several fragments.
    If the fragments of an erroneous stroke are numerous enough, they form a
    small, internally stable VP family instead of appearing as unassigned
    lines.  This source-neutral review rule compares only spatially adjacent,
    directionally competing families; it does not declare either source type.
    """

    stable_families = [family for family in families if family.stable]
    if applicability < minimum_applicability or len(stable_families) < 2:
        return []
    return _competing_family_candidates(
        lines,
        stable_families,
        image_size,
        config,
        reason="small_vanishing_family_conflicts_with_nearby_dominant_family",
    )


def _competing_family_candidates(
    lines: list[LineRecord],
    families: list[VPFamily],
    image_size: tuple[int, int],
    config: VanishingPointConfig,
    *,
    reason: str,
) -> list[AnomalyCandidate]:
    """Return member lines from a smaller locally competing direction family."""

    line_by_id = {line.line_id: line for line in lines}
    diagonal = math.hypot(*image_size)
    candidates_by_line: dict[str, AnomalyCandidate] = {}
    for smaller in families:
        smaller_lines = [
            line_by_id[line_id]
            for line_id in smaller.member_line_ids
            if line_id in line_by_id
        ]
        if len(smaller_lines) < config.min_family_lines:
            continue
        for dominant in families:
            if dominant.family_id == smaller.family_id:
                continue
            if dominant.weighted_inlier_ratio < smaller.weighted_inlier_ratio * 1.5:
                continue
            proximity = _family_midpoint_proximity(
                smaller_lines,
                [
                    line_by_id[line_id]
                    for line_id in dominant.member_line_ids
                    if line_id in line_by_id
                ],
                diagonal,
            )
            if proximity is None or proximity >= 1.0:
                continue
            residuals = [
                _residual_to_serialized_family(_axis(line), _midpoint(line), dominant)
                for line in smaller_lines
            ]
            residual = float(np.median(residuals))
            if not config.parallel_inlier_angle_deg < residual <= config.parallel_inlier_angle_deg * 12.0:
                continue
            residual_component = min(
                1.0,
                residual / max(config.parallel_inlier_angle_deg * 4.0, 1e-6),
            )
            dominance = min(
                1.0,
                1.0 - smaller.weighted_inlier_ratio / max(dominant.weighted_inlier_ratio, 1e-9),
            )
            score = (
                0.55 * residual_component
                + 0.25 * dominant.bootstrap_stability
                + 0.20 * dominance
            ) * (0.5 + 0.5 * (1.0 - proximity))
            for line in smaller_lines:
                candidate = AnomalyCandidate(
                    line_id=line.line_id,
                    anomaly_candidate_score=float(max(0.0, min(score, 1.0))),
                    nearest_family_id=dominant.family_id,
                    residual_deg=residual,
                    reason=reason,
                )
                existing = candidates_by_line.get(line.line_id)
                if (
                    existing is None
                    or candidate.anomaly_candidate_score > existing.anomaly_candidate_score
                ):
                    candidates_by_line[line.line_id] = candidate
    return sorted(
        candidates_by_line.values(),
        key=lambda item: (-item.anomaly_candidate_score, item.line_id),
    )


def _family_midpoint_proximity(
    first: list[LineRecord],
    second: list[LineRecord],
    diagonal: float,
) -> float | None:
    if not first or not second:
        return None
    first_midpoints = np.asarray([_midpoint(line) for line in first], dtype=float)
    second_midpoints = np.asarray([_midpoint(line) for line in second], dtype=float)
    lower = second_midpoints.min(axis=0)
    upper = second_midpoints.max(axis=0)
    distances = []
    for midpoint in first_midpoints:
        outside = np.maximum(lower - midpoint, 0.0) + np.maximum(midpoint - upper, 0.0)
        distances.append(float(np.linalg.norm(outside)))
    return min(1.0, float(np.median(distances)) / max(diagonal * 0.12, 1e-9))


def _parallel_family_comparison(
    line: LineRecord,
    midpoint: np.ndarray,
    family: VPFamily,
    line_by_id: dict[str, LineRecord],
    diagonal: float,
) -> tuple[float, float, VPFamily] | None:
    """Return angle residual and spatial distance to one family, if reviewable."""

    if family.direction_analysis is None:
        return None
    members = [line_by_id[line_id] for line_id in family.member_line_ids if line_id in line_by_id]
    if len(members) < 2:
        return None
    member_midpoints = np.asarray([_midpoint(member) for member in members], dtype=float)
    lower = member_midpoints.min(axis=0)
    upper = member_midpoints.max(axis=0)
    # A modest padding lets a line at the edge of a facade/roof be compared,
    # while preventing unrelated objects across the image from being linked.
    padding = diagonal * 0.12
    outside = np.maximum(lower - midpoint, 0.0) + np.maximum(midpoint - upper, 0.0)
    proximity = min(1.0, float(np.linalg.norm(outside)) / max(padding, 1e-9))
    if proximity >= 1.0:
        return None
    target = np.array(
        [math.cos(family.direction_analysis), math.sin(family.direction_analysis)]
    )
    residual = math.degrees(_axis_angle(_axis(line), target))
    return residual, proximity, family


def _to_geometry(index: int, line: LineRecord) -> _GeometryLine:
    p1 = np.array([line.p1_analysis.x, line.p1_analysis.y], dtype=float)
    p2 = np.array([line.p2_analysis.x, line.p2_analysis.y], dtype=float)
    axis = p2 - p1
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        axis = np.array([1.0, 0.0])
    else:
        axis /= norm
    homogeneous = np.cross(np.array([p1[0], p1[1], 1.0]), np.array([p2[0], p2[1], 1.0]))
    line_norm = float(np.linalg.norm(homogeneous[:2]))
    if line_norm > 0:
        homogeneous /= line_norm
    return _GeometryLine(
        index=index,
        record=line,
        midpoint=(p1 + p2) / 2.0,
        axis=axis,
        homogeneous=homogeneous,
        weight=max(line.length_analysis * max(line.quality, 0.05), 1e-6),
    )


def _fit_one_family(
    lines: list[_GeometryLine],
    config: VanishingPointConfig,
    rng: np.random.Generator,
) -> _Fit | None:
    if len(lines) < config.min_family_lines:
        return None
    candidates = _generate_candidates(lines, config, rng)
    best: _Fit | None = None
    threshold = config.inlier_angle_deg
    for candidate in candidates:
        residuals = np.asarray([_residual_deg(line, candidate) for line in lines])
        inlier_mask = residuals <= threshold
        members = [lines[index].index for index in np.flatnonzero(inlier_mask)]
        if len(members) < config.min_family_lines:
            continue
        weights = np.asarray([line.weight for line in lines])
        robust = np.maximum(0.0, 1.0 - (residuals / threshold) ** 2)
        score = float(np.sum(weights * robust))
        fit = _Fit(candidate=candidate, members=members, residuals_deg=residuals, score=score)
        if best is None or fit.score > best.score:
            best = fit
    return best


def _generate_candidates(
    lines: list[_GeometryLine],
    config: VanishingPointConfig,
    rng: np.random.Generator,
) -> list[_Candidate]:
    weights = np.asarray([line.weight for line in lines], dtype=float)
    probabilities = weights / weights.sum()
    candidates: list[_Candidate] = []
    pair_min = math.radians(config.pair_min_angle_deg)
    for _ in range(config.max_hypotheses):
        first, second = rng.choice(len(lines), size=2, replace=False, p=probabilities)
        line_a, line_b = lines[int(first)], lines[int(second)]
        if _axis_angle(line_a.axis, line_b.axis) < pair_min:
            continue
        point = np.cross(line_a.homogeneous, line_b.homogeneous)
        if abs(point[2]) <= 1e-9:
            continue
        candidates.append(_Candidate(kind="finite", finite_point=point[:2] / point[2]))
    sample_count = min(len(lines), max(32, config.max_hypotheses // 4))
    for index in rng.choice(len(lines), size=sample_count, replace=False, p=probabilities):
        candidates.append(_Candidate(kind="infinite", direction=lines[int(index)].axis))
    return candidates


def _residual_deg(line: _GeometryLine, candidate: _Candidate) -> float:
    if candidate.kind == "finite" and candidate.finite_point is not None:
        direction = candidate.finite_point - line.midpoint
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= 1e-9:
            return 90.0
        return math.degrees(_axis_angle(line.axis, direction / magnitude))
    if candidate.direction is None:
        return 90.0
    return math.degrees(_axis_angle(line.axis, candidate.direction))


def _axis_angle(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first, second)))
    return math.acos(max(-1.0, min(1.0, dot)))


def _refine_candidate(
    members: list[_GeometryLine],
    candidate: _Candidate,
) -> _Candidate:
    weights = np.sqrt(np.asarray([line.weight for line in members], dtype=float))
    if candidate.kind == "finite":
        matrix = np.asarray([line.homogeneous[:2] for line in members], dtype=float)
        vector = np.asarray([-line.homogeneous[2] for line in members], dtype=float)
        try:
            point, *_ = np.linalg.lstsq(matrix * weights[:, None], vector * weights, rcond=None)
            if np.all(np.isfinite(point)):
                return _Candidate(kind="finite", finite_point=point)
        except np.linalg.LinAlgError:
            pass
    axes = np.asarray([line.axis for line in members], dtype=float)
    scatter = (axes * weights[:, None]).T @ (axes * weights[:, None])
    _, eigenvectors = np.linalg.eigh(scatter)
    direction = eigenvectors[:, -1]
    return _Candidate(kind="infinite", direction=direction / np.linalg.norm(direction))


def _to_family(
    family_id: str,
    candidate: _Candidate,
    member_records: list[_GeometryLine],
    residuals: np.ndarray,
    image_size: tuple[int, int],
    transform: CoordinateTransform,
    total_weight: float,
    config: VanishingPointConfig,
    rng: np.random.Generator,
    scope: str = "global_vp",
    spatial_window_analysis: tuple[float, float, float, float] | None = None,
) -> VPFamily:
    weights = np.asarray([line.weight for line in member_records], dtype=float)
    member_weight = float(weights.sum())
    stability = _bootstrap_stability(member_records, candidate, image_size, config, rng)
    spatial_support = _midpoint_coverage(member_records, image_size)
    vp_analysis: Point | None = None
    vp: Point | None = None
    direction_analysis: float | None = None
    if candidate.kind == "finite" and candidate.finite_point is not None:
        vp_analysis = Point(x=float(candidate.finite_point[0]), y=float(candidate.finite_point[1]))
        vp = transform.analysis_to_canonical(vp_analysis)
    elif candidate.direction is not None:
        direction_analysis = float(math.atan2(candidate.direction[1], candidate.direction[0]) % math.pi)
    return VPFamily(
        family_id=family_id,
        vp_type=candidate.kind,
        vp_analysis=vp_analysis,
        vp=vp,
        direction_analysis=direction_analysis,
        member_line_ids=[line.record.line_id for line in member_records],
        weighted_inlier_ratio=float(
            min(1.0, max(0.0, member_weight / max(total_weight, 1e-9)))
        ),
        weighted_median_residual_deg=_weighted_median(residuals, weights),
        spatial_support=spatial_support,
        bootstrap_stability=stability,
        residual_quantiles_deg={
            "p50": float(np.quantile(residuals, 0.50)),
            "p90": float(np.quantile(residuals, 0.90)),
            "p95": float(np.quantile(residuals, 0.95)),
        },
        stable=stability >= config.stable_family_min_bootstrap,
        scope=scope,
        spatial_window_analysis=spatial_window_analysis,
    )


def _duplicate_family_index(
    candidate: _Candidate,
    member_records: list[_GeometryLine],
    families: list[VPFamily],
    image_size: tuple[int, int],
    config: VanishingPointConfig,
) -> int | None:
    """Merge the same VP hypothesis instead of reporting duplicate line families."""
    candidate_ids = {line.record.line_id for line in member_records}
    diagonal = math.hypot(*image_size)
    for index, family in enumerate(families):
        existing_ids = set(family.member_line_ids)
        union = existing_ids | candidate_ids
        jaccard = len(existing_ids & candidate_ids) / max(len(union), 1)
        if jaccard >= config.family_jaccard_merge:
            return index
        if candidate.kind == "finite" and candidate.finite_point is not None:
            if family.vp_type != "finite" or family.vp_analysis is None:
                continue
            existing = np.array([family.vp_analysis.x, family.vp_analysis.y])
            if float(np.linalg.norm(candidate.finite_point - existing)) <= 0.05 * diagonal:
                return index
        elif candidate.direction is not None and family.vp_type == "infinite":
            if family.direction_analysis is None:
                continue
            existing_direction = np.array(
                [math.cos(family.direction_analysis), math.sin(family.direction_analysis)]
            )
            if math.degrees(_axis_angle(candidate.direction, existing_direction)) <= config.inlier_angle_deg:
                return index
    return None


def _bootstrap_stability(
    members: list[_GeometryLine],
    candidate: _Candidate,
    image_size: tuple[int, int],
    config: VanishingPointConfig,
    rng: np.random.Generator,
) -> float:
    if len(members) < 3 or config.bootstrap_rounds <= 0:
        return 0.0
    sample_size = max(3, math.ceil(len(members) * config.bootstrap_fraction))
    values: list[float] = []
    diagonal = math.hypot(*image_size)
    for _ in range(config.bootstrap_rounds):
        sample_indices = rng.choice(len(members), size=sample_size, replace=True)
        sample = [members[int(index)] for index in sample_indices]
        refined = _refine_candidate(sample, candidate)
        if candidate.kind == "finite" and candidate.finite_point is not None and refined.finite_point is not None:
            distance = float(np.linalg.norm(candidate.finite_point - refined.finite_point))
            values.append(math.exp(-5.0 * distance / max(diagonal, 1e-9)))
        elif candidate.direction is not None and refined.direction is not None:
            delta = math.degrees(_axis_angle(candidate.direction, refined.direction))
            values.append(math.exp(-delta / 5.0))
    return float(np.mean(values)) if values else 0.0


def _midpoint_coverage(lines: list[_GeometryLine], image_size: tuple[int, int]) -> float:
    width, height = image_size
    grid_size = 4
    cells: set[tuple[int, int]] = set()
    for line in lines:
        col = min(grid_size - 1, max(0, int(line.midpoint[0] / max(width, 1) * grid_size)))
        row = min(grid_size - 1, max(0, int(line.midpoint[1] / max(height, 1) * grid_size)))
        cells.add((row, col))
    return len(cells) / (grid_size * grid_size)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = sorted_weights.sum() / 2.0
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _axis(line: LineRecord) -> np.ndarray:
    vector = np.array(
        [line.p2_analysis.x - line.p1_analysis.x, line.p2_analysis.y - line.p1_analysis.y],
        dtype=float,
    )
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.array([1.0, 0.0])


def _midpoint(line: LineRecord) -> np.ndarray:
    return np.array(
        [(line.p1_analysis.x + line.p2_analysis.x) / 2.0, (line.p1_analysis.y + line.p2_analysis.y) / 2.0],
        dtype=float,
    )


def _residual_to_serialized_family(
    axis: np.ndarray,
    midpoint: np.ndarray,
    family: VPFamily,
) -> float:
    if family.vp_type == "finite" and family.vp_analysis is not None:
        target = np.array([family.vp_analysis.x, family.vp_analysis.y], dtype=float) - midpoint
        norm = float(np.linalg.norm(target))
        if norm <= 1e-9:
            return 90.0
        return math.degrees(_axis_angle(axis, target / norm))
    if family.direction_analysis is not None:
        target = np.array(
            [math.cos(family.direction_analysis), math.sin(family.direction_analysis)]
        )
        return math.degrees(_axis_angle(axis, target))
    return 90.0
