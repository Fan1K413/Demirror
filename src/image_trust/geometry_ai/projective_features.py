"""Interpretable projective-consistency features from detected line segments.

Unlike a line-set appearance classifier, these measurements explicitly ask
whether several line families can be explained by shared vanishing points and
whether those explanations remain stable across the image.  They are still
scene-conditional measurements and require calibration before source use.
"""

from __future__ import annotations

import itertools
import math
from collections import OrderedDict

import numpy as np


MAX_FAMILIES = 4
INLIER_DEGREES = 2.5
MIN_FAMILY_LINES = 4


def projective_consistency_features(
    lines: np.ndarray,
    size: tuple[int, int],
) -> OrderedDict[str, float]:
    """Return a deterministic fixed-length projective feature vector."""
    width, height = size
    raw = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    raw = raw[np.isfinite(raw).all(axis=1)]
    output = _empty_features()
    if len(raw) < MIN_FAMILY_LINES:
        output["line_count_normalized"] = min(1.0, len(raw) / 250.0)
        return output

    points = raw.reshape(-1, 2, 2).copy()
    points[:, :, 0] = points[:, :, 0] / max(width - 1.0, 1.0) * 2.0 - 1.0
    points[:, :, 1] = points[:, :, 1] / max(height - 1.0, 1.0) * 2.0 - 1.0
    delta = points[:, 1] - points[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > 1e-6
    points = points[valid]
    delta = delta[valid]
    lengths = lengths[valid]
    if len(points) < MIN_FAMILY_LINES:
        output["line_count_normalized"] = min(1.0, len(points) / 250.0)
        return output
    directions = delta / lengths[:, None]
    midpoints = points.mean(axis=1)
    homogeneous_points = np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=2)
    equations = np.cross(homogeneous_points[:, 0], homogeneous_points[:, 1])
    equations /= np.linalg.norm(equations[:, :2], axis=1, keepdims=True).clip(1e-9)
    weights = lengths / lengths.sum().clip(1e-9)

    families = _fit_families(equations, directions, midpoints, weights)
    output["line_count_normalized"] = min(1.0, len(points) / 250.0)
    output["family_count_normalized"] = len(families) / MAX_FAMILIES
    explained = sum(float(weights[family["members"]].sum()) for family in families)
    output["explained_length_ratio"] = min(1.0, explained)
    output["unexplained_length_ratio"] = max(0.0, 1.0 - explained)

    all_residuals: list[float] = []
    all_drifts: list[float] = []
    for family_index in range(MAX_FAMILIES):
        prefix = f"family_{family_index + 1}"
        if family_index >= len(families):
            continue
        family = families[family_index]
        members = family["members"]
        residuals = family["residuals"][members]
        vp = family["vp"]
        all_residuals.extend(residuals.tolist())
        support = float(weights[members].sum())
        output[f"{prefix}_length_support"] = support
        output[f"{prefix}_count_support"] = float(members.mean())
        output[f"{prefix}_residual_p50"] = float(np.median(residuals) / 15.0)
        output[f"{prefix}_residual_p90"] = float(np.quantile(residuals, 0.9) / 15.0)
        output[f"{prefix}_spatial_support"] = _spatial_support(midpoints[members])
        finite = abs(vp[2]) > 1e-6
        output[f"{prefix}_finite"] = float(finite)
        if finite:
            finite_vp = vp[:2] / vp[2]
            output[f"{prefix}_vp_radius"] = min(10.0, float(np.linalg.norm(finite_vp))) / 10.0
        else:
            output[f"{prefix}_vp_radius"] = 1.0
        drifts = _local_vp_drifts(equations[members], midpoints[members], vp)
        if drifts:
            output[f"{prefix}_local_drift_p50"] = float(np.median(drifts) / 45.0)
            output[f"{prefix}_local_drift_max"] = float(np.max(drifts) / 45.0)
            all_drifts.extend(drifts)

    if all_residuals:
        output["family_residual_mean"] = float(np.mean(all_residuals) / 15.0)
        output["family_residual_p90"] = float(np.quantile(all_residuals, 0.9) / 15.0)
    if all_drifts:
        output["local_vp_drift_mean"] = float(np.mean(all_drifts) / 45.0)
        output["local_vp_drift_max"] = float(np.max(all_drifts) / 45.0)
    output.update(_manhattan_features([family["vp"] for family in families]))
    return output


def _empty_features() -> OrderedDict[str, float]:
    values: OrderedDict[str, float] = OrderedDict(
        (
            key,
            0.0,
        )
        for key in (
            "line_count_normalized",
            "family_count_normalized",
            "explained_length_ratio",
            "unexplained_length_ratio",
        )
    )
    for family_index in range(MAX_FAMILIES):
        prefix = f"family_{family_index + 1}"
        for suffix in (
            "length_support",
            "count_support",
            "residual_p50",
            "residual_p90",
            "spatial_support",
            "finite",
            "vp_radius",
            "local_drift_p50",
            "local_drift_max",
        ):
            values[f"{prefix}_{suffix}"] = 0.0
    for key in (
        "family_residual_mean",
        "family_residual_p90",
        "local_vp_drift_mean",
        "local_vp_drift_max",
        "finite_vp_pair_count_normalized",
        "orthogonal_positive_ratio",
        "orthogonal_focal_cv",
        "orthogonal_residual_mean",
        "orthogonal_residual_min",
        "best_manhattan_triad_residual",
    ):
        values[key] = 0.0
    return values


def _fit_families(
    equations: np.ndarray,
    directions: np.ndarray,
    midpoints: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    remaining = np.ones(len(equations), dtype=bool)
    families: list[dict[str, np.ndarray]] = []
    for _ in range(MAX_FAMILIES):
        indices = np.flatnonzero(remaining)
        if len(indices) < MIN_FAMILY_LINES:
            break
        # Long lines provide the most stable pair intersections.  Evaluating
        # all pairs among at most 72 candidates is deterministic and bounded.
        ranked = indices[np.argsort(weights[indices])[::-1]][:72]
        pair_rows = np.asarray(list(itertools.combinations(ranked, 2)), dtype=np.int32)
        if not len(pair_rows):
            break
        direction_dot = np.abs(np.sum(directions[pair_rows[:, 0]] * directions[pair_rows[:, 1]], axis=1))
        pair_rows = pair_rows[direction_dot < math.cos(math.radians(4.0))]
        if not len(pair_rows):
            break
        candidates = np.cross(equations[pair_rows[:, 0]], equations[pair_rows[:, 1]])
        norms = np.linalg.norm(candidates, axis=1)
        candidates = candidates[norms > 1e-8]
        if not len(candidates):
            break
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True).clip(1e-9)

        best_score = -1.0
        best_median = float("inf")
        best_vp: np.ndarray | None = None
        best_members: np.ndarray | None = None
        for start in range(0, len(candidates), 256):
            batch = candidates[start : start + 256]
            residuals = _residual_matrix(batch, directions, midpoints)
            inliers = (residuals <= INLIER_DEGREES) & remaining[None, :]
            support = inliers @ weights
            for local_index in np.argsort(support)[-8:]:
                members = inliers[local_index]
                if members.sum() < MIN_FAMILY_LINES:
                    continue
                median = float(np.median(residuals[local_index, members]))
                score = float(support[local_index])
                if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12 and median < best_median
                ):
                    best_score = score
                    best_median = median
                    best_vp = batch[local_index]
                    best_members = members
        if best_vp is None or best_members is None:
            break
        refined = _refit_vp(equations[best_members], weights[best_members])
        residuals = _residual_matrix(refined[None, :], directions, midpoints)[0]
        members = (residuals <= INLIER_DEGREES) & remaining
        if members.sum() < MIN_FAMILY_LINES or float(weights[members].sum()) < 0.025:
            break
        families.append({"vp": refined, "members": members, "residuals": residuals})
        remaining[members] = False
    return families


def _residual_matrix(
    vps: np.ndarray,
    directions: np.ndarray,
    midpoints: np.ndarray,
) -> np.ndarray:
    residuals = np.empty((len(vps), len(directions)), dtype=np.float64)
    for index, vp in enumerate(vps):
        if abs(vp[2]) > 1e-7:
            finite = vp[:2] / vp[2]
            rays = finite[None, :] - midpoints
            rays /= np.linalg.norm(rays, axis=1, keepdims=True).clip(1e-9)
        else:
            direction = vp[:2] / np.linalg.norm(vp[:2]).clip(1e-9)
            rays = np.broadcast_to(direction, directions.shape)
        sine = np.abs(directions[:, 0] * rays[:, 1] - directions[:, 1] * rays[:, 0])
        residuals[index] = np.degrees(np.arcsin(np.clip(sine, 0.0, 1.0)))
    return residuals


def _refit_vp(equations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normal = equations.T @ (weights[:, None] * equations)
    _, vectors = np.linalg.eigh(normal)
    vp = vectors[:, 0]
    return vp / np.linalg.norm(vp).clip(1e-9)


def _spatial_support(midpoints: np.ndarray) -> float:
    cells = np.floor((midpoints + 1.0) * 0.5 * 3.0).astype(int)
    cells = np.clip(cells, 0, 2)
    return len({(int(x), int(y)) for x, y in cells}) / 9.0


def _local_vp_drifts(
    equations: np.ndarray,
    midpoints: np.ndarray,
    global_vp: np.ndarray,
) -> list[float]:
    cells = np.floor((midpoints + 1.0) * 0.5 * 2.0).astype(int)
    cells = np.clip(cells, 0, 1)
    result: list[float] = []
    for x in range(2):
        for y in range(2):
            selected = (cells[:, 0] == x) & (cells[:, 1] == y)
            if selected.sum() < 3:
                continue
            local = _refit_vp(equations[selected], np.ones(selected.sum()))
            cosine = float(np.clip(abs(np.dot(global_vp, local)), 0.0, 1.0))
            result.append(math.degrees(math.acos(cosine)))
    return result


def _manhattan_features(vps: list[np.ndarray]) -> dict[str, float]:
    finite = [vp[:2] / vp[2] for vp in vps if abs(vp[2]) > 1e-6]
    pairs = list(itertools.combinations(finite, 2))
    if not pairs:
        return {
            "finite_vp_pair_count_normalized": 0.0,
            "orthogonal_positive_ratio": 0.0,
            "orthogonal_focal_cv": 0.0,
            "orthogonal_residual_mean": 0.0,
            "orthogonal_residual_min": 0.0,
            "best_manhattan_triad_residual": 0.0,
        }
    f2 = np.asarray([-float(np.dot(first, second)) for first, second in pairs])
    positive = f2[f2 > 1e-8]
    focal = float(np.median(positive)) if len(positive) else 1.0
    residuals = [
        abs(float(np.dot(first, second)) + focal)
        / math.sqrt((float(np.dot(first, first)) + focal) * (float(np.dot(second, second)) + focal))
        for first, second in pairs
    ]
    triad_residuals: list[float] = []
    for triad in itertools.combinations(finite, 3):
        triad_pairs = list(itertools.combinations(triad, 2))
        triad_f2 = np.asarray([-float(np.dot(first, second)) for first, second in triad_pairs])
        triad_positive = triad_f2[triad_f2 > 1e-8]
        if not len(triad_positive):
            continue
        triad_focal = float(np.median(triad_positive))
        triad_residuals.append(
            float(
                np.mean(
                    [
                        abs(float(np.dot(first, second)) + triad_focal)
                        / math.sqrt(
                            (float(np.dot(first, first)) + triad_focal)
                            * (float(np.dot(second, second)) + triad_focal)
                        )
                        for first, second in triad_pairs
                    ]
                )
            )
        )
    cv = float(np.std(positive) / max(np.mean(positive), 1e-8)) if len(positive) > 1 else 0.0
    return {
        "finite_vp_pair_count_normalized": min(1.0, len(pairs) / 6.0),
        "orthogonal_positive_ratio": len(positive) / len(pairs),
        "orthogonal_focal_cv": min(10.0, cv) / 10.0,
        "orthogonal_residual_mean": float(np.mean(residuals)),
        "orthogonal_residual_min": float(np.min(residuals)),
        "best_manhattan_triad_residual": min(triad_residuals) if triad_residuals else 1.0,
    }
