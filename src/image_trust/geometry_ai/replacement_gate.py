"""Independent, fail-closed replacement gate for geometry-origin candidates."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeometryGateSample(BaseModel):
    """One registered score row used by the replacement audit."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    base_sample_id: str
    label: Literal[0, 1]
    split: str
    scene_slice: str
    transformation: str = "original"
    candidate_probability: float = Field(ge=0.0, le=1.0)
    baseline_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class GeometryReplacementGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_threshold: float = Field(ge=0.0, le=1.0)
    baseline_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    recall_lower_bound: float = Field(default=0.55, ge=0.0, le=1.0)
    fpr_upper_bound: float = Field(default=0.25, ge=0.0, le=1.0)
    minimum_positive_scene_slices: int = Field(default=2, ge=1)
    minimum_scene_improvement: float = 0.0
    required_transformations: tuple[str, ...] = ("jpeg", "screenshot", "crop")
    minimum_pairs_per_transformation: int = Field(default=20, ge=1)
    minimum_transformation_agreement: float = Field(default=0.80, ge=0.0, le=1.0)
    maximum_transformation_p90_delta: float = Field(default=0.20, ge=0.0, le=1.0)
    confidence_z: float = Field(default=1.959963984540054, gt=0.0)


class WilsonInterval(BaseModel):
    model_config = ConfigDict(frozen=True)

    successes: int = Field(ge=0)
    total: int = Field(ge=0)
    estimate: float | None = Field(default=None, ge=0.0, le=1.0)
    lower: float | None = Field(default=None, ge=0.0, le=1.0)
    upper: float | None = Field(default=None, ge=0.0, le=1.0)


class GateCriterion(BaseModel):
    model_config = ConfigDict(frozen=True)

    criterion_id: str
    passed: bool
    status: Literal["passed", "failed", "missing"]
    observed: dict[str, Any] = Field(default_factory=dict)
    requirement: str
    reason: str


class GeometryReplacementGateReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "geometry-origin-replacement-gate-v2"
    eligible: bool
    criteria: list[GateCriterion]
    config: GeometryReplacementGateConfig
    sample_count: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_eligibility(self) -> "GeometryReplacementGateReport":
        if self.eligible != all(criterion.passed for criterion in self.criteria):
            raise ValueError("eligible must equal the conjunction of all criteria")
        return self


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> WilsonInterval:
    """Return the two-sided Wilson score interval; missing samples stay missing."""

    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total")
    if total == 0:
        return WilsonInterval(successes=successes, total=total)
    estimate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (estimate + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((estimate * (1.0 - estimate) + z2 / (4.0 * total)) / total) / denominator
    return WilsonInterval(
        successes=successes,
        total=total,
        estimate=float(estimate),
        lower=float(max(0.0, center - margin)),
        upper=float(min(1.0, center + margin)),
    )


def evaluate_geometry_replacement_gate(
    samples: list[GeometryGateSample | dict[str, Any]],
    config: GeometryReplacementGateConfig,
) -> GeometryReplacementGateReport:
    """Evaluate all hard conditions and fail closed for incomplete data."""

    rows = [sample if isinstance(sample, GeometryGateSample) else GeometryGateSample.model_validate(sample) for sample in samples]
    registration_keys = [
        (row.split, row.base_sample_id, row.transformation)
        for row in rows
    ]
    if len(set(registration_keys)) != len(registration_keys):
        raise ValueError(
            "duplicate (split, base_sample_id, transformation) rows would inflate the audit"
        )
    original_holdout = [
        row for row in rows if row.split == "unseen_generator" and row.transformation == "original"
    ]
    ai_rows = [row for row in original_holdout if row.label == 1]
    real_rows = [row for row in original_holdout if row.label == 0]
    recall = wilson_interval(
        sum(row.candidate_probability >= config.candidate_threshold for row in ai_rows),
        len(ai_rows),
        z=config.confidence_z,
    )
    fpr = wilson_interval(
        sum(row.candidate_probability >= config.candidate_threshold for row in real_rows),
        len(real_rows),
        z=config.confidence_z,
    )
    criteria = [
        _interval_criterion(
            "unseen_generator_ai_recall",
            recall,
            bound=config.recall_lower_bound,
            use_lower=True,
        ),
        _interval_criterion(
            "real_false_positive_rate",
            fpr,
            bound=config.fpr_upper_bound,
            use_lower=False,
        ),
        _scene_improvement_criterion(original_holdout, config),
        _transformation_stability_criterion(rows, config),
    ]
    reasons = [criterion.reason for criterion in criteria if not criterion.passed]
    return GeometryReplacementGateReport(
        eligible=all(criterion.passed for criterion in criteria),
        criteria=criteria,
        config=config,
        sample_count=len(rows),
        reasons=reasons,
    )


def _interval_criterion(
    criterion_id: str,
    interval: WilsonInterval,
    *,
    bound: float,
    use_lower: bool,
) -> GateCriterion:
    observed = interval.model_dump(mode="json")
    if interval.total == 0:
        return GateCriterion(
            criterion_id=criterion_id,
            passed=False,
            status="missing",
            observed=observed,
            requirement=(
                f"Wilson 95% lower bound >= {bound:.2f}"
                if use_lower
                else f"Wilson 95% upper bound <= {bound:.2f}"
            ),
            reason=f"{criterion_id}: no registered unseen-generator samples",
        )
    value = interval.lower if use_lower else interval.upper
    assert value is not None
    passed = value >= bound if use_lower else value <= bound
    return GateCriterion(
        criterion_id=criterion_id,
        passed=passed,
        status="passed" if passed else "failed",
        observed=observed,
        requirement=(
            f"Wilson 95% lower bound >= {bound:.2f}"
            if use_lower
            else f"Wilson 95% upper bound <= {bound:.2f}"
        ),
        reason=(
            f"{criterion_id}: passed"
            if passed
            else f"{criterion_id}: confidence bound {value:.4f} missed {bound:.4f}"
        ),
    )


def _scene_improvement_criterion(
    rows: list[GeometryGateSample],
    config: GeometryReplacementGateConfig,
) -> GateCriterion:
    grouped: dict[str, list[GeometryGateSample]] = defaultdict(list)
    for row in rows:
        grouped[row.scene_slice].append(row)
    metrics: dict[str, dict[str, float | int | str]] = {}
    positive = 0
    for scene, scene_rows in sorted(grouped.items()):
        if {row.label for row in scene_rows} != {0, 1}:
            metrics[scene] = {"status": "missing_class", "count": len(scene_rows)}
            continue
        if any(row.baseline_probability is None for row in scene_rows):
            metrics[scene] = {"status": "missing_baseline", "count": len(scene_rows)}
            continue
        candidate_ba = _balanced_accuracy(
            scene_rows,
            threshold=config.candidate_threshold,
            field="candidate_probability",
        )
        baseline_ba = _balanced_accuracy(
            scene_rows,
            threshold=config.baseline_threshold,
            field="baseline_probability",
        )
        improvement = candidate_ba - baseline_ba
        is_positive = improvement > config.minimum_scene_improvement
        positive += int(is_positive)
        metrics[scene] = {
            "status": "qualified",
            "count": len(scene_rows),
            "candidate_balanced_accuracy": candidate_ba,
            "baseline_balanced_accuracy": baseline_ba,
            "improvement": improvement,
            "positive": int(is_positive),
        }
    qualified = sum(metric.get("status") == "qualified" for metric in metrics.values())
    passed = positive >= config.minimum_positive_scene_slices
    status: Literal["passed", "failed", "missing"] = "passed" if passed else ("missing" if qualified < config.minimum_positive_scene_slices else "failed")
    return GateCriterion(
        criterion_id="scene_slice_improvement",
        passed=passed,
        status=status,
        observed={
            "positive_slice_count": positive,
            "qualified_slice_count": qualified,
            "slices": metrics,
        },
        requirement=(
            f"> {config.minimum_scene_improvement:.4f} balanced-accuracy improvement "
            f"in at least {config.minimum_positive_scene_slices} scene slices"
        ),
        reason=(
            "scene_slice_improvement: passed"
            if passed
            else f"scene_slice_improvement: only {positive} qualified positive slices"
        ),
    )


def _transformation_stability_criterion(
    rows: list[GeometryGateSample],
    config: GeometryReplacementGateConfig,
) -> GateCriterion:
    # Robustness is part of the untouched holdout gate.  Training or calibration
    # variants must never substitute for missing unseen-generator perturbations.
    rows = [row for row in rows if row.split == "unseen_generator"]
    originals = {
        (row.split, row.base_sample_id): row
        for row in rows
        if row.transformation == "original"
    }
    metrics: dict[str, dict[str, float | int | str]] = {}
    all_passed = True
    any_missing = False
    for transformation in config.required_transformations:
        pairs = [
            (originals[(row.split, row.base_sample_id)], row)
            for row in rows
            if row.transformation == transformation and (row.split, row.base_sample_id) in originals
        ]
        if len(pairs) < config.minimum_pairs_per_transformation:
            any_missing = True
            all_passed = False
            metrics[transformation] = {
                "status": "missing_pairs",
                "pair_count": len(pairs),
            }
            continue
        agreement = sum(
            (original.candidate_probability >= config.candidate_threshold)
            == (variant.candidate_probability >= config.candidate_threshold)
            for original, variant in pairs
        ) / len(pairs)
        deltas = np.asarray(
            [abs(original.candidate_probability - variant.candidate_probability) for original, variant in pairs],
            dtype=np.float64,
        )
        p90_delta = float(np.quantile(deltas, 0.90, method="higher"))
        passed = (
            agreement >= config.minimum_transformation_agreement
            and p90_delta <= config.maximum_transformation_p90_delta
        )
        all_passed = all_passed and passed
        metrics[transformation] = {
            "status": "passed" if passed else "failed",
            "pair_count": len(pairs),
            "prediction_agreement": float(agreement),
            "probability_delta_p90": p90_delta,
        }
    status: Literal["passed", "failed", "missing"] = "passed" if all_passed else ("missing" if any_missing else "failed")
    return GateCriterion(
        criterion_id="transformation_stability",
        passed=all_passed,
        status=status,
        observed={"transformations": metrics},
        requirement=(
            f"each of {', '.join(config.required_transformations)} has >= "
            f"{config.minimum_pairs_per_transformation} pairs, agreement >= "
            f"{config.minimum_transformation_agreement:.2f}, and p90 probability delta <= "
            f"{config.maximum_transformation_p90_delta:.2f}"
        ),
        reason=(
            "transformation_stability: passed"
            if all_passed
            else "transformation_stability: one or more required transformations failed or were missing"
        ),
    )


def _balanced_accuracy(
    rows: list[GeometryGateSample],
    *,
    threshold: float,
    field: Literal["candidate_probability", "baseline_probability"],
) -> float:
    positive = [row for row in rows if row.label == 1]
    negative = [row for row in rows if row.label == 0]
    true_positive_rate = sum(float(getattr(row, field)) >= threshold for row in positive) / len(positive)
    true_negative_rate = sum(float(getattr(row, field)) < threshold for row in negative) / len(negative)
    return float((true_positive_rate + true_negative_rate) / 2.0)
