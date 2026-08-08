"""Descriptive P1 calibration summaries with no fitted decision threshold."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from image_trust.camera.contracts import (
    CameraBackendProvenance,
    CameraCalibrationBackendIdentity,
    CameraCalibrationDecisionReadiness,
    CameraCalibrationSummary,
    CameraCropResult,
    CoordinateSpace,
    CameraEstimate,
    CameraExperimentResult,
    CameraGateAudit,
    CameraQualityGateConfig,
    NumericDistributionSummary,
    P1CameraConfig,
)
from image_trust.camera.crops import plan_overlapping_crops
from image_trust.camera.experiment import (
    camera_config_digest,
    compute_e_cam,
    qualifies_for_e_cam,
)


def load_camera_experiment_results(paths: Iterable[Path]) -> list[CameraExperimentResult]:
    """Load result JSON files emitted by ``camera-analyze``."""

    return [
        CameraExperimentResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    ]


def summarize_camera_calibration(
    results: Iterable[CameraExperimentResult],
    config: P1CameraConfig,
    cohort_name: str,
    *,
    result_filenames: Iterable[str] = (),
) -> CameraCalibrationSummary:
    """Summarize one strictly homogeneous camera-calibration cohort.

    This function deliberately reports distributions and gate exclusions only.
    It never fits, recommends, or serializes an ``E_cam`` decision threshold.
    """

    collected = list(results)
    if not collected:
        raise ValueError("At least one camera result is required for calibration summary.")
    expected_digest = camera_config_digest(config)
    backend_identity = _validate_homogeneous_results(
        collected,
        config,
        expected_digest,
    )

    full_qualified = [
        qualifies_for_e_cam(result.full_image, config.quality_gate)[0]
        for result in collected
    ]
    crops = [crop for result in collected for crop in result.crops]
    e_cam_counts = Counter(result.e_cam.observation.value for result in collected)
    e_cam_values = [
        result.e_cam.value for result in collected if result.e_cam.value is not None
    ]

    return CameraCalibrationSummary(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        cohort_name=cohort_name,
        config_version=config.config_version,
        config_digest=expected_digest,
        requested_backend=config.camera_backend.name,
        operational_quality_gate=config.quality_gate,
        result_count=len(collected),
        unique_image_count=len({result.input.sha256 for result in collected}),
        result_filenames=sorted(result_filenames),
        backend_identity=backend_identity,
        full_image_gate=_summarize_estimates(
            (result.full_image for result in collected),
            config.quality_gate,
            qualified=full_qualified,
        ),
        crop_gate=_summarize_crops(crops, config.quality_gate),
        e_cam_observation_counts=dict(sorted(e_cam_counts.items())),
        e_cam_value_distribution=_distribution(e_cam_values),
        decision_readiness=CameraCalibrationDecisionReadiness(
            requirements_before_registration=[
                "register_an_independent_calibration_cohort_with_provenance_and_family_splits",
                "freeze_backend_specific_gate_parameters_before_held_out_evaluation",
                "pre_register_e_cam_thresholds_and_statistical_tests_before_any_decision_use",
            ],
            limitations=[
                "descriptive_cohort_summary_does_not_fit_or_select_e_cam_thresholds",
                "e_cam_is_not_a_source_authenticity_or_ai_generation_score",
            ],
        ),
        limitations=[
            "results_must_share_one_config_digest_and_requested_backend",
            "duplicate_input_hashes_are_rejected_to_avoid_counting_one_image_twice",
            "operational_quality_gate_values_are_recorded_for_audit_not_validated_as_decision_thresholds",
        ],
    )


def write_camera_calibration_summary(path: Path, summary: CameraCalibrationSummary) -> None:
    """Write a stable JSON calibration artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_homogeneous_results(
    results: list[CameraExperimentResult],
    config: P1CameraConfig,
    expected_digest: str,
) -> CameraCalibrationBackendIdentity:
    hashes = [result.input.sha256 for result in results]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate input SHA-256 values are not allowed in one cohort summary.")
    provenance: list[CameraBackendProvenance] = []
    for index, result in enumerate(results, start=1):
        if result.run.config_digest != expected_digest:
            raise ValueError(
                f"Result {index} config digest does not match the supplied P1 configuration."
            )
        if result.run.config_version != config.config_version:
            raise ValueError(
                f"Result {index} config version does not match the supplied P1 configuration."
            )
        if result.run.requested_backend != config.camera_backend.name:
            raise ValueError(
                f"Result {index} requested backend does not match the supplied P1 configuration."
            )
        _validate_recorded_gate_results(result, config, index)
        provenance.extend(
            [
                result.full_image.provenance,
                *(crop.estimate.provenance for crop in result.crops),
            ]
        )
    stable_identities = {_stable_provenance_identity(item) for item in provenance}
    if len(stable_identities) != 1:
        raise ValueError(
            "Results do not share one backend/model/weights provenance identity."
        )
    backend_id, backend_version, model_commit, weights_sha256, weights_license = (
        stable_identities.pop()
    )
    return CameraCalibrationBackendIdentity(
        backend_id=backend_id,
        backend_version=backend_version,
        model_commit=model_commit,
        weights_sha256=weights_sha256,
        weights_license=weights_license,
        inference_devices=sorted({item.inference_device for item in provenance}),
        requested_inference_devices=sorted(
            {
                item.requested_inference_device
                for item in provenance
                if item.requested_inference_device is not None
            }
        ),
    )


def _validate_recorded_gate_results(
    result: CameraExperimentResult,
    config: P1CameraConfig,
    result_index: int,
) -> None:
    expected_crops = plan_overlapping_crops(
        result.input.canonical_size,
        config.crop_protocol,
    )
    actual_crops = [crop.crop for crop in result.crops]
    if actual_crops != expected_crops:
        raise ValueError(
            f"Result {result_index} crops do not match the registered P1 crop protocol."
        )
    estimates = [result.full_image, *(crop.estimate for crop in result.crops)]
    if any(
        estimate.coordinate_space is not CoordinateSpace.CANONICAL
        for estimate in estimates
    ):
        raise ValueError(
            f"Result {result_index} contains estimates outside canonical coordinates."
        )
    full_qualified, full_reasons = qualifies_for_e_cam(
        result.full_image,
        config.quality_gate,
    )
    for crop in result.crops:
        qualified, reasons = qualifies_for_e_cam(crop.estimate, config.quality_gate)
        if crop.qualified_for_e_cam != qualified or crop.exclusion_reasons != reasons:
            raise ValueError(
                f"Result {result_index} has a crop gate result inconsistent with its estimate."
            )
    expected_e_cam = compute_e_cam(
        full_image=result.full_image,
        full_qualified=full_qualified,
        full_exclusion_reasons=full_reasons,
        crops=result.crops,
        canonical_size=result.input.canonical_size,
        gate=config.quality_gate,
    )
    if result.e_cam.model_dump(mode="json") != expected_e_cam.model_dump(mode="json"):
        raise ValueError(
            f"Result {result_index} has E_cam data inconsistent with its estimates and gate."
        )


def _stable_provenance_identity(
    provenance: CameraBackendProvenance,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    return (
        provenance.backend_id,
        provenance.backend_version,
        provenance.model_commit,
        provenance.weights_sha256,
        provenance.weights_license,
    )


def _summarize_crops(
    crops: list[CameraCropResult], gate: CameraQualityGateConfig
) -> CameraGateAudit:
    estimates = [crop.estimate for crop in crops]
    qualified = [crop.qualified_for_e_cam for crop in crops]
    exclusions = Counter(
        reason for crop in crops for reason in crop.exclusion_reasons
    )
    return _summarize_estimates(
        estimates,
        gate,
        qualified=qualified,
        exclusion_counts=exclusions,
    )


def _summarize_estimates(
    estimates: Iterable[CameraEstimate],
    gate: CameraQualityGateConfig,
    *,
    qualified: Iterable[bool],
    exclusion_counts: Counter[str] | None = None,
) -> CameraGateAudit:
    collected = list(estimates)
    qualified_values = list(qualified)
    if len(collected) != len(qualified_values):
        raise ValueError("Estimate and qualification counts must be identical.")
    statuses = Counter(estimate.status.value for estimate in collected)
    if exclusion_counts is None:
        exclusion_counts = Counter(
            reason
            for estimate in collected
            for reason in qualifies_for_e_cam(estimate, gate)[1]
        )
    metrics: dict[str, list[float]] = defaultdict(list)
    for estimate in collected:
        metrics["applicability"].append(estimate.applicability)
        metrics["coverage"].append(estimate.coverage)
        if estimate.uncertainty.overall is not None:
            metrics["uncertainty_overall"].append(estimate.uncertainty.overall)
    return CameraGateAudit(
        estimate_count=len(collected),
        status_counts=dict(sorted(statuses.items())),
        qualified_count=sum(qualified_values),
        exclusion_reason_counts=dict(sorted(exclusion_counts.items())),
        metric_distributions={
            name: _distribution(values) for name, values in sorted(metrics.items())
        },
    )


def _distribution(values: Iterable[float | None]) -> NumericDistributionSummary:
    collected = sorted(float(value) for value in values if value is not None)
    if not collected:
        return NumericDistributionSummary(count=0)
    return NumericDistributionSummary(
        count=len(collected),
        minimum=collected[0],
        maximum=collected[-1],
        mean=sum(collected) / len(collected),
        p50=_quantile(collected, 0.50),
        p90=_quantile(collected, 0.90),
    )


def _quantile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
