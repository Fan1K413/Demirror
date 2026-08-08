"""P1 global--local camera-consistency measurement without source inference."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable

import numpy as np

from image_trust.camera.backends import CameraBackend, CameraBackendInput
from image_trust.camera.contracts import (
    CameraConsistencyMeasurement,
    CameraConsistencyObservation,
    CameraCropResult,
    CameraEstimate,
    CameraEstimateStatus,
    CameraExperimentResult,
    CameraExperimentRunInfo,
    CameraInputSummary,
    CameraQualityGateConfig,
    CropSpec,
    P1CameraConfig,
)
from image_trust.camera.crops import (
    crop_plan_limitations,
    map_estimate_to_canonical,
    plan_overlapping_crops,
)


_EXCLUSION_LIMITATIONS = {
    "low_texture",
    "special_imaging",
    "uncertain",
    "fisheye",
    "panoramic",
}


def run_camera_consistency_experiment(
    canonical_rgb: np.ndarray,
    input_summary: CameraInputSummary,
    config: P1CameraConfig,
    backend: CameraBackend,
) -> CameraExperimentResult:
    """Run full-image and crop measurements under the frozen P1 protocol."""

    _validate_rgb(canonical_rgb, input_summary.canonical_size)
    canonical_size = input_summary.canonical_size
    run = CameraExperimentRunInfo(
        run_id=f"p1-{input_summary.sha256[:16]}",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config_version=config.config_version,
        config_digest=camera_config_digest(config),
        requested_backend=config.camera_backend.name,
    )
    full_image = map_estimate_to_canonical(
        backend.estimate(
            CameraBackendInput(
                image_rgb=canonical_rgb,
                canonical_size=canonical_size,
            )
        ),
        crop=None,
    )
    crops = plan_overlapping_crops(canonical_size, config.crop_protocol)
    crop_results: list[CameraCropResult] = []
    for crop in crops:
        pixels = canonical_rgb[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
        estimate = map_estimate_to_canonical(
            backend.estimate(
                CameraBackendInput(
                    image_rgb=pixels,
                    canonical_size=canonical_size,
                    crop=crop,
                )
            ),
            crop,
        )
        qualified, reasons = qualifies_for_e_cam(estimate, config.quality_gate)
        crop_results.append(
            CameraCropResult(
                crop=crop,
                estimate=estimate,
                qualified_for_e_cam=qualified,
                exclusion_reasons=reasons,
            )
        )
    full_qualified, full_reasons = qualifies_for_e_cam(full_image, config.quality_gate)
    e_cam = compute_e_cam(
        full_image=full_image,
        full_qualified=full_qualified,
        full_exclusion_reasons=full_reasons,
        crops=crop_results,
        canonical_size=canonical_size,
        gate=config.quality_gate,
    )
    limitations = [
        "p1_e_cam_is_an_uncalibrated_camera_measurement_not_source_evidence",
        *crop_plan_limitations(crops),
    ]
    if full_image.status is not CameraEstimateStatus.OK:
        limitations.append("full_image_camera_estimate_not_available")
    return CameraExperimentResult(
        run=run,
        input=input_summary,
        full_image=full_image,
        crops=crop_results,
        e_cam=e_cam,
        limitations=sorted(set(limitations)),
    )


def qualifies_for_e_cam(
    estimate: CameraEstimate,
    gate: CameraQualityGateConfig,
) -> tuple[bool, list[str]]:
    """Gate camera measurements before they are allowed into E_cam."""

    reasons: list[str] = []
    if estimate.status is not CameraEstimateStatus.OK:
        reasons.append(f"estimate_status:{estimate.status.value}")
    if estimate.applicability < gate.min_applicability:
        reasons.append("applicability_below_gate")
    if estimate.coverage < gate.min_coverage:
        reasons.append("coverage_below_gate")
    if estimate.uncertainty.overall is None:
        reasons.append("uncertainty_not_reported")
    elif estimate.uncertainty.overall > gate.max_uncertainty:
        reasons.append("uncertainty_above_gate")
    for limitation in estimate.limitations:
        normalized = limitation.lower().replace("-", "_")
        if normalized in _EXCLUSION_LIMITATIONS:
            reasons.append(f"excluded:{normalized}")
    return not reasons, sorted(set(reasons))


def compute_e_cam(
    *,
    full_image: CameraEstimate,
    full_qualified: bool,
    full_exclusion_reasons: list[str],
    crops: Iterable[CameraCropResult],
    canonical_size: tuple[int, int],
    gate: CameraQualityGateConfig,
) -> CameraConsistencyMeasurement:
    """Compute a transparent, threshold-free global--local discrepancy value."""

    qualified = [crop for crop in crops if crop.qualified_for_e_cam]
    if not full_qualified or len(qualified) < gate.min_qualified_crops:
        limitations = [
            "e_cam_requires_a_qualified_full_image_and_at_least_three_qualified_crops",
        ]
        if not full_qualified:
            limitations.extend(f"full_image_excluded:{reason}" for reason in full_exclusion_reasons)
        if len(qualified) < gate.min_qualified_crops:
            limitations.append(
                f"qualified_crop_count:{len(qualified)}<{gate.min_qualified_crops}"
            )
        return CameraConsistencyMeasurement(
            observation=CameraConsistencyObservation.NOT_OBSERVED,
            qualified_crop_ids=[crop.crop.crop_id for crop in qualified],
            required_qualified_crops=gate.min_qualified_crops,
            limitations=sorted(set(limitations)),
        )
    component_values: dict[str, list[float]] = defaultdict(list)
    crop_values: list[float] = []
    for crop in qualified:
        components = _estimate_distance_components(
            full_image,
            crop.estimate,
            canonical_size,
        )
        if not components:
            continue
        for name, value in components.items():
            component_values[name].append(value)
        crop_values.append(sum(components.values()) / len(components))
    if not crop_values:
        return CameraConsistencyMeasurement(
            observation=CameraConsistencyObservation.NOT_OBSERVED,
            qualified_crop_ids=[crop.crop.crop_id for crop in qualified],
            required_qualified_crops=gate.min_qualified_crops,
            limitations=["qualified_estimates_have_no_comparable_camera_fields"],
        )
    return CameraConsistencyMeasurement(
        observation=CameraConsistencyObservation.MEASURED,
        value=sum(crop_values) / len(crop_values),
        qualified_crop_ids=[crop.crop.crop_id for crop in qualified],
        required_qualified_crops=gate.min_qualified_crops,
        component_means={
            name: sum(values) / len(values)
            for name, values in sorted(component_values.items())
        },
        limitations=[
            "e_cam_has_no_calibrated_source-decision_threshold",
            "e_cam_is_not_an_ai_generation_score",
        ],
    )


def _estimate_distance_components(
    full: CameraEstimate,
    crop: CameraEstimate,
    canonical_size: tuple[int, int],
) -> dict[str, float]:
    values: dict[str, float] = {}
    if full.roll is not None and crop.roll is not None:
        values["roll_circular_rad"] = _clip01(_circular_distance(full.roll, crop.roll) / math.pi)
    if full.pitch is not None and crop.pitch is not None:
        values["pitch_rad"] = _clip01(abs(full.pitch - crop.pitch) / math.pi)
    if full.vfov_or_focal is not None and crop.vfov_or_focal is not None:
        if (
            full.vfov_or_focal.kind is crop.vfov_or_focal.kind
            and full.vfov_or_focal.reference == "camera"
            and crop.vfov_or_focal.reference == "camera"
        ):
            if full.vfov_or_focal.kind.value == "vfov_deg":
                values["vfov_deg"] = _clip01(
                    abs(full.vfov_or_focal.value - crop.vfov_or_focal.value) / 180.0
                )
            else:
                diagonal = math.hypot(*canonical_size)
                values["focal_normalized_diagonal"] = _clip01(
                    abs(full.vfov_or_focal.value - crop.vfov_or_focal.value) / diagonal
                )
    diagonal = math.hypot(*canonical_size)
    if full.principal_point is not None and crop.principal_point is not None:
        values["principal_point_normalized_diagonal"] = _clip01(
            math.dist(
                (full.principal_point.x, full.principal_point.y),
                (crop.principal_point.x, crop.principal_point.y),
            )
            / diagonal
        )
    if full.horizon is not None and crop.horizon is not None:
        values["horizon_normalized_diagonal"] = _horizon_distance(
            full.horizon.p1,
            full.horizon.p2,
            crop.horizon.p1,
            crop.horizon.p2,
            canonical_size,
        )
    return values


def _horizon_distance(
    first_a, first_b, second_a, second_b, canonical_size: tuple[int, int]
) -> float:
    width, height = canonical_size
    del height
    samples = (0.0, width / 2.0, float(width))
    distances: list[float] = []
    for x in samples:
        first_y = _line_y_at_x(first_a, first_b, x)
        second_y = _line_y_at_x(second_a, second_b, x)
        if first_y is not None and second_y is not None:
            distances.append(abs(first_y - second_y))
    if not distances:
        return 1.0
    return _clip01((sum(distances) / len(distances)) / math.hypot(*canonical_size))


def _line_y_at_x(first, second, x: float) -> float | None:
    delta_x = second.x - first.x
    if abs(delta_x) < 1e-9:
        return None
    return first.y + (x - first.x) * (second.y - first.y) / delta_x


def _circular_distance(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_rgb(image: np.ndarray, canonical_size: tuple[int, int]) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("canonical_rgb must be an HxWx3 RGB array.")
    width, height = canonical_size
    if image.shape[:2] != (height, width):
        raise ValueError("canonical_size must match canonical_rgb dimensions.")


def camera_config_digest(config: P1CameraConfig) -> str:
    """Return the stable digest recorded by every P1 experiment run."""

    payload = config.model_dump_json(exclude_none=False)
    return sha256(payload.encode("utf-8")).hexdigest()
