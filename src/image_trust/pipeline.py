"""End-to-end P0 analysis orchestration."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from image_trust.geometry.applicability import assess_applicability
from image_trust.geometry.curve_filter import suppress_curve_fragments
from image_trust.geometry.line_backend import BackendUnavailableError, RawLine, resolve_backend
from image_trust.geometry.metrics import compute_line_metrics
from image_trust.geometry.overlays import write_overlays
from image_trust.geometry.vanishing_points import (
    fit_parallel_families,
    fit_local_parallel_families,
    fit_vanishing_families,
    identify_anomaly_candidates,
)
from image_trust.ingest import InputRejectedError, ingest_image
from image_trust.schemas import (
    AnalysisResult,
    Diagnostics,
    Direction,
    Evidence,
    LineRecord,
    Observation,
    P0Config,
    Point,
    RunInfo,
    RunStatus,
)


def analyze_image(
    input_path: Path,
    config: P0Config,
    output_dir: Path,
) -> AnalysisResult:
    """Analyze one local image and write P0 artifacts to output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run = _new_run(config)
    warnings: list[str] = []
    try:
        ingest_started = time.perf_counter()
        ingested = ingest_image(
            input_path,
            config.ingest,
            config.analysis.max_long_side,
        )
        run = run.model_copy(
            update={
                "run_id": f"run-{ingested.summary.sha256[:16]}",
                "deterministic_seed": _seed_from_hash(
                    ingested.summary.sha256, config.config_version
                ),
            }
        )
        ingest_ms = _elapsed_ms(ingest_started)

        prepare_started = time.perf_counter()
        analysis_rgb = _resize_rgb(
            ingested.canonical_rgb,
            ingested.summary.analysis_size,
        )
        grayscale = cv2.cvtColor(analysis_rgb, cv2.COLOR_RGB2GRAY)
        prepare_ms = _elapsed_ms(prepare_started)

        backend_started = time.perf_counter()
        backend = resolve_backend(config.line_backend)
        extraction = backend.extract(grayscale)
        run = run.model_copy(
            update={
                "resolved_backend": extraction.backend_id,
                "fallback_reason": getattr(backend, "fallback_reason", None),
                "dependency_versions": {
                    **run.dependency_versions,
                    extraction.backend_id: extraction.backend_version,
                },
            }
        )
        warnings.extend(extraction.warnings)
        all_lines = _materialize_lines(
            extraction.lines,
            ingested.transform,
            config.line_backend.min_length_px,
            config.line_backend.min_quality,
            max_lines=None,
        )
        curve_filter = suppress_curve_fragments(all_lines, config.line_backend)
        lines = _reindex_lines(
            curve_filter.lines[: config.line_backend.max_lines]
        )
        if curve_filter.suppressed_line_ids:
            warnings.append(
                f"curve_fragments_suppressed:{len(curve_filter.suppressed_line_ids)}"
            )
        backend_ms = _elapsed_ms(backend_started)

        metric_started = time.perf_counter()
        metrics = compute_line_metrics(
            lines,
            ingested.summary.analysis_size,
            config.applicability.grid_size,
            config.applicability.direction_bins,
        )
        assessment = assess_applicability(
            metrics,
            config.applicability,
            ingested.limitations,
        )
        families = fit_vanishing_families(
            lines,
            ingested.summary.analysis_size,
            ingested.transform,
            config.vanishing_points,
            run.deterministic_seed or 0,
        )
        parallel_families = fit_parallel_families(
            lines,
            ingested.summary.analysis_size,
            ingested.transform,
            config.vanishing_points,
            run.deterministic_seed or 0,
        )
        stable_global_families = [family for family in families if family.stable]
        local_direction_families = fit_local_parallel_families(
            lines,
            ingested.summary.analysis_size,
            ingested.transform,
            config.vanishing_points,
            run.deterministic_seed or 0,
            excluded_line_ids={
                line_id
                for family in stable_global_families
                for line_id in family.member_line_ids
            },
        )
        anomalies = identify_anomaly_candidates(
            lines,
            families,
            ingested.summary.analysis_size,
            config.vanishing_points,
            assessment.score,
            config.applicability.anomaly_min_applicability,
            explained_line_ids={
                line_id
                for family in parallel_families
                if family.stable
                for line_id in family.member_line_ids
            }
            | {
                line_id
                for family in local_direction_families
                if family.stable
                for line_id in family.member_line_ids
            },
        )
        metric_ms = _elapsed_ms(metric_started)

        overlay_started = time.perf_counter()
        lines_path = output_dir / "lines_overlay.png"
        anomalies_path = output_dir / "anomalous_lines_overlay.png"
        write_overlays(
            ingested.canonical_rgb,
            lines,
            [*stable_global_families, *local_direction_families]
            or parallel_families,
            anomalies,
            config.overlays,
            lines_path,
            anomalies_path,
        )
        _write_json(output_dir / "lines.json", [line.model_dump(mode="json") for line in lines])
        overlay_ms = _elapsed_ms(overlay_started)

        if assessment.special_imaging or assessment.low_information:
            run_status = RunStatus.NOT_APPLICABLE
        else:
            run_status = RunStatus.OK
        stable_families = stable_global_families
        if (
            assessment.special_imaging
            or assessment.low_information
            or assessment.score < config.applicability.anomaly_min_applicability
            or not stable_families
        ):
            observation = Observation.NOT_OBSERVED
        elif anomalies:
            observation = Observation.POSITIVE
        else:
            observation = Observation.NEGATIVE
        limitations = sorted(
            set(
                [
                    *assessment.limitations,
                    "p0_geometry_is_uncalibrated_not_ai_evidence",
                    "opencv_lsd_quality_is_backend_relative_not_probability",
                    "special_imaging_is_only_metadata_gated_in_p0",
                    "special_imaging_not_assessed_without_metadata_or_manual_tag",
                ]
            )
        )
        effective_line_support = min(
            1.0,
            metrics.line_count / max(config.applicability.target_line_count, 1),
        )
        coverage = _measurement_coverage(
            spatial_coverage=metrics.spatial_coverage,
            effective_line_support=effective_line_support,
        )
        reliability = 0.0 if run_status is RunStatus.NOT_APPLICABLE else coverage
        result = AnalysisResult(
            run=run,
            input=ingested.summary,
            evidence=Evidence(
                run_status=run_status,
                observation=observation,
                direction=Direction.NEUTRAL,
                raw_score=None,
                applicability=assessment.score,
                coverage=coverage,
                reliability=reliability,
                features={
                    "line_count": metrics.line_count,
                    "curve_suppression": {
                        "enabled": config.line_backend.suppress_curve_fragments,
                        "suppressed_line_count": len(curve_filter.suppressed_line_ids),
                        "definition": "short tangent-continuous chains with a broad, consecutive direction sweep are excluded before line capping",
                    },
                    "total_length_normalized": metrics.total_length_normalized,
                    "spatial_coverage": metrics.spatial_coverage,
                    "effective_line_support": effective_line_support,
                    "coverage_definition": {
                        "version": "p0-coverage-v1",
                        "formula": "0.60 * spatial_coverage + 0.40 * effective_line_support",
                        "components": {
                            "spatial_coverage": metrics.spatial_coverage,
                            "effective_line_support": effective_line_support,
                        },
                    },
                    "spatial_entropy": metrics.spatial_entropy,
                    "direction_entropy": metrics.direction_entropy,
                    "occupied_cells": metrics.occupied_cells,
                    "applicability_components": assessment.components,
                    "vp_family_count": len(families),
                    "vp_inlier_ratio": max(
                        (family.weighted_inlier_ratio for family in families),
                        default=0.0,
                    ),
                    "vp_residual_deg": min(
                        (family.weighted_median_residual_deg for family in families),
                        default=None,
                    ),
                    "family_stability": max(
                        (family.bootstrap_stability for family in families),
                        default=0.0,
                    ),
                    "families": [family.model_dump(mode="json") for family in families],
                    "parallel_families": [
                        family.model_dump(mode="json") for family in parallel_families
                    ],
                    "parallel_family_count": len(parallel_families),
                    "parallel_family_definition": {
                        "scope": "image_plane_parallel_orientation",
                        "inlier_angle_deg": config.vanishing_points.parallel_inlier_angle_deg,
                        "limitation": "parallel overlay groups are descriptive and not scene semantics or source evidence",
                    },
                    "local_families": [
                        family.model_dump(mode="json")
                        for family in local_direction_families
                    ],
                    "local_family_count": len(local_direction_families),
                    "local_family_definition": {
                        "scope": "spatial_cell_local_image_plane_parallel_orientation",
                        "grid_size": config.vanishing_points.local_family_grid_size,
                        "families_per_cell": config.vanishing_points.local_direction_families_per_cell,
                        "inlier_angle_deg": config.vanishing_points.local_direction_inlier_angle_deg,
                        "excluded_stable_global_members": True,
                        "limitation": "local direction families are review aids and do not replace global VP measurement or scene semantics",
                    },
                    "anomalous_lines": [
                        anomaly.model_dump(mode="json") for anomaly in anomalies
                    ],
                    "config_snapshot": config.model_dump(mode="json"),
                },
                artifacts=[
                    "lines.json",
                    "lines_overlay.png",
                    "anomalous_lines_overlay.png",
                ],
                limitations=limitations,
            ),
            diagnostics=Diagnostics(
                timing_ms={
                    "ingest": ingest_ms,
                    "prepare": prepare_ms,
                    "line_backend": backend_ms,
                    "metrics_and_vp": metric_ms,
                    "overlays_and_json": overlay_ms,
                    "total": _elapsed_ms(started),
                },
                warnings=sorted(set(warnings)),
            ),
        )
    except InputRejectedError as exc:
        result = AnalysisResult(
            run=run,
            evidence=Evidence(
                run_status=RunStatus.REJECTED,
                observation=Observation.NOT_OBSERVED,
                direction=Direction.NEUTRAL,
                applicability=0.0,
                coverage=0.0,
                reliability=0.0,
                limitations=["input_rejected"],
            ),
            diagnostics=Diagnostics(
                timing_ms={"total": _elapsed_ms(started)},
                errors=[{"code": exc.code, "message": exc.message}],
            ),
        )
    except BackendUnavailableError as exc:
        result = AnalysisResult(
            run=run,
            evidence=Evidence(
                run_status=RunStatus.UNAVAILABLE,
                observation=Observation.NOT_OBSERVED,
                direction=Direction.NEUTRAL,
                applicability=0.0,
                coverage=0.0,
                reliability=0.0,
                limitations=["requested_backend_unavailable"],
            ),
            diagnostics=Diagnostics(
                timing_ms={"total": _elapsed_ms(started)},
                errors=[{"code": f"backend:{exc.backend}", "message": exc.message}],
            ),
        )
    except Exception as exc:
        result = AnalysisResult(
            run=run,
            evidence=Evidence(
                run_status=RunStatus.FAILED,
                observation=Observation.NOT_OBSERVED,
                direction=Direction.NEUTRAL,
                applicability=0.0,
                coverage=0.0,
                reliability=0.0,
                limitations=["analysis_failed"],
            ),
            diagnostics=Diagnostics(
                timing_ms={"total": _elapsed_ms(started)},
                errors=[{"code": type(exc).__name__, "message": str(exc)}],
            ),
        )
    _write_json(output_dir / "result.json", result.model_dump(mode="json"))
    return result


def _new_run(config: P0Config) -> RunInfo:
    config_digest = _config_digest(config)
    return RunInfo(
        run_id=f"run-pending-{config_digest[:16]}",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config_version=config.config_version,
        config_digest=config_digest,
        requested_backend=config.line_backend.name,
        dependency_versions=_dependency_versions(),
        runtime_environment=_runtime_environment(),
    )


def _dependency_versions() -> dict[str, str]:
    packages = ["numpy", "opencv-python-headless", "Pillow", "pydantic", "PyYAML"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _runtime_environment() -> dict[str, str]:
    """Record the CPU/Python environment that produced this measurement."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _resize_rgb(canonical_rgb: np.ndarray, analysis_size: tuple[int, int]) -> np.ndarray:
    width, height = analysis_size
    if canonical_rgb.shape[1] == width and canonical_rgb.shape[0] == height:
        return canonical_rgb.copy()
    return cv2.resize(canonical_rgb, (width, height), interpolation=cv2.INTER_AREA)


def _materialize_lines(
    raw_lines: list[RawLine],
    transform,
    min_length_px: float,
    min_quality: float,
    max_lines: int | None,
) -> list[LineRecord]:
    candidates: list[tuple[RawLine, float]] = []
    for raw in raw_lines:
        length = math.dist(raw.p1, raw.p2)
        if length >= min_length_px and raw.quality >= min_quality:
            candidates.append((raw, length))
    candidates.sort(
        key=lambda item: (
            -item[0].quality,
            -item[1],
            item[0].p1[1],
            item[0].p1[0],
            item[0].p2[1],
            item[0].p2[0],
        )
    )
    records: list[LineRecord] = []
    retained = candidates if max_lines is None else candidates[:max_lines]
    for index, (raw, length_analysis) in enumerate(retained, start=1):
        p1_analysis = Point(x=raw.p1[0], y=raw.p1[1])
        p2_analysis = Point(x=raw.p2[0], y=raw.p2[1])
        p1 = transform.analysis_to_canonical(p1_analysis)
        p2 = transform.analysis_to_canonical(p2_analysis)
        angle = math.atan2(p2_analysis.y - p1_analysis.y, p2_analysis.x - p1_analysis.x) % math.pi
        records.append(
            LineRecord(
                line_id=f"l{index:06d}",
                p1_analysis=p1_analysis,
                p2_analysis=p2_analysis,
                p1=p1,
                p2=p2,
                length_analysis=length_analysis,
                length=math.dist((p1.x, p1.y), (p2.x, p2.y)),
                angle_rad=angle,
                quality=raw.quality,
                backend_features=raw.backend_features,
                selected=True,
            )
        )
    return records


def _reindex_lines(lines: list[LineRecord]) -> list[LineRecord]:
    """Assign contiguous IDs after every line-selection stage.

    Curve-fragment suppression happens after the backend's stable ordering.
    Renumbering only at that point keeps the published ``lines.json`` contract
    simple: every retained line is named ``l000001`` through ``lNNNNNN`` with
    no hidden references to filtered-out records.
    """
    return [
        line.model_copy(update={"line_id": f"l{index:06d}"})
        for index, line in enumerate(lines, start=1)
    ]


def _seed_from_hash(input_hash: str, config_version: str) -> int:
    digest = hashlib.sha256(f"{input_hash}:{config_version}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _config_digest(config: P0Config) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _measurement_coverage(
    spatial_coverage: float,
    effective_line_support: float,
) -> float:
    """Report measurement coverage, never source credibility or an AI score."""
    return float(
        max(
            0.0,
            min(1.0, 0.60 * spatial_coverage + 0.40 * effective_line_support),
        )
    )


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)
