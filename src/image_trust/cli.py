"""Command-line entry point for P0."""

from __future__ import annotations

import argparse
from pathlib import Path

from image_trust.camera.calibration import (
    load_camera_experiment_results,
    summarize_camera_calibration,
    write_camera_calibration_summary,
)
from image_trust.camera.config import load_camera_config
from image_trust.camera.contracts import CameraEstimateStatus
from image_trust.camera.dataset import (
    CalibrationDatasetSplit,
    audit_calibration_registry,
    load_calibration_registry,
    run_camera_calibration_dataset,
)
from image_trust.camera.pipeline import analyze_camera_image
from image_trust.pipeline import analyze_image
from image_trust.provenance.c2pa import inspect_c2pa_asset, write_c2pa_record
from image_trust.provenance.config import load_c2pa_config
from image_trust.schemas import RunStatus
from image_trust.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-trust",
        description="P0 projection-geometry measurement baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Analyze one local image.")
    analyze.add_argument("input", type=Path, help="PNG, JPEG, or static WebP input.")
    analyze.add_argument(
        "--config",
        type=Path,
        required=True,
        help="P0 YAML configuration.",
    )
    analyze.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Artifact directory to create or update.",
    )
    camera = subparsers.add_parser(
        "camera-analyze",
        help="Run the P1 global--local camera-consistency experiment.",
    )
    camera.add_argument("input", type=Path, help="Local static image input.")
    camera.add_argument(
        "--config",
        type=Path,
        required=True,
        help="P1 camera YAML configuration.",
    )
    camera.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Artifact directory to create or update.",
    )
    calibration = subparsers.add_parser(
        "camera-calibration-summary",
        help="Summarize a homogeneous P1 calibration cohort without fitting thresholds.",
    )
    calibration.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="camera_result.json files from one backend and one P1 configuration.",
    )
    calibration.add_argument(
        "--config",
        type=Path,
        required=True,
        help="The exact P1 camera YAML used for every result.",
    )
    calibration.add_argument(
        "--cohort",
        required=True,
        help="Local descriptive name for this independent calibration cohort.",
    )
    calibration.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the descriptive calibration summary.",
    )
    c2pa = subparsers.add_parser(
        "c2pa-analyze",
        help="Inspect embedded C2PA data in one local asset without network access.",
    )
    c2pa.add_argument("input", type=Path, help="Local static image input.")
    c2pa.add_argument(
        "--config",
        type=Path,
        required=True,
        help="P1 offline C2PA YAML configuration.",
    )
    c2pa.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Artifact directory to create or update.",
    )
    registry_audit = subparsers.add_parser(
        "camera-calibration-registry-audit",
        help="Validate local P1 calibration registry files, hashes, dimensions, and family splits.",
    )
    registry_audit.add_argument("--registry", type=Path, required=True)
    registry_audit.add_argument("--dataset-root", type=Path, required=True)
    calibration_run = subparsers.add_parser(
        "camera-calibration-run",
        help="Run one registered P1 camera cohort and write descriptive calibration artifacts.",
    )
    calibration_run.add_argument("--registry", type=Path, required=True)
    calibration_run.add_argument("--dataset-root", type=Path, required=True)
    calibration_run.add_argument("--config", type=Path, required=True)
    calibration_run.add_argument(
        "--split",
        choices=[split.value for split in CalibrationDatasetSplit],
        default=CalibrationDatasetSplit.CALIBRATION.value,
    )
    calibration_run.add_argument(
        "--allow-control-smoke",
        action="store_true",
        help="Required to run a control_smoke registry's control split.",
    )
    calibration_run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        config = load_config(args.config)
        result = analyze_image(args.input, config, args.output)
        print(
            f"run_status={result.evidence.run_status.value} "
            f"observation={result.evidence.observation.value} "
            f"output={args.output}"
        )
        return (
            0
            if result.evidence.run_status in {RunStatus.OK, RunStatus.NOT_APPLICABLE}
            else 2
        )
    if args.command == "camera-analyze":
        config = load_camera_config(args.config)
        result = analyze_camera_image(args.input, config, args.output)
        print(
            f"camera_status={result.full_image.status.value} "
            f"e_cam_observation={result.e_cam.observation.value} "
            f"output={args.output}"
        )
        return 0 if result.full_image.status is CameraEstimateStatus.OK else 2
    if args.command == "camera-calibration-summary":
        config = load_camera_config(args.config)
        results = load_camera_experiment_results(args.results)
        summary = summarize_camera_calibration(
            results,
            config,
            args.cohort,
            result_filenames=[str(path) for path in args.results],
        )
        write_camera_calibration_summary(args.output, summary)
        measured_count = summary.e_cam_observation_counts.get("measured", 0)
        print(
            f"result_count={summary.result_count} "
            f"e_cam_measured={measured_count} "
            f"output={args.output}"
        )
        return 0
    if args.command == "c2pa-analyze":
        config = load_c2pa_config(args.config)
        record = inspect_c2pa_asset(args.input, config)
        args.output.mkdir(parents=True, exist_ok=True)
        write_c2pa_record(args.output / "c2pa_result.json", record)
        print(
            f"c2pa_status={record.status.value} "
            f"manifest_present={record.manifest_present} "
            f"output={args.output}"
        )
        return 0 if record.status.value in {"present", "not_observed"} else 2
    if args.command == "camera-calibration-registry-audit":
        registry = load_calibration_registry(args.registry)
        audit = audit_calibration_registry(registry, args.dataset_root)
        print(
            f"registry_valid={audit.valid} "
            f"entry_count={audit.entry_count} "
            f"errors={len(audit.errors)}"
        )
        if audit.errors:
            for error in audit.errors:
                print(f"registry_error={error}")
        return 0 if audit.valid else 2
    if args.command == "camera-calibration-run":
        registry = load_calibration_registry(args.registry)
        config = load_camera_config(args.config)
        run = run_camera_calibration_dataset(
            registry,
            args.dataset_root,
            config,
            args.output,
            split=CalibrationDatasetSplit(args.split),
            allow_control_smoke=args.allow_control_smoke,
        )
        print(
            f"calibration_cohort={run.cohort_name} "
            f"split={run.split.value} "
            f"image_count={len(run.image_ids)} "
            f"output={args.output}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
