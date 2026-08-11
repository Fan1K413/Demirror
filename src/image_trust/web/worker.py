"""Killable child-process entry point for one local web analysis job."""

from __future__ import annotations

import argparse
from pathlib import Path

from image_trust.web.jobs import WebJobOutcome, _write_json, build_local_runner
from image_trust.watermark.remote import load_remote_verification_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--upload-path", required=True, type=Path)
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    job_dir = args.job_dir.resolve()
    progress_path = job_dir / "worker_progress.json"
    outcome_path = job_dir / "worker_outcome.json"

    def report_progress(stage: str, progress_percent: int) -> None:
        _write_json(progress_path, {"stage": stage, "progress_percent": progress_percent})

    try:
        runner = build_local_runner(project_root, load_remote_verification_settings(project_root))
        outcome = runner(args.upload_path, job_dir, report_progress)
    except Exception as error:
        outcome = WebJobOutcome(
            status="failed",
            limitations=["web_worker_failed"],
            errors=[{"code": type(error).__name__, "message": str(error)}],
        )
    _write_json(outcome_path, outcome.model_dump(mode="json"))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
