"""Audit P0's automatic completion evidence without faking human visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from image_trust.schemas import AnalysisResult


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_f1_f5(summary_path: Path) -> list[str]:
    summary = _read_json(summary_path)
    errors: list[str] = []
    if summary.get("schema_version") != "p0-evaluation-summary-v1":
        errors.append("F1-F5 summary schema version is unexpected.")
    if int(summary.get("image_count", 0)) < 36:
        errors.append("F1-F5 evaluation contains fewer than the required 36 fixtures.")
    if int(summary.get("expectation_mismatch_count", 1)) != 0:
        errors.append("F1-F5 fixture expectations have mismatches.")
    return errors


def _audit_f6(source_manifest_path: Path, evaluation_dir: Path) -> list[str]:
    source_manifest = _read_json(source_manifest_path)
    source_root = source_manifest_path.parent
    fixtures = source_manifest.get("fixtures", [])
    errors: list[str] = []
    if len(fixtures) != 10:
        errors.append(f"F6 source manifest must contain 10 records, found {len(fixtures)}.")
    for fixture in fixtures:
        image_path = source_root / fixture["relative_path"]
        if not image_path.is_file():
            errors.append(f"F6 source file is missing: {image_path}")
        elif _sha256(image_path) != fixture.get("original_file_hash"):
            errors.append(f"F6 source hash mismatch: {image_path.name}")
        if not fixture.get("source_url") or not fixture.get("license"):
            errors.append(f"F6 provenance metadata is incomplete: {fixture.get('image_id')}")

    evaluation_manifest = evaluation_dir / "evaluation.jsonl"
    if not evaluation_manifest.is_file():
        return [*errors, "F6 evaluation.jsonl is missing."]
    records = [json.loads(line) for line in evaluation_manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 10:
        errors.append(f"F6 evaluation must contain 10 records, found {len(records)}.")
    for record in records:
        result = AnalysisResult.model_validate(record["result"])
        if result.evidence.direction.value != "neutral":
            errors.append(f"P0 direction is not neutral: {record['relative_path']}")
        if result.evidence.raw_score is not None:
            errors.append(f"P0 raw_score must be null: {record['relative_path']}")
        forbidden_keys = _forbidden_evidence_keys(record["result"])
        if forbidden_keys:
            errors.append(
                f"P0 result contains prohibited source-score fields ({', '.join(forbidden_keys)}): "
                f"{record['relative_path']}"
            )
        artifact_dir = evaluation_dir / record["artifact_dir"]
        for artifact in result.evidence.artifacts:
            artifact_path = artifact_dir / artifact
            if not artifact_path.is_file():
                errors.append(f"Missing F6 artifact: {artifact_path}")
        if result.input is not None:
            for overlay_name in ("lines_overlay.png", "anomalous_lines_overlay.png"):
                overlay_path = artifact_dir / overlay_name
                if overlay_path.is_file():
                    with Image.open(overlay_path) as overlay:
                        if overlay.size != result.input.canonical_size:
                            errors.append(
                                f"Overlay size mismatch for {record['relative_path']}: {overlay_name}"
                            )
    return errors


def _forbidden_evidence_keys(value: Any) -> list[str]:
    """Reject fields that would turn the uncalibrated P0 result into a verdict."""
    forbidden = {"t", "q", "ai_score", "ai_probability", "fake_probability"}
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key.lower() in forbidden:
                    found.add(key)
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(found)


def _audit_execution_contract(evaluation_dir: Path, dependency_lock_path: Path) -> list[str]:
    """Verify the traceability fields required by the P0 completion gate."""
    errors: list[str] = []
    if not dependency_lock_path.is_file():
        errors.append(f"Dependency lock is missing: {dependency_lock_path}")
    else:
        locked_versions = {
            line.split("==", 1)[0].strip().lower(): line.split("==", 1)[1].strip()
            for line in dependency_lock_path.read_text(encoding="utf-8").splitlines()
            if "==" in line and not line.lstrip().startswith("#")
        }
        for package in ("numpy", "opencv-python-headless", "pillow", "pydantic", "pyyaml"):
            if package not in locked_versions:
                errors.append(f"Dependency lock does not pin required package: {package}")

    evaluation_manifest = evaluation_dir / "evaluation.jsonl"
    if not evaluation_manifest.is_file():
        return [*errors, "F6 evaluation.jsonl is missing for execution-contract audit."]
    records = [json.loads(line) for line in evaluation_manifest.read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        result = AnalysisResult.model_validate(record["result"])
        label = record["relative_path"]
        if result.input is None or not result.input.sha256:
            errors.append(f"Input hash is missing: {label}")
            continue
        if not result.run.config_version or not result.run.config_digest:
            errors.append(f"Config version or digest is missing: {label}")
        if result.run.deterministic_seed is None:
            errors.append(f"Deterministic seed is missing: {label}")
        required_environment = {"python_version", "python_implementation", "platform", "machine"}
        missing_environment = sorted(
            key for key in required_environment if not result.run.runtime_environment.get(key)
        )
        if missing_environment:
            errors.append(
                f"Runtime environment is incomplete ({', '.join(missing_environment)}): {label}"
            )
        snapshot = result.evidence.features.get("config_snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("config_version") != result.run.config_version:
            errors.append(f"Config snapshot does not match run metadata: {label}")
        if result.run.requested_backend == "auto" and result.run.resolved_backend != "deeplsd":
            if not result.run.fallback_reason:
                errors.append(f"Auto-backend fallback reason is missing: {label}")
        version_keys = {
            "numpy": "numpy",
            "opencv-python-headless": "opencv-python-headless",
            "pillow": "Pillow",
            "pydantic": "pydantic",
            "pyyaml": "PyYAML",
        }
        for locked_name, runtime_name in version_keys.items():
            if dependency_lock_path.is_file() and result.run.dependency_versions.get(runtime_name) != locked_versions.get(locked_name):
                errors.append(
                    f"Runtime dependency version differs from lock ({runtime_name}): {label}"
                )
    return errors


def _human_review_errors(source_manifest_path: Path) -> list[str]:
    review_path = source_manifest_path.parent / "validation_log.json"
    if not review_path.is_file():
        return ["F6 validation_log.json is missing."]
    review_log = _read_json(review_path)
    reviews = review_log.get("reviews", [])
    required_ids = {source["image_id"] for source in _read_json(source_manifest_path).get("fixtures", [])}
    by_id = {
        review.get("image_id"): review
        for review in reviews
        if isinstance(review, dict) and review.get("image_id")
    }
    errors: list[str] = []
    for image_id in sorted(required_ids):
        review = by_id.get(image_id)
        if review is None:
            errors.append(f"Missing human review: {image_id}")
            continue
        for field in ("reviewer", "review_date", "screenshot", "findings"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                errors.append(f"Human review field '{field}' is empty: {image_id}")
        screenshot_value = review.get("screenshot")
        if isinstance(screenshot_value, str) and screenshot_value.strip():
            screenshot_paths = [
                item.strip() for item in screenshot_value.split(";") if item.strip()
            ]
            if not screenshot_paths:
                errors.append(f"Human review has no screenshot paths: {image_id}")
            project_root = source_manifest_path.parents[2]
            for screenshot in screenshot_paths:
                path = Path(screenshot)
                resolved = path if path.is_absolute() else project_root / path
                if not resolved.is_file():
                    errors.append(
                        f"Human review screenshot is missing: {image_id}: {screenshot}"
                    )
        if review.get("overlay_alignment") is not True:
            errors.append(f"Overlay alignment is not accepted: {image_id}")
        if review.get("family_mixing_blocker") is not False:
            errors.append(f"Family mixing remains a blocker: {image_id}")
        if review.get("anomaly_gate_correct") is not True:
            errors.append(f"Anomaly gate is not accepted: {image_id}")
        if review.get("disposition") != "accepted":
            errors.append(f"Human review disposition is not accepted: {image_id}")
    unexpected_ids = set(by_id) - required_ids
    if unexpected_ids:
        errors.append(f"Unexpected review IDs: {', '.join(sorted(unexpected_ids))}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-f5-summary", type=Path, required=True)
    parser.add_argument("--f6-source-manifest", type=Path, required=True)
    parser.add_argument("--f6-evaluation", type=Path, required=True)
    parser.add_argument(
        "--dependency-lock",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "requirements.lock",
        help="Pinned local environment used for the P0 baseline.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-human-review", action="store_true")
    args = parser.parse_args()

    automatic_errors = _audit_f1_f5(args.f1_f5_summary)
    automatic_errors.extend(_audit_f6(args.f6_source_manifest, args.f6_evaluation))
    automatic_errors.extend(_audit_execution_contract(args.f6_evaluation, args.dependency_lock))
    human_review_errors = _human_review_errors(args.f6_source_manifest)
    report = {
        "schema_version": "p0-completion-audit-v1",
        "automatic_checks_passed": not automatic_errors,
        "automatic_errors": automatic_errors,
        "human_review_passed": not human_review_errors,
        "human_review_errors": human_review_errors,
        "p0_completion_gate_passed": not automatic_errors and not human_review_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if automatic_errors or (args.require_human_review and human_review_errors):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
