import importlib.util
import json
from pathlib import Path

from image_trust.schemas import (
    AnalysisResult,
    CoordinateTransformSummary,
    Direction,
    Evidence,
    InputSummary,
    Observation,
    RunInfo,
    RunStatus,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "p0_completion_audit",
        REPOSITORY_ROOT / "scripts" / "audit_p0_validation.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(runtime_environment: dict[str, str]) -> AnalysisResult:
    transform = CoordinateTransformSummary(
        encoded_size=(100, 100),
        canonical_size=(100, 100),
        analysis_size=(100, 100),
        scale_x=1.0,
        scale_y=1.0,
    )
    return AnalysisResult(
        run=RunInfo(
            run_id="run-test",
            created_at_utc="2026-08-08T00:00:00+00:00",
            config_version="p0-test",
            config_digest="config-digest",
            deterministic_seed=123,
            requested_backend="opencv_lsd",
            resolved_backend="opencv_lsd",
            dependency_versions={
                "numpy": "2.2.6",
                "opencv-python-headless": "4.14.0.94",
                "Pillow": "12.3.0",
                "pydantic": "2.13.4",
                "PyYAML": "6.0.3",
            },
            runtime_environment=runtime_environment,
        ),
        input=InputSummary(
            sha256="a" * 64,
            detected_format="png",
            original_filename="input.png",
            filename_format_mismatch=False,
            file_size_bytes=100,
            encoded_size=(100, 100),
            canonical_size=(100, 100),
            analysis_size=(100, 100),
            pixel_count=10_000,
            color_mode="RGB",
            coordinate_transform=transform,
        ),
        evidence=Evidence(
            run_status=RunStatus.OK,
            observation=Observation.NEGATIVE,
            direction=Direction.NEUTRAL,
            raw_score=None,
            applicability=1.0,
            coverage=1.0,
            reliability=1.0,
            features={"config_snapshot": {"config_version": "p0-test"}},
        ),
    )


def _write_lock(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "numpy==2.2.6",
                "opencv-python-headless==4.14.0.94",
                "pillow==12.3.0",
                "pydantic==2.13.4",
                "PyYAML==6.0.3",
            ]
        ),
        encoding="utf-8",
    )


def test_execution_contract_requires_complete_runtime_and_matching_lock(tmp_path: Path) -> None:
    audit = _load_audit_module()
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    lock_path = tmp_path / "requirements.lock"
    _write_lock(lock_path)
    good_environment = {
        "python_version": "3.10.6",
        "python_implementation": "CPython",
        "platform": "Windows",
        "machine": "AMD64",
    }
    record = {
        "relative_path": "input.png",
        "result": _result(good_environment).model_dump(mode="json"),
    }
    (evaluation_dir / "evaluation.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    assert audit._audit_execution_contract(evaluation_dir, lock_path) == []

    record["result"]["run"]["runtime_environment"].pop("machine")
    (evaluation_dir / "evaluation.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    errors = audit._audit_execution_contract(evaluation_dir, lock_path)
    assert any("Runtime environment is incomplete (machine)" in error for error in errors)


def test_human_review_needs_real_screenshot_paths_and_p0_prohibits_scores(tmp_path: Path) -> None:
    audit = _load_audit_module()
    source_root = tmp_path / "project" / "data" / "f6"
    source_root.mkdir(parents=True)
    source_manifest = source_root / "source_manifest.json"
    source_manifest.write_text(
        json.dumps({"fixtures": [{"image_id": "f6_01"}]}), encoding="utf-8"
    )
    review_path = source_root / "validation_log.json"
    review = {
        "image_id": "f6_01",
        "reviewer": "reviewer",
        "review_date": "2026-08-08",
        "overlay_alignment": True,
        "family_mixing_blocker": False,
        "anomaly_gate_correct": True,
        "screenshot": "outputs/missing.png",
        "findings": "No coordinate offset or blocking family mixing observed.",
        "disposition": "accepted",
    }
    review_path.write_text(json.dumps({"reviews": [review]}), encoding="utf-8")
    errors = audit._human_review_errors(source_manifest)
    assert any("Human review screenshot is missing" in error for error in errors)

    screenshot = tmp_path / "project" / "outputs" / "review.png"
    screenshot.parent.mkdir()
    screenshot.write_bytes(b"review-evidence")
    review["screenshot"] = "outputs/review.png"
    review_path.write_text(json.dumps({"reviews": [review]}), encoding="utf-8")
    assert audit._human_review_errors(source_manifest) == []
    assert audit._forbidden_evidence_keys({"evidence": {"ai_score": 0.2}}) == ["ai_score"]
