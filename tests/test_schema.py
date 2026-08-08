import json
from pathlib import Path

import pytest

from image_trust.pipeline import analyze_image
from image_trust.schemas import (
    AnalysisConfig,
    AnalysisResult,
    Direction,
    Evidence,
    Observation,
    RunInfo,
    RunStatus,
)
from image_trust.utils.config import load_config
from pydantic import ValidationError


def test_result_contract_separates_run_and_observation_states() -> None:
    result = AnalysisResult(
        run=RunInfo(
            run_id="run-test",
            created_at_utc="2026-08-08T00:00:00+00:00",
            config_version="test",
            requested_backend="opencv_lsd",
        ),
        evidence=Evidence(
            run_status=RunStatus.OK,
            observation=Observation.NOT_OBSERVED,
            direction=Direction.NEUTRAL,
            applicability=0.1,
            coverage=0.1,
            reliability=1.0,
        ),
    )
    dumped = result.model_dump(mode="json")
    assert dumped["evidence"]["run_status"] == "ok"
    assert dumped["evidence"]["observation"] == "not_observed"
    assert dumped["evidence"]["direction"] == "neutral"


def test_rejected_pipeline_output_still_validates_against_result_contract(tmp_path: Path) -> None:
    config = load_config(Path("configs/p0.yaml"))
    output_dir = tmp_path / "rejected"
    result = analyze_image(tmp_path / "missing.png", config, output_dir)
    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    validated = AnalysisResult.model_validate(payload)
    assert validated == result
    assert validated.evidence.run_status == RunStatus.REJECTED
    assert validated.evidence.observation == Observation.NOT_OBSERVED
    assert validated.evidence.direction == Direction.NEUTRAL


def test_analysis_long_side_matches_the_frozen_p0_range() -> None:
    assert AnalysisConfig(max_long_side=1024).max_long_side == 1024
    assert AnalysisConfig(max_long_side=1536).max_long_side == 1536
    with pytest.raises(ValidationError):
        AnalysisConfig(max_long_side=1023)
    with pytest.raises(ValidationError):
        AnalysisConfig(max_long_side=1537)
