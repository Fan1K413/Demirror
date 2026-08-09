from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load("p0_natural_mutation_builder", "build_p0_natural_mutation_benchmark.py")
evaluator = _load("p0_natural_mutation_evaluator", "evaluate_p0_natural_mutation_benchmark.py")


def test_choose_mutation_preserves_a_detectable_local_contrast_profile() -> None:
    image = np.full((120, 120, 3), 210, dtype=np.uint8)
    image[:, 55:58] = (30, 55, 85)
    lines = [
        {
            "line_id": "l1",
            "length_analysis": 90.0,
            "p1_analysis": {"x": 56.0, "y": 15.0},
            "p2_analysis": {"x": 56.0, "y": 105.0},
        }
    ]
    families = [{"family_id": "parallel001", "stable": True, "direction_analysis": 1.57, "member_line_ids": ["l1"]}]

    mutation = builder.choose_mutation(
        image,
        lines,
        families,
        rotation_deg=22.0,
        offset_px=13.0,
        minimum_contrast=40.0,
    )

    assert mutation is not None
    assert mutation.source_line_id == "l1"
    assert mutation.core_bgr != mutation.side_bgr
    changed = builder.apply_mutation(image, mutation)
    assert not np.array_equal(changed, image)


def test_segment_match_requires_direction_and_substantial_overlap() -> None:
    target = {"p1": [10.0, 20.0], "p2": [70.0, 20.0], "tolerance_px": 4.0}
    matching = {"p1_analysis": {"x": 28.0, "y": 21.0}, "p2_analysis": {"x": 65.0, "y": 21.0}}
    crossing = {"p1_analysis": {"x": 40.0, "y": 2.0}, "p2_analysis": {"x": 40.0, "y": 38.0}}
    too_short = {"p1_analysis": {"x": 38.0, "y": 20.0}, "p2_analysis": {"x": 50.0, "y": 20.0}}

    assert evaluator._segment_match(matching, target)
    assert not evaluator._segment_match(crossing, target)
    assert not evaluator._segment_match(too_short, target)


def test_bounded_source_paths_uses_numeric_ids_not_lexical_sort(tmp_path: Path) -> None:
    for filename in ("1.jpg", "10.jpg", "100.jpg", "351.jpg", "352.jpg", "425.jpg", "426.jpg"):
        (tmp_path / filename).write_bytes(b"fixture")

    paths = builder.bounded_source_paths(tmp_path, minimum_id=351, maximum_id=425, limit=10)

    assert [path.name for path in paths] == ["351.jpg", "352.jpg", "425.jpg"]


def test_baseline_eligibility_rejects_not_applicable_measurements() -> None:
    class Evidence:
        def __init__(self, status: str, applicability: float) -> None:
            self.run_status = type("Status", (), {"value": status})()
            self.applicability = applicability

    usable = type("Result", (), {"evidence": Evidence("ok", 0.75)})()
    rejected = type("Result", (), {"evidence": Evidence("not_applicable", 0.2)})()

    assert builder.baseline_is_eligible(usable, 0.45)
    assert not builder.baseline_is_eligible(rejected, 0.45)


def test_evaluator_records_a_safe_backend_failure_without_aborting(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "sample.jpg"
    input_path.write_bytes(b"fixture")
    evidence = SimpleNamespace(
        run_status=SimpleNamespace(value="failed"),
        applicability=0.0,
        coverage=0.0,
        features={},
    )
    monkeypatch.setattr(evaluator, "analyze_image", lambda *_args, **_kwargs: SimpleNamespace(evidence=evidence))

    row = evaluator._one_run(
        input_path,
        {"p1": [0.0, 0.0], "p2": [1.0, 1.0], "tolerance_px": 1.0},
        object(),
        tmp_path / "artifacts",
        0.5,
    )

    assert row["run_status"] == "failed"
    assert row["candidate_count"] == 0
    assert row["target_detected"] is False
    assert row["measurement_error"] == "lines_artifact_unavailable"


def test_checkpoint_round_trip_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    checkpoint = tmp_path / "records.jsonl"
    rows = [{"sample_id": "sample-a"}, {"sample_id": "sample-b"}]

    evaluator._write_records(checkpoint, rows)

    assert evaluator._load_records(checkpoint) == {"sample-a": rows[0], "sample-b": rows[1]}
    checkpoint.write_text('{"sample_id": "sample-a"}\n{"sample_id": "sample-a"}\n', encoding="utf-8")
    try:
        evaluator._load_records(checkpoint)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("Duplicate checkpoint IDs must be rejected")
