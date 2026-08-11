from __future__ import annotations

import pytest

from image_trust.geometry_ai.replacement_gate import (
    GeometryGateSample,
    GeometryReplacementGateConfig,
    evaluate_geometry_replacement_gate,
    wilson_interval,
)


def _passing_rows() -> list[GeometryGateSample]:
    originals: list[GeometryGateSample] = []
    for scene in ("indoor", "outdoor"):
        for label in (0, 1):
            for index in range(15):
                base_id = f"{scene}-{label}-{index}"
                originals.append(
                    GeometryGateSample(
                        sample_id=base_id,
                        base_sample_id=base_id,
                        label=label,
                        split="unseen_generator",
                        scene_slice=scene,
                        transformation="original",
                        candidate_probability=0.9 if label else 0.1,
                        # Baseline predicts every sample as AI: balanced accuracy 0.5.
                        baseline_probability=0.7,
                    )
                )
    rows = list(originals)
    for transformation, delta in (("jpeg", -0.03), ("screenshot", 0.02), ("crop", -0.05)):
        for original in originals:
            rows.append(
                original.model_copy(
                    update={
                        "sample_id": f"{original.sample_id}-{transformation}",
                        "transformation": transformation,
                        "candidate_probability": original.candidate_probability + delta,
                    }
                )
            )
    return rows


def test_wilson_interval_matches_known_all_success_bounds() -> None:
    interval = wilson_interval(30, 30)

    assert interval.estimate == 1.0
    assert interval.lower == pytest.approx(0.8865, abs=0.001)
    assert interval.upper == pytest.approx(1.0)


def test_complete_strong_audit_passes_every_hard_condition() -> None:
    config = GeometryReplacementGateConfig(candidate_threshold=0.5)

    report = evaluate_geometry_replacement_gate(_passing_rows(), config)

    assert report.eligible is True
    assert all(criterion.status == "passed" for criterion in report.criteria)
    assert report.reasons == []
    by_id = {criterion.criterion_id: criterion for criterion in report.criteria}
    assert by_id["unseen_generator_ai_recall"].observed["lower"] >= 0.55
    assert by_id["real_false_positive_rate"].observed["upper"] <= 0.25
    assert by_id["scene_slice_improvement"].observed["positive_slice_count"] == 2


def test_missing_registered_evidence_fails_closed() -> None:
    report = evaluate_geometry_replacement_gate(
        [],
        GeometryReplacementGateConfig(candidate_threshold=0.5),
    )

    assert report.eligible is False
    assert {criterion.status for criterion in report.criteria} == {"missing"}
    assert len(report.reasons) == 4


def test_one_unstable_required_transformation_blocks_replacement() -> None:
    rows = _passing_rows()
    damaged = [
        row.model_copy(
            update={"candidate_probability": 1.0 - row.candidate_probability}
        )
        if row.transformation == "crop"
        else row
        for row in rows
    ]

    report = evaluate_geometry_replacement_gate(
        damaged,
        GeometryReplacementGateConfig(candidate_threshold=0.5),
    )

    assert report.eligible is False
    criterion = next(item for item in report.criteria if item.criterion_id == "transformation_stability")
    assert criterion.status == "failed"
    assert criterion.observed["transformations"]["crop"]["prediction_agreement"] == 0.0


def test_training_variants_cannot_substitute_for_missing_holdout_variants() -> None:
    rows = [row for row in _passing_rows() if row.transformation == "original"]
    rows.extend(
        row.model_copy(update={"split": "train"})
        for row in _passing_rows()
        if row.transformation != "original"
    )

    report = evaluate_geometry_replacement_gate(
        rows,
        GeometryReplacementGateConfig(candidate_threshold=0.5),
    )

    stability = next(item for item in report.criteria if item.criterion_id == "transformation_stability")
    assert report.eligible is False
    assert stability.status == "missing"


def test_duplicate_registration_rows_are_rejected_instead_of_inflating_confidence() -> None:
    rows = _passing_rows()

    with pytest.raises(ValueError, match="inflate the audit"):
        evaluate_geometry_replacement_gate(
            [*rows, rows[0]],
            GeometryReplacementGateConfig(candidate_threshold=0.5),
        )
