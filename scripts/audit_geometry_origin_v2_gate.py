"""Audit a frozen geometry-v2 score file and optionally promote its model.

The audit is fail-closed.  A promoted model is written only when all four hard
conditions pass.  Missing unseen-generator, baseline, scene or transformation
data produces ``eligible: false`` and no promoted model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_trust.geometry_ai.origin_v2 import GeometryOriginV2Model
from image_trust.geometry_ai.replacement_gate import (
    GeometryGateSample,
    GeometryReplacementGateConfig,
    evaluate_geometry_replacement_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--promoted-model-output", type=Path)
    parser.add_argument("--baseline-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-transform-pairs", type=int, default=20)
    args = parser.parse_args()
    if args.promoted_model_output and args.promoted_model_output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing promoted model: {args.promoted_model_output}"
        )

    model = GeometryOriginV2Model.model_validate_json(
        args.candidate_model.read_text(encoding="utf-8")
    )
    if model.deployment_eligible:
        raise ValueError("--candidate-model must be an ineligible frozen candidate")
    rows = [
        GeometryGateSample.model_validate_json(line)
        for line in args.scores.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = GeometryReplacementGateConfig(
        candidate_threshold=model.decision_threshold,
        baseline_threshold=args.baseline_threshold,
        minimum_pairs_per_transformation=args.minimum_transform_pairs,
    )
    report = evaluate_geometry_replacement_gate(rows, config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    promoted = None
    if report.eligible and args.promoted_model_output:
        promoted_model = model.model_copy(
            update={
                "deployment_eligible": True,
                "replacement_gate": report.model_dump(mode="json"),
            }
        )
        # Re-validate after model_copy because Pydantic deliberately skips validation there.
        promoted_model = GeometryOriginV2Model.model_validate(promoted_model.model_dump())
        args.promoted_model_output.parent.mkdir(parents=True, exist_ok=True)
        args.promoted_model_output.write_text(
            promoted_model.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        promoted = str(args.promoted_model_output)
    print(
        json.dumps(
            {
                "eligible": report.eligible,
                "report": str(args.report),
                "promoted_model": promoted,
                "reasons": report.reasons,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
