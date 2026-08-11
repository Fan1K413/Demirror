"""Evaluate geometry-v2 measurements on exact controlled fixture labels.

This script measures sensitivity to known line-family violations and false
alarms on their paired clean controls.  It does not train or authorize an AI
source classifier.  The output is resumable and deliberately keeps G1--G5
separate so later calibration cannot hide a weak check behind a combined score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from image_trust.geometry_ai.measurement_v2 import assess_geometry_measurement_v2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_registry(path: Path, splits: set[str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if str(row["split"]) in splits]
    if not selected:
        raise ValueError("No fixture records selected")
    return selected


def _read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        str(row["sample_id"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def _summarize(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positive = [row for row in rows if row["anomaly_present"]]
    negative = [row for row in rows if not row["anomaly_present"]]
    predicted = lambda row: float(row["maximum_anomaly_score"]) >= threshold
    true_positive = sum(predicted(row) for row in positive)
    false_positive = sum(predicted(row) for row in negative)
    return {
        "count": len(rows),
        "anomaly_count": len(positive),
        "clean_count": len(negative),
        "measurement_available_rate": (
            sum(row["status"] == "measurable" for row in rows) / len(rows) if rows else None
        ),
        "sensitivity": true_positive / len(positive) if positive else None,
        "false_alarm_rate": false_positive / len(negative) if negative else None,
        "true_positive": true_positive,
        "false_positive": false_positive,
    }


def _evaluate_row(root: Path, row: dict[str, Any], artifacts_root: Path) -> dict[str, Any]:
    image_path = root / str(row["relative_path"])
    if _sha256(image_path) != row["sha256"]:
        raise ValueError(f"Fixture hash mismatch: {image_path}")
    sample_id = str(row["sample_id"])
    result = assess_geometry_measurement_v2(
        image_path,
        output_dir=artifacts_root / sample_id,
    )
    checks = {
        check.check_id: {
            "status": check.status,
            "anomaly_score": check.anomaly_score,
            "finding_count": len(check.findings),
        }
        for check in result.checks
    }
    available_scores = [
        float(check["anomaly_score"])
        for check in checks.values()
        if check["status"] == "available" and check["anomaly_score"] is not None
    ]
    return {
        "sample_id": sample_id,
        "split": row["split"],
        "fixture_family": row["fixture_family"],
        "anomaly_present": bool(row["geometry_annotation"]["anomaly_present"]),
        "status": result.status,
        "applicability": result.applicability,
        "region_count": len(result.regions),
        "stable_family_count": sum(family.stable for family in result.families),
        "maximum_anomaly_score": max(available_scores, default=0.0),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["development", "holdout"])
    parser.add_argument("--fixture-family", action="append")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")

    registry_path = args.benchmark_root / "fixture_registry.jsonl"
    records = _read_registry(registry_path, set(args.splits))
    if args.fixture_family:
        selected_families = set(args.fixture_family)
        records = [
            row for row in records if str(row["fixture_family"]) in selected_families
        ]
        if not records:
            raise ValueError("No records match --fixture-family")
    if args.limit is not None:
        records = records[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "evaluated_samples.jsonl"
    completed = _read_completed(samples_path)
    evaluated = dict(completed)
    cv2.setNumThreads(1)

    for index, row in enumerate(records, start=1):
        sample_id = str(row["sample_id"])
        if sample_id not in completed:
            evaluated[sample_id] = _evaluate_row(
                args.benchmark_root,
                row,
                args.output_dir / "artifacts",
            )
            samples_path.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                    for item in evaluated.values()
                ),
                encoding="utf-8",
            )
        print(f"evaluated={index}/{len(records)} sample={sample_id}", flush=True)

    selected_ids = {str(row["sample_id"]) for row in records}
    selected = [row for sample_id, row in evaluated.items() if sample_id in selected_ids]
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({str(row["fixture_family"]) for row in selected}):
        by_family[family] = _summarize(
            [row for row in selected if row["fixture_family"] == family],
            args.threshold,
        )
    by_check: dict[str, dict[str, Any]] = {}
    for check_id in ("G1", "G2", "G3", "G4", "G5"):
        statuses: dict[str, int] = defaultdict(int)
        positive_scores: list[float] = []
        negative_scores: list[float] = []
        for row in selected:
            check = row["checks"].get(check_id, {"status": "not_run", "anomaly_score": None})
            statuses[str(check["status"])] += 1
            score = check["anomaly_score"]
            if score is not None:
                (positive_scores if row["anomaly_present"] else negative_scores).append(float(score))
        by_check[check_id] = {
            "statuses": dict(sorted(statuses.items())),
            "anomaly_mean": sum(positive_scores) / len(positive_scores) if positive_scores else None,
            "clean_mean": sum(negative_scores) / len(negative_scores) if negative_scores else None,
        }

    report = {
        "schema_version": "geometry-measurement-v2-controlled-evaluation-v1",
        "fixture_registry_sha256": _sha256(registry_path),
        "selected_splits": args.splits,
        "candidate_threshold": args.threshold,
        "overall": _summarize(selected, args.threshold),
        "by_fixture_family": by_family,
        "by_check": by_check,
        "origin_scoring_authorized": False,
        "limitations": [
            "Controlled fixtures test known line-family violations, not natural-image prevalence.",
            "No AI or camera source label is consumed by the measurement pipeline.",
            "A separate source-isolated holdout gate is required before replacing P0 scoring.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["overall"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
