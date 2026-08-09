"""Create P0 review packets and source-blind behavior slices for PixArt/SDXL.

This script intentionally has no geometry-accuracy metric: the queue has no
human geometry labels yet.  Declared source data is used only after P0 runs to
report whether measurement availability or candidate volume drifts by slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2

from image_trust.pipeline import analyze_image
from image_trust.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"count": 0}
    available = [row for row in rows if row["p0_run_status"] == "ok"]
    candidate_counts = [int(row["candidate_count"]) for row in available]
    maximum_scores = [float(row["maximum_candidate_score"]) for row in available]
    return {
        "count": len(rows),
        "measurement_available_count": len(available),
        "measurement_available_rate": len(available) / len(rows),
        "candidate_positive_rate": (
            sum(count > 0 for count in candidate_counts) / len(available) if available else None
        ),
        "mean_candidate_count": sum(candidate_counts) / len(available) if available else None,
        "mean_maximum_candidate_score": sum(maximum_scores) / len(available) if available else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["development", "holdout"])
    parser.add_argument("--minimum-candidate-score", type=float, default=0.50)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_candidate_score <= 1.0:
        raise ValueError("minimum-candidate-score must be within [0, 1]")
    queue_path = args.benchmark_root / "cross_generator_review_registry.jsonl"
    queue = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    selected = [entry for entry in queue if entry["split"] in set(args.splits)]
    if not selected:
        raise ValueError("No review entries selected")
    if any(entry["geometry_annotation"]["status"] != "pending_blinded_human_review" for entry in selected):
        raise ValueError("This command accepts only pending blinded-review entries")
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(selected, start=1):
        image_path = args.project_root / str(entry["relative_path"])
        if _sha256(image_path) != entry["sha256"]:
            raise ValueError(f"Image hash mismatch: {image_path}")
        result = analyze_image(image_path, config, args.output_dir / "packets" / str(entry["sample_id"]))
        candidates = [
            candidate
            for candidate in result.evidence.features.get("anomalous_lines", [])
            if float(candidate["anomaly_candidate_score"]) >= args.minimum_candidate_score
        ]
        source_slice = dict(entry["declared_source_slice"])
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "split": entry["split"],
                "source_slice_for_posthoc_reporting_only": source_slice,
                "p0_run_status": result.evidence.run_status.value,
                "p0_applicability": result.evidence.applicability,
                "p0_coverage": result.evidence.coverage,
                "candidate_count": len(candidates),
                "maximum_candidate_score": max(
                    (float(candidate["anomaly_candidate_score"]) for candidate in candidates),
                    default=0.0,
                ),
                "review_packet": f"packets/{entry['sample_id']}",
            }
        )
        print(f"packet={index}/{len(selected)} sample={entry['sample_id']}", flush=True)
    by_archive: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        source_slice = dict(row["source_slice_for_posthoc_reporting_only"])
        by_archive[str(source_slice["archive_name"])].append(row)
        by_label[str(source_slice["label_name"])].append(row)
    report = {
        "schema_version": "p0-cross-generator-review-behavior-v1",
        "purpose": "Blind-review packet generation and post-hoc behavior slices; not geometry accuracy or AI-origin evaluation.",
        "review_registry_sha256": _sha256(queue_path),
        "p0_config_sha256": _sha256(args.config),
        "selected_splits": args.splits,
        "minimum_candidate_score": args.minimum_candidate_score,
        "overall": _summarize(rows),
        "by_archive": {name: _summarize(group) for name, group in sorted(by_archive.items())},
        "by_declared_source_label": {name: _summarize(group) for name, group in sorted(by_label.items())},
        "limitations": [
            "No human geometry labels are present; candidate rates are not precision, recall, or AI detection accuracy.",
            "Declared source slices are excluded from P0 execution and are post-hoc reporting only.",
            "Packets require two blinded human reviews before any natural-image geometry evaluation.",
        ],
    }
    (args.output_dir / "review_packets.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
