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

from image_trust.geometry.overlays import FAMILY_COLORS
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


def _reviewed_families(features: dict[str, object]) -> list[dict[str, object]]:
    """Return the exact family order and colours used by the review overlay."""

    global_families = [
        dict(family)
        for family in features.get("families", [])
        if isinstance(family, dict) and bool(family.get("stable"))
    ]
    local_families = [
        dict(family)
        for family in features.get("local_families", [])
        if isinstance(family, dict) and bool(family.get("stable"))
    ]
    selected = [*global_families, *local_families]
    if not selected:
        selected = [
            dict(family)
            for family in features.get("parallel_families", [])
            if isinstance(family, dict) and bool(family.get("stable"))
        ]
    return selected


def _family_review_template(reviewer_id: str, features: dict[str, object]) -> dict[str, object]:
    """Create a source-blind, colour-linked template for family-purity review."""

    families = _reviewed_families(features)
    entries: list[dict[str, object]] = []
    for index, family in enumerate(families):
        red, green, blue = FAMILY_COLORS[index % len(FAMILY_COLORS)]
        members = family.get("member_line_ids")
        entries.append(
            {
                "family_id": family.get("family_id"),
                "overlay_color_rgb": [red, green, blue],
                "scope": family.get("scope"),
                "vp_type": family.get("vp_type"),
                "direction_analysis_radians": family.get("direction_analysis"),
                "spatial_window_analysis": family.get("spatial_window_analysis"),
                "member_line_count": len(members) if isinstance(members, list) else 0,
                "member_line_ids": members if isinstance(members, list) else [],
                "review_label": "pending",
                "review_label_options": [
                    "coherent_single_structure",
                    "overmerged_unrelated_structures",
                    "unassessable",
                ],
                "review_note": "",
            }
        )
    return {
        "schema_version": "p0-family-purity-review-template-v1",
        "purpose": "Blind human review of whether each coloured P0 family represents one visible structural relation; not AI-origin annotation.",
        "reviewer_id": reviewer_id,
        "source_label_visibility": "forbidden",
        "review_instructions": [
            "Inspect the original image and anomalous-lines overlay together.",
            "Mark coherent_single_structure only when the coloured members plausibly belong to one repeated edge, plane, or parallel structure.",
            "Mark overmerged_unrelated_structures when a colour joins visibly unrelated objects or planes merely because their image directions are similar.",
            "Use unassessable when occlusion, curvature, reflection, insufficient resolution, or missing straight-line support prevents a decision.",
            "Do not infer or record whether the image is real or AI-generated.",
        ],
        "families": entries,
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
    posthoc_rows: list[dict[str, object]] = []
    blind_rows: list[dict[str, object]] = []
    for index, entry in enumerate(selected, start=1):
        reviewer_id = f"p0r{index:04d}"
        image_path = args.project_root / str(entry["relative_path"])
        if _sha256(image_path) != entry["sha256"]:
            raise ValueError(f"Image hash mismatch: {image_path}")
        packet_dir = args.output_dir / "blind_packets" / "packets" / reviewer_id
        result = analyze_image(image_path, config, packet_dir)
        candidates = [
            candidate
            for candidate in result.evidence.features.get("anomalous_lines", [])
            if float(candidate["anomaly_candidate_score"]) >= args.minimum_candidate_score
        ]
        template = _family_review_template(
            reviewer_id,
            dict(result.evidence.features),
        )
        (packet_dir / "family_review_template.json").write_text(
            json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_slice = dict(entry["declared_source_slice"])
        blind_rows.append(
            {
                "reviewer_id": reviewer_id,
                "review_packet": f"packets/{reviewer_id}",
                "family_review_template": f"packets/{reviewer_id}/family_review_template.json",
            }
        )
        posthoc_rows.append(
            {
                "reviewer_id": reviewer_id,
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
                "review_packet": f"blind_packets/packets/{reviewer_id}",
                "family_review_template": f"blind_packets/packets/{reviewer_id}/family_review_template.json",
            }
        )
        print(f"packet={index}/{len(selected)} reviewer_id={reviewer_id}", flush=True)
    by_archive: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in posthoc_rows:
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
        "overall": _summarize(posthoc_rows),
        "by_archive": {name: _summarize(group) for name, group in sorted(by_archive.items())},
        "by_declared_source_label": {name: _summarize(group) for name, group in sorted(by_label.items())},
        "limitations": [
            "No human geometry labels are present; candidate rates are not precision, recall, or AI detection accuracy.",
            "Declared source slices are excluded from P0 execution and are post-hoc reporting only.",
            "Packets require two blinded human reviews before any natural-image geometry evaluation.",
        ],
    }
    blind_dir = args.output_dir / "blind_packets"
    blind_dir.mkdir(parents=True, exist_ok=True)
    (blind_dir / "review_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_rows), encoding="utf-8"
    )
    posthoc_dir = args.output_dir / "posthoc"
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    (posthoc_dir / "review_key.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in posthoc_rows), encoding="utf-8"
    )
    (posthoc_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
