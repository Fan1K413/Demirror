"""Audit a frozen geometry-relationship model at a recall-oriented threshold.

The model itself is trained only on DeepFloyd/Kandinsky.  This script chooses
one threshold on the generator-isolated PixArt calibration split, then reports
the result on the already-defined SDXL holdout without using its labels to make
any decision.  It is deliberately serial by default: geometry extraction is
small, and this avoids accumulating workers on machines with limited memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from image_trust.geometry_ai.inference import assess_geometry_ai


@dataclass(frozen=True)
class Sample:
    archive: str
    identifier: int
    label: int
    path: str
    scene: str
    split: str


def discover(dataset_root: Path) -> list[Sample]:
    """Discover exactly the pre-registered calibration and holdout slices."""

    rules = (
        ("Pixart", range(351, 426), "calibration"),
        ("SDXL", range(426, 501), "test"),
    )
    rows: list[Sample] = []
    for generator, identifiers, split in rules:
        for path in sorted(dataset_root.glob(f"Recent_{generator}_*/Recent_{generator}_*/test/*/*.jpg")):
            label_name = path.parent.name
            if label_name not in {"real", "gen"} or not path.stem.isdecimal():
                continue
            identifier = int(path.stem)
            if identifier not in identifiers:
                continue
            archive = path.parents[2].name
            rows.append(
                Sample(
                    archive=archive,
                    identifier=identifier,
                    label=int(label_name == "gen"),
                    path=str(path.resolve()),
                    scene=archive.rsplit("_", 1)[-1].lower(),
                    split=split,
                )
            )
    rows.sort(key=lambda item: (item.split, item.scene, item.label, item.identifier))
    expected = {"calibration": 300, "test": 300}
    actual = {name: sum(item.split == name for item in rows) for name in expected}
    if actual != expected:
        raise ValueError(f"Expected fixed PixArt/SDXL slices {expected}, found {actual}")
    return rows


def threshold_at_fpr(probabilities: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    real_probabilities = probabilities[labels == 0]
    if not len(real_probabilities):
        raise ValueError("Calibration has no real-image probabilities")
    return float(np.quantile(real_probabilities, 1.0 - target_fpr, method="higher"))


def metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    real = labels == 0
    generated = labels == 1
    predicted = probabilities >= threshold
    return {
        "count": int(len(labels)),
        "real_count": int(real.sum()),
        "generated_count": int(generated.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "threshold": float(threshold),
        "true_positive_rate": float(predicted[generated].mean()),
        "false_positive_rate": float(predicted[real].mean()),
        "mean_probability_real": float(probabilities[real].mean()),
        "mean_probability_generated": float(probabilities[generated].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fpr", type=float, default=0.20)
    args = parser.parse_args()
    if not 0.0 < args.target_fpr < 1.0:
        raise ValueError("--target-fpr must be between 0 and 1")

    samples = discover(args.dataset_root)
    rows: list[dict[str, object]] = []
    for number, sample in enumerate(samples, start=1):
        result = assess_geometry_ai(Path(sample.path), model_path=args.model)
        row = {
            **asdict(sample),
            "status": result.status,
            "line_count": result.line_count,
            "probability": result.probability,
        }
        rows.append(row)
        print(f"geometry_relationship {number}/{len(samples)}", flush=True)

    available = [row for row in rows if row["status"] == "available" and row["probability"] is not None]
    by_split: dict[str, list[dict[str, object]]] = {
        name: [row for row in available if row["split"] == name] for name in ("calibration", "test")
    }
    if any(len(by_split[name]) != 300 for name in by_split):
        counts = {name: len(items) for name, items in by_split.items()}
        raise RuntimeError(f"Geometry model must score the complete fixed slices, got {counts}")

    calibration_probabilities = np.asarray([float(row["probability"]) for row in by_split["calibration"]])
    calibration_labels = np.asarray([int(row["label"]) for row in by_split["calibration"]])
    threshold = threshold_at_fpr(calibration_probabilities, calibration_labels, args.target_fpr)
    test_probabilities = np.asarray([float(row["probability"]) for row in by_split["test"]])
    test_labels = np.asarray([int(row["label"]) for row in by_split["test"]])

    report = {
        "schema_version": "geometry-relationship-threshold-audit-v1",
        "purpose": "Recall-oriented threshold audit; not a source or provenance verdict.",
        "model": str(args.model),
        "protocol": {
            "fit": "DeepFloyd + Kandinsky IDs 1-350 (frozen model)",
            "threshold_calibration": "PixArt IDs 351-425",
            "untouched_test": "SDXL IDs 426-500",
            "target_false_positive_rate": args.target_fpr,
        },
        "calibration": metrics(calibration_probabilities, calibration_labels, threshold),
        "held_out_test": metrics(test_probabilities, test_labels, threshold),
        "availability": {
            name: {
                "available": len(by_split[name]),
                "total": sum(row["split"] == name for row in rows),
            }
            for name in by_split
        },
        "limitations": [
            "Only line coordinates and their spatial/directional relationships are used.",
            "The reported rate is on a balanced research pool, not real-world prevalence.",
            "A product score requires a separate integration decision after this audit passes.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps(report["held_out_test"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
