"""Run a fixed, CPU-bounded FSD smoke evaluation on held-out SDXL pairs.

This is an experiment harness for the upstream FSD repository, not a product
integration.  It deliberately uses a small fixed subset of 20 images: five
paired real/SDXL examples from each of the indoor and outdoor test folders.
The subset was never used to tune a Demirror model.

The FSD method is a pixel-forensic detector, not a geometric detector.  Its
score is reported separately from P0 so a failed external detector cannot
silently affect the website's AI-likelihood conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score


DEFAULT_IDS = (426, 427, 428, 429, 430)
DEFAULT_SCENES = ("Indoor", "Outdoor")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_paths(data_root: Path, scene: str, image_id: int) -> list[tuple[Path, int]]:
    root = data_root / f"Recent_SDXL_{scene}" / f"Recent_SDXL_{scene}" / "test"
    return [(root / "real" / f"{image_id}.jpg", 0), (root / "gen" / f"{image_id}.jpg", 1)]


def _summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    ai_scores = [-float(row["z_score"]) for row in rows]
    is_fake = [bool(row["z_score"] < threshold) for row in rows]
    return {
        "count": len(rows),
        "roc_auc_ai_negative_z": float(roc_auc_score(labels, ai_scores)),
        "default_threshold": threshold,
        "true_positive": sum(flag and label == 1 for flag, label in zip(is_fake, labels)),
        "false_positive": sum(flag and label == 0 for flag, label in zip(is_fake, labels)),
        "true_negative": sum(not flag and label == 0 for flag, label in zip(is_fake, labels)),
        "false_negative": sum(not flag and label == 1 for flag, label in zip(is_fake, labels)),
        "all_default_fake": sum(is_fake),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.fsd_source.is_dir():
        raise FileNotFoundError(f"FSD source directory does not exist: {args.fsd_source}")
    if not args.weights_dir.is_dir():
        raise FileNotFoundError(f"FSD weight directory does not exist: {args.weights_dir}")
    sys.path.insert(0, str(args.fsd_source.resolve()))
    from fsd import FSDDetector  # type: ignore[import-not-found]

    detector = FSDDetector.load(weights_dir=args.weights_dir, device="cpu")
    rows: list[dict[str, Any]] = []
    for scene in DEFAULT_SCENES:
        for image_id in DEFAULT_IDS:
            for path, label in _sample_paths(args.data_root, scene, image_id):
                if not path.is_file():
                    raise FileNotFoundError(f"Required fixed pilot image is missing: {path}")
                result = detector.score(str(path))
                rows.append(
                    {
                        "scene": scene.lower(),
                        "id": image_id,
                        "label": label,
                        "input_sha256": _sha256(path),
                        "z_score": float(result.z_score),
                    }
                )
                print(f"scored={len(rows)}/20 scene={scene.lower()} id={image_id} label={label}", flush=True)
    report = {
        "schema_version": "fsd-sdxl-pilot-v1",
        "purpose": "External detector screening only; not a product calibration or a geometric evaluation.",
        "upstream_source": str(args.fsd_source),
        "weights": {path.name: _sha256(path) for path in sorted(args.weights_dir.iterdir()) if path.is_file()},
        "evaluation": _summary(rows, float(detector.threshold)),
        "rows": rows,
        "limitations": [
            "This is a 20-image held-out SDXL pilot, not a cross-generator benchmark.",
            "FSD is an external pixel-forensic detector; it has no bearing on P0 geometric-candidate validity.",
            "A result below the project's preregistered AUC gate must not be integrated into the website.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/p3_aigc_v2/extracted"))
    parser.add_argument("--fsd-source", type=Path, default=Path("data/vendor/forensic-self-descriptions"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights/fsd-v1.2.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fsd_sdxl_pilot_v1"))
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report["evaluation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
