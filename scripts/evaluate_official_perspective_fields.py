"""Evaluate the authors' official PerspectiveFields geometry checkpoint.

This is intentionally a fixed-checkpoint evaluation.  The model weights were
released with the Projective Geometry repository; no samples from the target
SDXL set are used to train, calibrate, choose a threshold, or alter the model.
Only the derived latitude/gravity fields are passed to the classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


@dataclass(frozen=True)
class Sample:
    archive: str
    identifier: int
    label: int
    path: str
    scene: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(root: Path) -> list[Sample]:
    rows: list[Sample] = []
    for path in sorted(root.glob("Recent_SDXL_*/Recent_SDXL_*/test/*/*.jpg")):
        label_name = path.parent.name
        if label_name not in {"real", "gen"} or not path.stem.isdecimal():
            continue
        # IDs 426--500 are the frozen held-out SDXL slice.  Earlier IDs belong
        # to the local train/calibration protocol and must never enter this run.
        identifier = int(path.stem)
        if not 426 <= identifier <= 500:
            continue
        archive = path.parents[2].name
        scene = archive.rsplit("_", 1)[-1].lower()
        rows.append(
            Sample(
                archive=archive,
                identifier=identifier,
                label=int(label_name == "gen"),
                path=str(path.resolve()),
                scene=scene,
                sha256=_sha256(path),
            )
        )
    rows.sort(key=lambda row: (row.scene, row.label, row.identifier))
    if len(rows) != 300 or {row.label for row in rows} != {0, 1}:
        raise ValueError(f"Expected the 300-image SDXL evaluation slice, found {len(rows)}")
    return rows


def _load_models(field_weights: Path, classifier_weights: Path, vendor_root: Path, threads: int):
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)
    import sys

    sys.path.insert(0, str((vendor_root / "perspective_fields").resolve()))
    from fields_model import FieldsClassifier
    from perspective2d.perspectivefields import PerspectiveFields

    original_loader = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = lambda *_args, **_kwargs: torch.load(
        field_weights, map_location="cpu"
    )
    try:
        field_model = PerspectiveFields("Paramnet-360Cities-edina-uncentered").eval()
    finally:
        torch.hub.load_state_dict_from_url = original_loader
    classifier = FieldsClassifier().eval()
    classifier.load_state_dict(torch.load(classifier_weights, map_location="cpu", weights_only=True))
    return field_model, classifier


def _score_worker(
    samples: list[Sample],
    field_weights: str,
    classifier_weights: str,
    vendor_root: str,
    output: str,
    threads: int,
) -> None:
    field_model, classifier = _load_models(
        Path(field_weights), Path(classifier_weights), Path(vendor_root), threads
    )
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for sample in samples:
            image = cv2.imread(sample.path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unable to decode {sample.path}")
            prediction = field_model.inference(img_bgr=image)
            field = torch.cat(
                (
                    prediction["pred_latitude_original"].unsqueeze(0) / 90.0,
                    prediction["pred_gravity_original"],
                ),
                dim=0,
            ).unsqueeze(0)
            probability = float(torch.softmax(classifier(field), dim=1)[0, 1])
            rows.append({**asdict(sample), "generated_probability": probability})
    Path(output).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _valid_shard(path: Path, samples: list[Sample]) -> bool:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return [
            {key: value for key, value in row.items() if key != "generated_probability"} for row in rows
        ] == [asdict(sample) for sample in samples] and all(
            0.0 <= float(row["generated_probability"]) <= 1.0 for row in rows
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows])
    probabilities = np.asarray([float(row["generated_probability"]) for row in rows])
    predicted = probabilities >= 0.5  # official fixed binary head decision; no target-set tuning
    real = labels == 0
    generated = labels == 1
    return {
        "accuracy_at_official_0_5": float(accuracy_score(labels, predicted)),
        "balanced_accuracy_at_official_0_5": float(balanced_accuracy_score(labels, predicted)),
        "false_positive_rate_at_official_0_5": float(predicted[real].mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "true_positive_rate_at_official_0_5": float(predicted[generated].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--field-weights", type=Path, required=True)
    parser.add_argument("--classifier-weights", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.shard_size < 1 or args.torch_threads < 1:
        raise ValueError("shard-size and torch-threads must be positive")
    for path in (args.field_weights, args.classifier_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    samples = discover(args.dataset_root)
    parts = args.output_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    part_paths: list[Path] = []
    for number, start in enumerate(range(0, len(samples), args.shard_size), start=1):
        stop = min(start + args.shard_size, len(samples))
        shard_samples = samples[start:stop]
        part = parts / f"part_{number:03d}.jsonl"
        part_paths.append(part)
        if not _valid_shard(part, shard_samples):
            if part.exists():
                part.unlink()
            worker = context.Process(
                target=_score_worker,
                args=(
                    shard_samples,
                    str(args.field_weights),
                    str(args.classifier_weights),
                    str(args.vendor_root),
                    str(part),
                    args.torch_threads,
                ),
            )
            worker.start()
            worker.join()
            if worker.exitcode != 0 or not _valid_shard(part, shard_samples):
                raise RuntimeError(f"PerspectiveFields checkpoint shard {number} failed ({worker.exitcode=})")
        print(f"official_perspective_fields {stop}/{len(samples)}", flush=True)
    rows = [
        json.loads(line)
        for part in part_paths
        for line in part.read_text(encoding="utf-8").splitlines()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "schema_version": "official-perspective-fields-audit-v1",
        "checkpoint": {
            "classifier_sha256": _sha256(args.classifier_weights),
            "classifier_path": str(args.classifier_weights),
            "field_model_sha256": _sha256(args.field_weights),
            "field_model_path": str(args.field_weights),
        },
        "decision": "research_only_not_installed_in_origin_assessment",
        "input": "official PerspectiveFields latitude/90 plus gravity fields; no RGB, metadata, P0 candidate or P3 detector input",
        "protocol": "Official fixed combined classifier and fixed 0.5 binary decision; SDXL labels never used for selection or threshold tuning.",
        "sample_count": len(rows),
        "overall": _metrics(rows),
        "by_scene": {scene: _metrics([row for row in rows if row["scene"] == scene]) for scene in ("indoor", "outdoor")},
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["overall"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
