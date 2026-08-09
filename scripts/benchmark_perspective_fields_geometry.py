"""Audit dense perspective-field cues for AI-image detection.

This experiment reproduces the *representation boundary* of the official
Projective Geometry perspective-field route: an image is first converted to a
latitude/gravity field by PerspectiveFields, then the classifier sees only the
three derived geometry channels.  It deliberately excludes RGB pixels, EXIF,
C2PA, line-count features and every production AI detector.

The split is generator-family isolated:

* train: DeepFloyd and Kandinsky, IDs 1--350;
* calibration: PixArt, IDs 351--425;
* evaluation: SDXL, IDs 426--500.

The PerspectiveFields model is loaded in a disposable worker for each shard.
This keeps its large CPU allocation out of the web process and makes extraction
resumable after an interrupted run.  The resulting model is research-only;
even a promising score must be replicated on a new, never-opened holdout before
it can affect the product.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260809
FIELD_SIDE = 64


@dataclass(frozen=True)
class Sample:
    """One source-labelled image with a split assigned before extraction."""

    archive: str
    generator: str
    identifier: int
    label: int
    path: str
    scene: str
    sha256: str
    split: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generator_and_scene(archive: str) -> tuple[str, str]:
    parts = archive.lower().split("_")
    if len(parts) != 3 or parts[0] != "recent":
        raise ValueError(f"Unexpected Projective Geometry archive name: {archive}")
    return parts[1], parts[2]


def _split_for(generator: str, identifier: int) -> str | None:
    if generator in {"deepfloyd", "kandinsky"} and 1 <= identifier <= 350:
        return "train"
    if generator == "pixart" and 351 <= identifier <= 425:
        return "calibration"
    if generator == "sdxl" and 426 <= identifier <= 500:
        return "test"
    return None


def discover(roots: list[Path]) -> list[Sample]:
    """Discover the four-generator protocol while removing duplicate reals."""

    rows: list[Sample] = []
    seen_hashes: dict[str, set[str]] = {"train": set(), "calibration": set(), "test": set()}
    for root in roots:
        for path in sorted(root.glob("**/test/*/*.jpg")):
            label_name = path.parent.name
            if label_name not in {"real", "gen"} or not path.stem.isdecimal():
                continue
            archive = path.parents[2].name
            generator, scene = _generator_and_scene(archive)
            identifier = int(path.stem)
            split = _split_for(generator, identifier)
            if split is None:
                continue
            # The supplied benchmark repeats the same real image across
            # generator folders.  Retain one predetermined owner per split so
            # a source copy cannot appear in both classes or train/test.
            if label_name == "real":
                owner = {"train": "deepfloyd", "calibration": "pixart", "test": "sdxl"}[split]
                if generator != owner:
                    continue
            digest = _sha256(path)
            if digest in seen_hashes[split]:
                continue
            seen_hashes[split].add(digest)
            rows.append(
                Sample(
                    archive=archive,
                    generator=generator,
                    identifier=identifier,
                    label=int(label_name == "gen"),
                    path=str(path.resolve()),
                    scene=scene,
                    sha256=digest,
                    split=split,
                )
            )
    rows.sort(key=lambda row: (row.split, row.archive, row.label, row.identifier, row.path))
    if not rows:
        raise ValueError("No Projective Geometry samples found under the requested roots")
    for split in ("train", "calibration", "test"):
        labels = {row.label for row in rows if row.split == split}
        if labels != {0, 1}:
            raise ValueError(f"Split {split} must contain both real and generated images")
    return rows


def _load_perspective_fields(weights_path: Path, *, threads: int):
    """Load the local-only official PerspectiveFields model without downloads."""

    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)
    from perspective2d.perspectivefields import PerspectiveFields

    original_loader = torch.hub.load_state_dict_from_url

    def local_loader(*_args: Any, **_kwargs: Any):
        return torch.load(weights_path, map_location="cpu")

    torch.hub.load_state_dict_from_url = local_loader
    try:
        return PerspectiveFields("Paramnet-360Cities-edina-uncentered").to("cpu").eval()
    finally:
        torch.hub.load_state_dict_from_url = original_loader


def _field_from_prediction(prediction: dict[str, Any]) -> np.ndarray:
    latitude = prediction["pred_latitude_original"].detach().float().cpu().unsqueeze(0) / 90.0
    gravity = prediction["pred_gravity_original"].detach().float().cpu()
    field = torch.cat((latitude, gravity), dim=0)
    field = functional.interpolate(
        field.unsqueeze(0),
        size=(FIELD_SIDE, FIELD_SIDE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if not bool(torch.isfinite(field).all()):
        raise ValueError("PerspectiveFields produced non-finite geometry values")
    return field.numpy().astype(np.float16, copy=False)


def _extract_shard_worker(
    samples: list[Sample], weights_path: str, output_path: str, threads: int
) -> None:
    """Extract a bounded shard in its own process to keep peak RSS bounded."""

    model = _load_perspective_fields(Path(weights_path), threads=threads)
    fields: list[np.ndarray] = []
    with torch.inference_mode():
        for sample in samples:
            image = cv2.imread(sample.path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unable to decode {sample.path}")
            fields.append(_field_from_prediction(model.inference(img_bgr=image)))
    np.savez_compressed(
        output_path,
        fields=np.stack(fields),
        records=np.asarray([json.dumps(asdict(sample), sort_keys=True) for sample in samples]),
    )


def _valid_shard(path: Path, samples: list[Sample]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as shard:
            records = [json.loads(value) for value in shard["records"].tolist()]
            fields = shard["fields"]
            return records == [asdict(sample) for sample in samples] and fields.shape == (
                len(samples),
                3,
                FIELD_SIDE,
                FIELD_SIDE,
            )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def extract_fields(
    samples: list[Sample],
    *,
    weights_path: Path,
    cache_path: Path,
    shard_size: int,
    threads: int,
    max_new_shards: int | None,
) -> np.ndarray | None:
    """Return cached fields, safely resuming only missing or invalid shards."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if _valid_shard(cache_path, samples):
        with np.load(cache_path, allow_pickle=False) as cached:
            print(f"Using verified field cache: {cache_path}", flush=True)
            return cached["fields"].copy()

    shard_dir = cache_path.parent / f"{cache_path.stem}_parts"
    shard_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    shard_paths: list[Path] = []
    new_shards = 0
    for shard_number, start in enumerate(range(0, len(samples), shard_size), start=1):
        stop = min(start + shard_size, len(samples))
        shard_samples = samples[start:stop]
        shard_path = shard_dir / f"part_{shard_number:04d}.npz"
        shard_paths.append(shard_path)
        if not _valid_shard(shard_path, shard_samples):
            if max_new_shards is not None and new_shards >= max_new_shards:
                print(
                    f"extraction_paused_after_new_shards={new_shards}; rerun the same command to resume",
                    flush=True,
                )
                return None
            if shard_path.exists():
                shard_path.unlink()
            process = context.Process(
                target=_extract_shard_worker,
                args=(shard_samples, str(weights_path), str(shard_path), threads),
            )
            process.start()
            process.join()
            if process.exitcode != 0 or not _valid_shard(shard_path, shard_samples):
                raise RuntimeError(f"PerspectiveFields shard {shard_number} failed ({process.exitcode=})")
            new_shards += 1
        print(f"PerspectiveFields {stop}/{len(samples)} (shard {shard_number}/{len(range(0, len(samples), shard_size))})", flush=True)

    field_parts: list[np.ndarray] = []
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            field_parts.append(shard["fields"].copy())
    fields = np.concatenate(field_parts, axis=0)
    np.savez_compressed(
        cache_path,
        fields=fields,
        records=np.asarray([json.dumps(asdict(sample), sort_keys=True) for sample in samples]),
    )
    return fields


class FieldGeometryNet(nn.Module):
    """Small classifier for derived geometry fields, not raw image content."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.20), nn.Linear(128, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(value))


def _probability(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(features), dim=1)[:, 1].cpu().numpy()


def _threshold_from_calibration(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        score = balanced_accuracy_score(labels, probabilities >= threshold)
        candidate = (float(score), float(threshold))
        if candidate > best:
            best = candidate
    return best[1]


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    real = labels == 0
    generated = labels == 1
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "false_positive_rate": float(predicted[real].mean()),
        "true_positive_rate": float(predicted[generated].mean()),
        "threshold": float(threshold),
    }


def train_and_evaluate(
    samples: list[Sample], fields: np.ndarray, *, epochs: int, threads: int
) -> tuple[FieldGeometryNet, dict[str, Any], dict[str, torch.Tensor]]:
    """Select the checkpoint and threshold using PixArt only; SDXL stays evaluation-only."""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)

    split = np.asarray([row.split for row in samples])
    labels = np.asarray([row.label for row in samples], dtype=np.int64)
    indices = {name: np.flatnonzero(split == name) for name in ("train", "calibration", "test")}
    train_values = fields[indices["train"]].astype(np.float32)
    channel_mean = train_values.mean(axis=(0, 2, 3), keepdims=True)
    channel_scale = train_values.std(axis=(0, 2, 3), keepdims=True)
    channel_scale[channel_scale < 1e-6] = 1.0
    normalised = (fields.astype(np.float32) - channel_mean) / channel_scale

    features = torch.from_numpy(normalised)
    label_tensor = torch.from_numpy(labels)
    train_indices = torch.from_numpy(indices["train"])
    calibration_indices = torch.from_numpy(indices["calibration"])
    test_indices = torch.from_numpy(indices["test"])
    dataset = TensorDataset(features[train_indices], label_tensor[train_indices])
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    model = FieldGeometryNet().cpu()
    train_labels = labels[indices["train"]]
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    loss = nn.CrossEntropyLoss(weight=torch.from_numpy(class_counts.sum() / (2.0 * class_counts)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state: dict[str, torch.Tensor] | None = None
    best_auc = float("-inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        accumulated_loss = 0.0
        for batch, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(batch), batch_labels)
            value.backward()
            optimizer.step()
            accumulated_loss += float(value.item()) * len(batch)
        calibration_probability = _probability(model, features[calibration_indices])
        calibration_auc = float(roc_auc_score(labels[indices["calibration"]], calibration_probability))
        history.append(
            {
                "epoch": epoch,
                "training_loss": accumulated_loss / len(dataset),
                "calibration_roc_auc": calibration_auc,
            }
        )
        print(f"epoch={epoch} calibration_auc={calibration_auc:.4f}", flush=True)
        if calibration_auc > best_auc:
            best_auc = calibration_auc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 8:
                break
    if best_state is None:
        raise RuntimeError("No PerspectiveFields checkpoint was selected")
    model.load_state_dict(best_state)
    calibration_probability = _probability(model, features[calibration_indices])
    threshold = _threshold_from_calibration(labels[indices["calibration"]], calibration_probability)
    test_probability = _probability(model, features[test_indices])
    report = {
        "representation": "PerspectiveFields pred_latitude_original/90 plus pred_gravity_original; bilinear downsampled to 3x64x64",
        "model": "FieldGeometryNet; RGB pixels and all provenance/pixel-detector outputs excluded",
        "selection": "best calibration ROC-AUC epoch and threshold selected only from PixArt IDs 351-425",
        "normalisation": {"mean": channel_mean.reshape(-1).tolist(), "scale": channel_scale.reshape(-1).tolist()},
        "history": history,
        "calibration": _metrics(labels[indices["calibration"]], calibration_probability, threshold),
        "held_out_test": _metrics(labels[indices["test"]], test_probability, threshold),
        "held_out_by_scene": {
            scene: _metrics(
                labels[test_indices][np.asarray([samples[index].scene == scene for index in test_indices])],
                test_probability[np.asarray([samples[index].scene == scene for index in test_indices])],
                threshold,
            )
            for scene in ("indoor", "outdoor")
        },
    }
    saved = {
        "model": best_state,
        "channel_mean": torch.from_numpy(channel_mean.astype(np.float32)),
        "channel_scale": torch.from_numpy(channel_scale.astype(np.float32)),
    }
    return model, report, saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--max-new-shards",
        type=int,
        help="Stop cleanly after this many newly extracted shards; use the same command to resume.",
    )
    args = parser.parse_args()
    if args.shard_size < 1 or args.torch_threads < 1 or args.epochs < 1 or (
        args.max_new_shards is not None and args.max_new_shards < 1
    ):
        raise ValueError("numeric arguments must all be positive")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    samples = discover(args.roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(asdict(sample), sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )
    fields = extract_fields(
        samples,
        weights_path=args.weights,
        cache_path=args.output_dir / "perspective_fields_64.npz",
        shard_size=args.shard_size,
        threads=args.torch_threads,
        max_new_shards=args.max_new_shards,
    )
    if fields is None:
        return 0
    if args.extract_only:
        print(f"extracted={len(samples)}", flush=True)
        return 0
    model, report, saved = train_and_evaluate(
        samples,
        fields,
        epochs=args.epochs,
        threads=args.torch_threads,
    )
    del model
    torch.save(saved, args.output_dir / "field_geometry_model.pt")
    report.update(
        {
            "schema_version": "perspective-fields-geometry-audit-v1",
            "source": {
                "dataset": "https://huggingface.co/datasets/amitabh3/Projective-Geometry",
                "repository": "https://github.com/hanlinm2/projective-geometry/",
                "weights": str(args.weights),
                "weights_sha256": _sha256(args.weights),
            },
            "sample_counts": {
                split: {
                    "real": sum(row.split == split and row.label == 0 for row in samples),
                    "generated": sum(row.split == split and row.label == 1 for row in samples),
                }
                for split in ("train", "calibration", "test")
            },
            "decision": "research_only_not_installed_in_origin_assessment",
        }
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["held_out_test"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
