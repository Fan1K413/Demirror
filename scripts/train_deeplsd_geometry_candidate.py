"""Train and audit a DeepLSD geometry-only AI-image detector.

The official Projective Geometry checkpoint does not transfer reliably to an
unseen generator in our local evaluation.  This script keeps the paper's
DeepLSD representation but trains a small permutation-invariant classifier on
multiple generators, calibrates it on PixArt, and evaluates exactly once on
held-out SDXL identifiers.  Extracted line sets are cached so model iterations
do not repeatedly run DeepLSD.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from image_trust.geometry_ai.features import relationship_features


SEED = 20260809
MAX_LINES = 250


@dataclass(frozen=True)
class Sample:
    path: str
    archive: str
    generator: str
    scene: str
    label: int
    identifier: int
    split: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(roots: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    seen: dict[str, set[str]] = {"train": set(), "calibration": set(), "test": set()}
    for root in roots:
        for label_dir in root.rglob("*"):
            if not label_dir.is_dir() or label_dir.name not in {"real", "gen"}:
                continue
            archive = label_dir.parents[2].name
            generator = archive.removeprefix("Recent_").split("_", 1)[0].lower()
            scene = "indoor" if "Indoor" in archive else "outdoor"
            for path in sorted(label_dir.iterdir()):
                if not path.is_file() or not path.stem.isdigit():
                    continue
                identifier = int(path.stem)
                if identifier <= 350 and generator in {"deepfloyd", "kandinsky"}:
                    split = "train"
                elif 351 <= identifier <= 425 and generator == "pixart":
                    split = "calibration"
                elif 426 <= identifier <= 500 and generator == "sdxl":
                    split = "test"
                else:
                    continue
                # The four archives contain byte-identical real controls.
                # Retain one copy per scene/split and deduplicate all exact files.
                real_owner = {"train": "deepfloyd", "calibration": "pixart", "test": "sdxl"}[split]
                if label_dir.name == "real" and generator != real_owner:
                    continue
                digest = _sha256(path)
                if digest in seen[split]:
                    continue
                seen[split].add(digest)
                samples.append(
                    Sample(
                        path=str(path),
                        archive=archive,
                        generator=generator,
                        scene=scene,
                        label=int(label_dir.name == "gen"),
                        identifier=identifier,
                        split=split,
                        sha256=digest,
                    )
                )
    return sorted(samples, key=lambda row: (row.split, row.archive, row.label, row.identifier))


def _load_deeplsd(checkpoint_path: Path) -> nn.Module:
    from deeplsd.models.deeplsd_inference import DeepLSD

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DeepLSD(
        {
            "detect_lines": True,
            "line_detection_params": {
                "merge": False,
                "filtering": "normal",
                "grad_thresh": 3,
                "grad_nfa": True,
            },
        }
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def _prepare_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    if image.shape != (256, 256):
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def _canonical_lines(lines: np.ndarray) -> np.ndarray:
    lines = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
    if not len(lines):
        return lines
    swap = (lines[:, 0] > lines[:, 2]) | (
        (lines[:, 0] == lines[:, 2]) & (lines[:, 1] > lines[:, 3])
    )
    lines[swap] = lines[swap][:, [2, 3, 0, 1]]
    lengths = np.linalg.norm(lines[:, 2:4] - lines[:, :2], axis=1)
    return lines[np.argsort(lengths)[::-1]][:MAX_LINES]


def _extract_shard_worker(
    samples: list[Sample], checkpoint_path: str, shard_path: str, threads: int
) -> None:
    """Extract one shard in a disposable process.

    pytlsd's Windows extension slowly retains native allocations across many
    calls.  Keeping extraction in bounded child processes prevents that native
    growth from accumulating until the machine becomes unresponsive.
    """
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)
    model = _load_deeplsd(Path(checkpoint_path))
    padded = np.zeros((len(samples), MAX_LINES, 4), dtype=np.float32)
    counts = np.zeros(len(samples), dtype=np.int16)
    relation_rows: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        image = _prepare_image(Path(sample.path))
        with torch.inference_mode():
            output = model({"image": torch.from_numpy(image[None, None])})
        lines = _canonical_lines(output["lines"][0])
        count = len(lines)
        padded[index, :count] = lines
        counts[index] = count
        relation_rows.append(
            np.asarray(
                list(relationship_features(lines, (256, 256)).values()),
                dtype=np.float32,
            )
        )
        del output
        if (index + 1) % 10 == 0:
            gc.collect()
    np.savez_compressed(
        shard_path,
        lines=padded,
        counts=counts,
        relations=np.stack(relation_rows),
        records=np.asarray([json.dumps(asdict(sample), sort_keys=True) for sample in samples]),
    )


def _valid_shard(path: Path, samples: list[Sample]) -> bool:
    if not path.exists():
        return False
    try:
        shard = np.load(path, allow_pickle=False)
        records = [json.loads(value) for value in shard["records"].tolist()]
        return records == [asdict(sample) for sample in samples]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def extract_cache(
    samples: list[Sample],
    checkpoint_path: Path,
    cache_path: Path,
    shard_size: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract resumable shards with a hard process-level memory boundary."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = cache_path.parent / f"{cache_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    shard_paths: list[Path] = []
    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        stop = min(start + shard_size, len(samples))
        shard_samples = samples[start:stop]
        shard_path = parts_dir / f"part_{shard_index:04d}.npz"
        shard_paths.append(shard_path)
        if not _valid_shard(shard_path, shard_samples):
            if shard_path.exists():
                shard_path.unlink()
            process = context.Process(
                target=_extract_shard_worker,
                args=(shard_samples, str(checkpoint_path), str(shard_path), threads),
            )
            process.start()
            process.join()
            if process.exitcode != 0 or not _valid_shard(shard_path, shard_samples):
                raise RuntimeError(
                    f"DeepLSD shard {shard_index} failed with exit code {process.exitcode}"
                )
        print(f"DeepLSD {stop}/{len(samples)} (shard {shard_index + 1}/{len(range(0, len(samples), shard_size))})", flush=True)

    line_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    relation_parts: list[np.ndarray] = []
    for shard_path in shard_paths:
        shard = np.load(shard_path, allow_pickle=False)
        line_parts.append(shard["lines"])
        count_parts.append(shard["counts"])
        relation_parts.append(shard["relations"])
    padded = np.concatenate(line_parts)
    counts = np.concatenate(count_parts)
    relations = np.concatenate(relation_parts)
    np.savez_compressed(
        cache_path,
        lines=padded,
        counts=counts,
        relations=relations,
        records=np.asarray([json.dumps(asdict(sample), sort_keys=True) for sample in samples]),
    )
    return padded, counts, relations


def _legacy_extract_cache(
    samples: list[Sample], checkpoint_path: Path, cache_path: Path, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kept as a profiling reference; production extraction uses shards."""
    model = _load_deeplsd(checkpoint_path)
    padded = np.zeros((len(samples), MAX_LINES, 4), dtype=np.float32)
    counts = np.zeros(len(samples), dtype=np.int16)
    relation_rows: list[np.ndarray] = []
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        images = np.stack([_prepare_image(Path(sample.path)) for sample in samples[start:stop]])
        with torch.inference_mode():
            output = model({"image": torch.from_numpy(images[:, None])})
        for offset, raw_lines in enumerate(output["lines"]):
            index = start + offset
            lines = _canonical_lines(raw_lines)
            count = len(lines)
            padded[index, :count] = lines
            counts[index] = count
            relation_rows.append(
                np.asarray(
                    list(relationship_features(lines, (256, 256)).values()),
                    dtype=np.float32,
                )
            )
        print(f"DeepLSD {stop}/{len(samples)}", flush=True)
    relations = np.stack(relation_rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        lines=padded,
        counts=counts,
        relations=relations,
        records=np.asarray([json.dumps(asdict(sample), sort_keys=True) for sample in samples]),
    )
    return padded, counts, relations


def load_or_extract(
    samples: list[Sample],
    checkpoint_path: Path,
    cache_path: Path,
    shard_size: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        records = [json.loads(value) for value in cache["records"].tolist()]
        expected = [asdict(sample) for sample in samples]
        if records == expected:
            print(f"Using verified DeepLSD cache: {cache_path}", flush=True)
            return cache["lines"], cache["counts"], cache["relations"]
        print("DeepLSD cache manifest changed; rebuilding", flush=True)
    return extract_cache(samples, checkpoint_path, cache_path, shard_size, threads)


def line_features(lines: np.ndarray) -> np.ndarray:
    result = np.zeros((len(lines), MAX_LINES, 11), dtype=np.float32)
    for index, row in enumerate(lines):
        p1 = row[:, :2]
        p2 = row[:, 2:4]
        delta = p2 - p1
        midpoint = (p1 + p2) * 0.5
        length = np.linalg.norm(delta, axis=1)
        angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), np.pi)
        result[index] = np.column_stack(
            [
                p1[:, 0] / 127.5 - 1.0,
                p1[:, 1] / 127.5 - 1.0,
                p2[:, 0] / 127.5 - 1.0,
                p2[:, 1] / 127.5 - 1.0,
                midpoint[:, 0] / 127.5 - 1.0,
                midpoint[:, 1] / 127.5 - 1.0,
                delta[:, 0] / math.hypot(256, 256),
                delta[:, 1] / math.hypot(256, 256),
                length / math.hypot(256, 256),
                np.cos(2.0 * angle),
                np.sin(2.0 * angle),
            ]
        )
    return result


class DeepLSDSetModel(nn.Module):
    def __init__(self, relation_dimension: int) -> None:
        super().__init__()
        self.line_encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, lines: torch.Tensor, mask: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        encoded = self.line_encoder(lines)
        expanded = mask.unsqueeze(-1)
        denominator = expanded.sum(dim=1).clamp_min(1.0)
        mean = (encoded * expanded).sum(dim=1) / denominator
        variance = (((encoded - mean.unsqueeze(1)) ** 2) * expanded).sum(dim=1) / denominator
        maximum = encoded.masked_fill(expanded == 0.0, -1e4).max(dim=1).values
        maximum = torch.where(mask.sum(dim=1, keepdim=True) > 0, maximum, torch.zeros_like(maximum))
        relation_embedding = self.relation_encoder(relations)
        pooled = torch.cat([mean, variance.clamp_min(1e-6).sqrt(), maximum, relation_embedding], dim=1)
        return self.classifier(pooled).squeeze(1)


def _logits(
    model: DeepLSDSetModel,
    lines: np.ndarray,
    masks: np.ndarray,
    relations: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            batch = indices[start : start + 256]
            values.append(
                model(
                    torch.from_numpy(lines[batch]),
                    torch.from_numpy(masks[batch]),
                    torch.from_numpy(relations[batch]),
                ).numpy()
            )
    return np.concatenate(values)


def _threshold_at_fpr(real_probability: np.ndarray, fpr: float) -> float:
    return float(np.quantile(real_probability, 1.0 - fpr, method="higher"))


def _metrics(labels: np.ndarray, values: np.ndarray, threshold: float) -> dict[str, Any]:
    real = labels == 0
    generated = labels == 1
    predicted = values >= threshold
    return {
        "count": int(len(labels)),
        "real_count": int(real.sum()),
        "generated_count": int(generated.sum()),
        "roc_auc": float(roc_auc_score(labels, values)),
        "brier_score": float(brier_score_loss(labels, values)),
        "threshold": threshold,
        "true_positive_rate": float(predicted[generated].mean()),
        "false_positive_rate": float(predicted[real].mean()),
        "mean_probability_real": float(values[real].mean()),
        "mean_probability_generated": float(values[generated].mean()),
    }


def train(args: argparse.Namespace) -> tuple[DeepLSDSetModel, dict[str, Any], dict[str, np.ndarray]]:
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(SEED)
    samples = discover(args.dataset_roots)
    raw_lines, counts, relations = load_or_extract(
        samples, args.deeplsd_checkpoint, args.cache, args.extract_shard_size, args.threads
    )
    lines = line_features(raw_lines)
    masks = (np.arange(MAX_LINES)[None, :] < counts[:, None]).astype(np.float32)
    labels = np.asarray([sample.label for sample in samples], dtype=np.float32)
    splits = np.asarray([sample.split for sample in samples])
    train_indices = np.flatnonzero(splits == "train")
    calibration_indices = np.flatnonzero(splits == "calibration")
    test_indices = np.flatnonzero(splits == "test")

    relation_mean = relations[train_indices].astype(np.float64).mean(axis=0)
    relation_scale = relations[train_indices].astype(np.float64).std(axis=0)
    relation_scale[relation_scale == 0.0] = 1.0
    relations = np.nan_to_num(relations, nan=0.0, posinf=10.0, neginf=-10.0)
    relations = np.clip((relations - relation_mean) / relation_scale, -12.0, 12.0).astype(np.float32)

    model = DeepLSDSetModel(relations.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    train_labels = labels[train_indices]
    positive_weight = float((train_labels == 0).sum() / (train_labels == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            batch_masks = torch.from_numpy(masks[batch])
            keep = (torch.rand_like(batch_masks) >= 0.08).float()
            batch_masks = batch_masks * keep
            optimizer.zero_grad(set_to_none=True)
            output = model(
                torch.from_numpy(lines[batch]),
                batch_masks,
                torch.from_numpy(relations[batch]),
            )
            loss = loss_fn(output, torch.from_numpy(labels[batch]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        calibration_logit = _logits(model, lines, masks, relations, calibration_indices)
        calibration_auc = float(roc_auc_score(labels[calibration_indices], calibration_logit))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "calibration_auc": calibration_auc})
        print(f"epoch {epoch}: loss={np.mean(losses):.4f} calibration_auc={calibration_auc:.4f}", flush=True)
        if calibration_auc > best_auc + 1e-4:
            best_auc = calibration_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No candidate checkpoint was produced")
    model.load_state_dict(best_state)

    calibration_logit = _logits(model, lines, masks, relations, calibration_indices)
    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(
        calibration_logit.reshape(-1, 1), labels[calibration_indices]
    )
    calibration_probability = calibrator.predict_proba(calibration_logit.reshape(-1, 1))[:, 1]
    ai_threshold = _threshold_at_fpr(
        calibration_probability[labels[calibration_indices] == 0], args.target_fpr
    )
    test_logit = _logits(model, lines, masks, relations, test_indices)
    test_probability = calibrator.predict_proba(test_logit.reshape(-1, 1))[:, 1]
    evaluation = {
        "best_epoch": best_epoch,
        "best_calibration_auc_during_training": best_auc,
        "calibration": _metrics(labels[calibration_indices], calibration_probability, ai_threshold),
        "held_out_test": _metrics(labels[test_indices], test_probability, ai_threshold),
        "by_scene": {
            scene: _metrics(
                labels[test_indices[np.asarray([samples[i].scene == scene for i in test_indices])]],
                test_probability[np.asarray([samples[i].scene == scene for i in test_indices])],
                ai_threshold,
            )
            for scene in ("indoor", "outdoor")
        },
        "protocol": {
            "train": "DeepFloyd + Kandinsky ids 1-350; exact-hash-deduplicated real controls",
            "calibration": "PixArt ids 351-425",
            "held_out_test": "SDXL ids 426-500",
            "target_false_positive_rate": args.target_fpr,
            "deeplsd_sha256": _sha256(args.deeplsd_checkpoint),
        },
        "history": history,
    }
    bundle = {
        "relation_mean": relation_mean.astype(np.float32),
        "relation_scale": relation_scale.astype(np.float32),
        "platt_coefficient": np.asarray([float(calibrator.coef_[0][0])], dtype=np.float32),
        "platt_intercept": np.asarray([float(calibrator.intercept_[0])], dtype=np.float32),
        "ai_threshold": np.asarray([ai_threshold], dtype=np.float32),
    }
    return model, evaluation, bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=[Path("data/p2_projective_geometry_v1/extracted"), Path("data/p3_aigc_v2/extracted")],
    )
    parser.add_argument("--deeplsd-checkpoint", type=Path, default=Path("weights/deeplsd/deeplsd_md.tar"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/deeplsd_geometry_v1/line_cache.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deeplsd_geometry_v1"))
    parser.add_argument("--extract-shard-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-fpr", type=float, default=0.08)
    args = parser.parse_args()
    model, evaluation, bundle = train(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "relation_dimension": model.relation_encoder[0].in_features,
            **{name: torch.from_numpy(value) for name, value in bundle.items()},
        },
        args.output_dir / "candidate.pt",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in evaluation.items() if key != "history"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
