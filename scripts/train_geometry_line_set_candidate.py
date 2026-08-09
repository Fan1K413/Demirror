"""Train a permutation-invariant line-set geometry candidate.

This is the higher-capacity successor to the relationship-histogram baseline.
It consumes individual normalized line segments and pooled relationship
features, while retaining the same generator- and identifier-held-out protocol.
The script is deliberately separate from web integration until its gate passes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from image_trust.geometry_ai.features import (
    MAX_LINES,
    extract_image_lines,
    relationship_features,
)


SEED = 20260809


@dataclass(frozen=True)
class Sample:
    path: Path
    archive: str
    scene: str
    label: int
    identifier: int
    split: str
    sha256: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def discover(roots: list[Path]) -> list[Sample]:
    result: list[Sample] = []
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
                if label_dir.name == "real" and generator != {
                    "train": "deepfloyd",
                    "calibration": "pixart",
                    "test": "sdxl",
                }[split]:
                    continue
                file_hash = sha256(path)
                if file_hash in seen[split]:
                    continue
                seen[split].add(file_hash)
                result.append(
                    Sample(
                        path=path,
                        archive=archive,
                        scene=scene,
                        label=int(label_dir.name == "gen"),
                        identifier=identifier,
                        split=split,
                        sha256=file_hash,
                    )
                )
    return sorted(result, key=lambda item: (item.split, item.archive, item.label, item.identifier))


def line_tensor(lines: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    output = np.zeros((MAX_LINES, 11), dtype=np.float32)
    mask = np.zeros(MAX_LINES, dtype=np.float32)
    if not len(lines):
        return output, mask
    lines = np.asarray(lines[:MAX_LINES], dtype=np.float64).copy()
    swap = (lines[:, 0] > lines[:, 2]) | ((lines[:, 0] == lines[:, 2]) & (lines[:, 1] > lines[:, 3]))
    lines[swap] = lines[swap][:, [2, 3, 0, 1]]
    p1 = lines[:, :2]
    p2 = lines[:, 2:4]
    delta = p2 - p1
    diagonal = math.hypot(width, height)
    midpoint = (p1 + p2) * 0.5
    angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), np.pi)
    features = np.column_stack(
        [
            p1[:, 0] / width * 2.0 - 1.0,
            p1[:, 1] / height * 2.0 - 1.0,
            p2[:, 0] / width * 2.0 - 1.0,
            p2[:, 1] / height * 2.0 - 1.0,
            midpoint[:, 0] / width * 2.0 - 1.0,
            midpoint[:, 1] / height * 2.0 - 1.0,
            delta[:, 0] / diagonal,
            delta[:, 1] / diagonal,
            np.linalg.norm(delta, axis=1) / diagonal,
            np.cos(2.0 * angle),
            np.sin(2.0 * angle),
        ]
    ).astype(np.float32)
    output[: len(features)] = features
    mask[: len(features)] = 1.0
    return output, mask


def flip_lines(lines: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, _ = size
    flipped = np.asarray(lines, dtype=np.float64).copy()
    flipped[:, [0, 2]] = (width - 1.0) - flipped[:, [0, 2]]
    return flipped


def extract(samples: list[Sample], workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    cv2.setNumThreads(1)

    def one(sample: Sample) -> tuple[Sample, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        lines, size = extract_image_lines(sample.path)
        points, mask = line_tensor(lines, size)
        relations = np.asarray(list(relationship_features(lines, size).values()), dtype=np.float32)
        flipped = flip_lines(lines, size)
        flipped_points, flipped_mask = line_tensor(flipped, size)
        flipped_relations = np.asarray(list(relationship_features(flipped, size).values()), dtype=np.float32)
        return sample, points, mask, relations, np.stack([flipped_points, flipped_mask[:, None].repeat(11, axis=1)]), len(lines)

    # Return flip arrays separately; the stacked carrier above keeps executor
    # result construction compact without retaining images.
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="line-set") as executor:
        raw = list(executor.map(one, samples))
    points = np.stack([row[1] for row in raw])
    masks = np.stack([row[2] for row in raw])
    relations = np.stack([row[3] for row in raw])
    labels = np.asarray([row[0].label for row in raw], dtype=np.float32)
    records = [
        {
            "path": str(row[0].path),
            "archive": row[0].archive,
            "scene": row[0].scene,
            "label": row[0].label,
            "identifier": row[0].identifier,
            "split": row[0].split,
            "sha256": row[0].sha256,
            "line_count": row[5],
        }
        for row in raw
    ]
    # Recompute the inexpensive relation vector for flips here because it was
    # intentionally not placed in the carrier tensor.
    flip_points: list[np.ndarray] = []
    flip_masks: list[np.ndarray] = []
    flip_relations: list[np.ndarray] = []
    train_indices = [index for index, sample in enumerate(samples) if sample.split == "train"]
    for index in train_indices:
        lines, size = extract_image_lines(samples[index].path)
        flipped = flip_lines(lines, size)
        fp, fm = line_tensor(flipped, size)
        flip_points.append(fp)
        flip_masks.append(fm)
        flip_relations.append(np.asarray(list(relationship_features(flipped, size).values()), dtype=np.float32))
    if flip_points:
        points = np.concatenate([points, np.stack(flip_points)], axis=0)
        masks = np.concatenate([masks, np.stack(flip_masks)], axis=0)
        relations = np.concatenate([relations, np.stack(flip_relations)], axis=0)
        labels = np.concatenate([labels, labels[train_indices]], axis=0)
        records.extend([{**records[index], "split": "train_augmented", "augmentation": "horizontal_flip"} for index in train_indices])
    return points, masks, relations, labels, records


class LineSetModel(nn.Module):
    def __init__(self, relation_dimension: int) -> None:
        super().__init__()
        self.line_encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
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
        # The derivative of sqrt at exactly zero is unbounded.  Padded or very
        # small line sets can have zero pooled variance, so clamp before sqrt.
        standard_deviation = variance.clamp_min(1e-6).sqrt()
        return self.classifier(torch.cat([mean, standard_deviation, maximum, relation_embedding], dim=1)).squeeze(1)


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    shuffled = rng.permutation(indices)
    return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]


def logits(model: LineSetModel, points: np.ndarray, masks: np.ndarray, relations: np.ndarray, indices: np.ndarray) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            batch = indices[start : start + 256]
            value = model(
                torch.from_numpy(points[batch]),
                torch.from_numpy(masks[batch]),
                torch.from_numpy(relations[batch]),
            )
            result.append(value.numpy())
    return np.concatenate(result)


def probability(calibrator: LogisticRegression, values: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(values.reshape(-1, 1))[:, 1]


def threshold_at_fpr(real_probability: np.ndarray, fpr: float) -> float:
    return float(np.quantile(real_probability, 1.0 - fpr, method="higher"))


def metrics(labels: np.ndarray, values: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = values >= threshold
    real = labels == 0
    generated = labels == 1
    return {
        "count": int(len(labels)),
        "real_count": int(real.sum()),
        "generated_count": int(generated.sum()),
        "roc_auc": float(roc_auc_score(labels, values)),
        "brier_score": float(brier_score_loss(labels, values)),
        "threshold": threshold,
        "true_positive_rate": float(prediction[generated].mean()),
        "false_positive_rate": float(prediction[real].mean()),
        "mean_probability_real": float(values[real].mean()),
        "mean_probability_generated": float(values[generated].mean()),
    }


def train(args: argparse.Namespace) -> tuple[LineSetModel, dict[str, object], dict[str, np.ndarray]]:
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(SEED)
    samples = discover(args.dataset_roots)
    points, masks, relations, labels, records = extract(samples, args.workers)
    original_count = len(samples)
    record_split = np.asarray([record["split"] for record in records])
    train_indices = np.flatnonzero(np.isin(record_split, ["train", "train_augmented"]))
    calibration_indices = np.flatnonzero(record_split == "calibration")
    test_indices = np.flatnonzero(record_split == "test")
    relations = np.nan_to_num(relations, nan=0.0, posinf=10.0, neginf=-10.0)
    relation_mean = relations[train_indices].astype(np.float64).mean(axis=0)
    relation_scale = relations[train_indices].astype(np.float64).std(axis=0)
    relation_scale[relation_scale == 0.0] = 1.0
    relations = np.clip((relations - relation_mean) / relation_scale, -12.0, 12.0).astype(np.float32)
    if not np.isfinite(points).all() or not np.isfinite(relations).all():
        raise ValueError("Non-finite geometry features remain after bounded standardization")

    model = LineSetModel(relations.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    train_labels = labels[train_indices]
    negative_count = float((train_labels == 0).sum())
    positive_count = float((train_labels == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative_count / positive_count))
    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in batches(train_indices, args.batch_size, rng):
            batch_lines = torch.from_numpy(points[batch])
            batch_mask = torch.from_numpy(masks[batch])
            # Random line dropout improves robustness to compression and LSD
            # instability without inventing new geometric relationships.
            if len(batch):
                keep = torch.rand_like(batch_mask) >= 0.08
                batch_mask = batch_mask * keep
            batch_relations = torch.from_numpy(relations[batch])
            batch_labels = torch.from_numpy(labels[batch])
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_lines, batch_mask, batch_relations)
            loss = loss_fn(output, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        calibration_logit = logits(model, points, masks, relations, calibration_indices)
        calibration_auc = float(roc_auc_score(labels[calibration_indices], calibration_logit))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "calibration_auc": calibration_auc})
        if calibration_auc > best_auc + 1e-4:
            best_auc = calibration_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No geometry line-set checkpoint was produced")
    model.load_state_dict(best_state)
    calibration_logit = logits(model, points, masks, relations, calibration_indices)
    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(
        calibration_logit.reshape(-1, 1), labels[calibration_indices]
    )
    calibration_probability = probability(calibrator, calibration_logit)
    ai_threshold = threshold_at_fpr(calibration_probability[labels[calibration_indices] == 0], 0.05)
    strong_threshold = threshold_at_fpr(calibration_probability[labels[calibration_indices] == 0], 0.01)
    test_logit = logits(model, points, masks, relations, test_indices)
    test_probability = probability(calibrator, test_logit)
    evaluation = {
        "best_epoch": best_epoch,
        "best_calibration_auc_during_training": best_auc,
        "calibration": metrics(labels[calibration_indices], calibration_probability, ai_threshold),
        "held_out_test": metrics(labels[test_indices], test_probability, ai_threshold),
        "protocol": {
            "train": "DeepFloyd + Kandinsky ids 1-350 plus horizontal flips",
            "calibration": "PixArt ids 351-425",
            "held_out_test": "SDXL ids 426-500",
            "exact_file_hash_deduplication": True,
            "original_sample_count": original_count,
            "training_sample_count_with_augmentation": int(len(train_indices)),
        },
        "history": history,
    }
    bundle = {
        "relation_mean": relation_mean,
        "relation_scale": relation_scale,
        "platt_coefficient": np.asarray([float(calibrator.coef_[0][0])]),
        "platt_intercept": np.asarray([float(calibrator.intercept_[0])]),
        "ai_threshold": np.asarray([ai_threshold]),
        "strong_threshold": np.asarray([strong_threshold]),
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry_line_set_candidate_v1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
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
