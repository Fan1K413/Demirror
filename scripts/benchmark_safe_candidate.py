"""Calibrate the local AIGIBench SAFE checkpoint on unseen generators."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import brier_score_loss, roc_auc_score


def _tensor(path: Path, jpeg_quality: int | None = None) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if jpeg_quality is not None:
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=jpeg_quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        width, height = image.size
        if width < 256 or height < 256:
            scale = 256.0 / min(width, height)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
            width, height = image.size
        left = (width - 256) // 2
        top = (height - 256) // 2
        image = image.crop((left, top, left + 256, top + 256))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def _scores(model: torch.nn.Module, paths: list[Path], batch_size: int, jpeg_quality: int | None) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([_tensor(path, jpeg_quality) for path in batch_paths])
            probability = torch.softmax(model(images), dim=1)[:, 1]
            values.append(probability.numpy())
            print(f"SAFE {min(start + batch_size, len(paths))}/{len(paths)} q={jpeg_quality}", flush=True)
    return np.concatenate(values)


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    real = labels == 0
    generated = labels == 1
    predicted = scores >= threshold
    return {
        "count": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "threshold": threshold,
        "true_positive_rate": float(predicted[generated].mean()),
        "false_positive_rate": float(predicted[real].mean()),
        "real_mean": float(scores[real].mean()),
        "generated_mean": float(scores[generated].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-cache", type=Path, default=Path("outputs/deeplsd_geometry_v1/line_cache.npz"))
    parser.add_argument("--safe-source", type=Path, default=Path("data/vendor/AIGIBench/detector_codes/SAFE-main"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/aigibench-safe/SAFE-main/checkpoint-best.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/safe_candidate_v1"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--target-fpr", type=float, default=0.08)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.safe_source.resolve()))
    from models.resnet import resnet50

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = resnet50(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    manifest = np.load(args.line_cache, allow_pickle=False)
    records = [json.loads(value) for value in manifest["records"].tolist()]
    splits = np.asarray([record["split"] for record in records])
    labels = np.asarray([record["label"] for record in records])
    scenes = np.asarray([record["scene"] for record in records])
    paths = np.asarray([Path(record["path"]) for record in records], dtype=object)
    calibration = np.flatnonzero(splits == "calibration")
    test = np.flatnonzero(splits == "test")
    calibration_score = _scores(model, paths[calibration].tolist(), args.batch_size, None)
    real_score = calibration_score[labels[calibration] == 0]
    threshold = float(np.quantile(real_score, 1.0 - args.target_fpr, method="higher"))
    test_score = _scores(model, paths[test].tolist(), args.batch_size, None)
    compressed_score = _scores(model, paths[test].tolist(), args.batch_size, 75)
    report = {
        "calibration": _metrics(labels[calibration], calibration_score, threshold),
        "held_out_test": _metrics(labels[test], test_score, threshold),
        "held_out_jpeg75": _metrics(labels[test], compressed_score, threshold),
        "held_out_by_scene": {
            scene: _metrics(
                labels[test[scenes[test] == scene]],
                test_score[scenes[test] == scene],
                threshold,
            )
            for scene in ("indoor", "outdoor")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores.npz").write_bytes(b"")
    np.savez_compressed(
        args.output_dir / "scores.npz",
        calibration_indices=calibration,
        calibration_scores=calibration_score,
        test_indices=test,
        test_scores=test_score,
        test_jpeg75_scores=compressed_score,
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
