"""Probe the local AIGIBench LaDeDa checkpoint on compression controls."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score


def tensor(path: Path, jpeg_quality: int | None) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if jpeg_quality is not None:
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=jpeg_quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        value = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (value - mean) / std


def score(model: torch.nn.Module, paths: list[Path], quality: int | None) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), 8):
            batch = torch.stack([tensor(path, quality) for path in paths[start : start + 8]])
            values.append(torch.sigmoid(model(batch)).flatten().numpy())
    return np.concatenate(values)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str((root / "data/vendor/AIGIBench/detector_codes/RealTime-DeepfakeDetection-in-the-RealWorld-main").resolve()))
    from networks.LaDeDa import LaDeDa9

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = LaDeDa9(num_classes=1)
    checkpoint = root / "weights/aigibench-candidates/RealTime-DeepfakeDetection-in-the-RealWorld-main/model_epoch_best.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    safe_report = json.loads((root / "outputs/safe_local_controls_v1/report.json").read_text(encoding="utf-8"))
    paths = [root / row["path"] for row in safe_report["rows"]]
    labels = np.asarray([row["label"] for row in safe_report["rows"]])
    original = score(model, paths, None)
    jpeg75 = score(model, paths, 75)
    report = {
        "original_auc": float(roc_auc_score(labels, original)),
        "jpeg75_auc": float(roc_auc_score(labels, jpeg75)),
        "original_real_mean": float(original[labels == 0].mean()),
        "original_ai_mean": float(original[labels == 1].mean()),
        "jpeg75_real_mean": float(jpeg75[labels == 0].mean()),
        "jpeg75_ai_mean": float(jpeg75[labels == 1].mean()),
        "rows": [
            {"path": row["path"], "label": row["label"], "score": float(first), "jpeg75_score": float(second)}
            for row, first, second in zip(safe_report["rows"], original, jpeg75)
        ],
    }
    output = root / "outputs/ladeda_local_controls_v1/report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
