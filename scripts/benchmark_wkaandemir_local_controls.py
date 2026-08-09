"""Probe the local CLIP ViT-B/16 forensic detector on JPEG controls."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from torchvision import transforms


def main() -> int:
    import timm

    root = Path(__file__).resolve().parents[1]
    model_root = root / "weights/wkaandemir-ai-detector"
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = timm.create_model(
        config["backbone"], pretrained=False, num_classes=1, img_size=config["image_size"]
    )
    missing, unexpected = model.load_state_dict(load_file(str(model_root / "model.safetensors")), strict=False)
    if missing or unexpected:
        raise ValueError(f"state mismatch: {missing}, {unexpected}")
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(config["normalization_mean"], config["normalization_std"]),
        ]
    )
    safe_report = json.loads((root / "outputs/safe_local_controls_v1/report.json").read_text(encoding="utf-8"))
    paths = [root / row["path"] for row in safe_report["rows"]]
    labels = np.asarray([row["label"] for row in safe_report["rows"]])

    def scores(quality: int | None) -> np.ndarray:
        values: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(paths), 8):
                tensors: list[torch.Tensor] = []
                for path in paths[start : start + 8]:
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                        if quality is not None:
                            buffer = io.BytesIO()
                            image.save(buffer, "JPEG", quality=quality)
                            buffer.seek(0)
                            image = Image.open(buffer).convert("RGB")
                        tensors.append(transform(image))
                logit = model(torch.stack(tensors)).reshape(-1) / float(config["temperature"])
                values.append((1.0 - torch.sigmoid(logit)).numpy())
                print(f"CLIP forensic {min(start + 8, len(paths))}/{len(paths)} q={quality}", flush=True)
        return np.concatenate(values)

    original = scores(None)
    jpeg75 = scores(75)
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
    output = root / "outputs/wkaandemir_local_controls_v1/report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
