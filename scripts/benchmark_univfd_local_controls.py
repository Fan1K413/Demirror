"""Probe UniversalFakeDetect on original and JPEG local controls."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    vendor = root / "data/vendor/UniversalFakeDetect"
    sys.path.insert(0, str(vendor.resolve()))
    from models.clip import clip

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    backbone, preprocess = clip.load(str(root / "weights/univfd/ViT-L-14.pt"), device="cpu")
    head = nn.Linear(768, 1)
    head.load_state_dict(torch.load(vendor / "pretrained_weights/fc_weights.pth", map_location="cpu", weights_only=True))
    backbone.eval()
    head.eval()
    safe_report = json.loads((root / "outputs/safe_local_controls_v1/report.json").read_text(encoding="utf-8"))
    paths = [root / row["path"] for row in safe_report["rows"]]
    labels = np.asarray([row["label"] for row in safe_report["rows"]])

    def scores(quality: int | None) -> np.ndarray:
        values: list[float] = []
        with torch.inference_mode():
            for index, path in enumerate(paths):
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    if quality is not None:
                        buffer = io.BytesIO()
                        image.save(buffer, "JPEG", quality=quality)
                        buffer.seek(0)
                        image = Image.open(buffer).convert("RGB")
                    tensor = preprocess(image).unsqueeze(0)
                values.append(float(torch.sigmoid(head(backbone.encode_image(tensor)))[0, 0]))
                if (index + 1) % 5 == 0:
                    print(f"UniFD {index + 1}/{len(paths)} q={quality}", flush=True)
        return np.asarray(values)

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
    output = root / "outputs/univfd_local_controls_v1/report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
