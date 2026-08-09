"""Evaluate a fixed CLIP-forensic threshold on held-out SDXL and JPEG75."""

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


TARGET_FPR = 0.08


def main() -> int:
    import timm

    root = Path(__file__).resolve().parents[1]
    model_root = root / "weights/wkaandemir-ai-detector"
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = timm.create_model(config["backbone"], pretrained=False, num_classes=1, img_size=256)
    missing, unexpected = model.load_state_dict(load_file(str(model_root / "model.safetensors")), strict=False)
    if missing or unexpected:
        raise ValueError(f"state mismatch: {missing}, {unexpected}")
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(config["normalization_mean"], config["normalization_std"]),
        ]
    )
    cache = np.load(root / "outputs/deeplsd_geometry_v1/line_cache.npz", allow_pickle=False)
    records = [json.loads(value) for value in cache["records"].tolist()]
    calibration_records = [record for record in records if record["split"] == "calibration"]
    test_records = [record for record in records if record["split"] == "test"]

    def scores(paths: list[Path], quality: int | None) -> np.ndarray:
        values: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(paths), 16):
                tensors: list[torch.Tensor] = []
                for path in paths[start : start + 16]:
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
                print(f"CLIP forensic {min(start + 16, len(paths))}/{len(paths)} q={quality}", flush=True)
        return np.concatenate(values)

    def metrics(
        values: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        mask: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        active = np.ones(len(values), dtype=bool) if mask is None else mask
        y = labels[active]
        score = values[active]
        real = y == 0
        gen = y == 1
        prediction = score >= threshold
        return {
            "count": int(len(y)),
            "roc_auc": float(roc_auc_score(y, score)),
            "true_positive_rate": float(prediction[gen].mean()),
            "false_positive_rate": float(prediction[real].mean()),
            "real_mean": float(score[real].mean()),
            "generated_mean": float(score[gen].mean()),
        }

    calibration_paths = [Path(record["path"]) for record in calibration_records]
    calibration_labels = np.asarray([record["label"] for record in calibration_records])
    calibration_scores = scores(calibration_paths, None)
    threshold = float(
        np.quantile(
            calibration_scores[calibration_labels == 0],
            1.0 - TARGET_FPR,
            method="higher",
        )
    )
    paths = [Path(record["path"]) for record in test_records]
    labels = np.asarray([record["label"] for record in test_records])
    scenes = np.asarray([record["scene"] for record in test_records])
    original = scores(paths, None)
    jpeg75 = scores(paths, 75)
    report = {
        "threshold": threshold,
        "target_calibration_fpr": TARGET_FPR,
        "calibration_pixart": metrics(calibration_scores, calibration_labels, threshold),
        "held_out_sdxl": metrics(original, labels, threshold),
        "held_out_sdxl_jpeg75": metrics(jpeg75, labels, threshold),
        "by_scene": {
            scene: {
                "original": metrics(original, labels, threshold, scenes == scene),
                "jpeg75": metrics(jpeg75, labels, threshold, scenes == scene),
            }
            for scene in ("indoor", "outdoor")
        },
    }
    output = root / "outputs/wkaandemir_cross_generator_v1/report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".npz"),
        calibration_scores=calibration_scores,
        calibration_labels=calibration_labels,
        test_scores=original,
        test_jpeg75_scores=jpeg75,
        test_labels=labels,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
