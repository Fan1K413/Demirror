"""Screen Polimi's official EfficientNet-B4 synthetic-image detector safely.

The upstream script extracts 800 random 96-pixel patches and forwards all of
them in one tensor.  This harness preserves the upstream patch generator,
fixed random seed, normalization and top-M aggregation, but forwards patches
in small batches.  It is intentionally a *screening-only* tool: it cannot
modify the web decision or the P0 geometry result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

# Must be set before importing PyTorch so an isolated CPU job cannot oversubscribe
# the workstation while the website is running.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
# Albumentations otherwise performs a version-check network request at import.
# Evaluation must remain offline once its audited assets are present.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import albumentations.pytorch as Ap
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score


DEFAULT_SCENES = ("Indoor", "Outdoor")
UPSTREAM_REPO = "https://github.com/polimi-ispl/synthetic-image-detection"
UPSTREAM_REVISION = "345fdd88c90274d963a48f75619aeaaea08109aa"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ids(text: str) -> tuple[int, ...]:
    if ":" in text:
        start_text, end_text = text.split(":", maxsplit=1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError("ID range end must be at least its start")
        return tuple(range(start, end + 1))
    ids = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not ids:
        raise ValueError("At least one image ID is required")
    return ids


def _sample_paths(data_root: Path, generator: str, scene: str, image_id: int) -> list[tuple[Path, int]]:
    root = data_root / f"Recent_{generator}_{scene}" / f"Recent_{generator}_{scene}" / "test"
    return [(root / "real" / f"{image_id}.jpg", 0), (root / "gen" / f"{image_id}.jpg", 1)]


def _iter_samples(data_root: Path, generator: str, image_ids: Iterable[int]) -> Iterable[tuple[str, int, Path, int]]:
    for scene in DEFAULT_SCENES:
        for image_id in image_ids:
            for path, label in _sample_paths(data_root, generator, scene, image_id):
                yield scene.lower(), image_id, path, label


def _load_model(source_dir: Path, weights_path: Path) -> torch.nn.Module:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Upstream source does not exist: {source_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Official model weights do not exist: {weights_path}")

    sys.path.insert(0, str(source_dir.resolve()))
    try:
        from utils import architectures  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    network_class = getattr(architectures, "EfficientNetB4")
    model = network_class(n_classes=2, pretrained=False).eval()
    state_tmp = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "net" not in state_tmp:
        state = {"net": {f"model.{key}": value for key, value in state_tmp.items()}}
    else:
        state = state_tmp
    model.load_state_dict(state["net"], strict=True)
    return model


def _score_image(model: torch.nn.Module, image_path: Path, top_patch_count: int, batch_size: int) -> float:
    """Replicate upstream scoring while bounding tensor peak memory."""

    random.seed(21)
    np.random.seed(21)
    torch.manual_seed(21)
    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("RGB"))
    if image.shape[0] < 256 or image.shape[1] < 256:
        image = A.SmallestMaxSize(max_size=256, interpolation=1, p=1.0)(image=image)["image"]

    # These are the released preprocessing components and 800-patch sampling
    # count.  Only the forward pass is chunked below.
    cropper = A.RandomCrop(width=96, height=96, p=1.0)
    normalizer = model.get_normalizer()
    transform = A.Compose(
        [
            A.Normalize(mean=normalizer.mean, std=normalizer.std),
            Ap.transforms.ToTensorV2(),
        ]
    )
    patches = [transform(image=cropper(image=image)["image"])["image"] for _ in range(800)]
    logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for offset in range(0, len(patches), batch_size):
            batch = torch.stack(patches[offset : offset + batch_size], dim=0)
            logits.append(model(batch)[:, 1].detach().cpu())
    scores = torch.cat(logits)
    return float(torch.mean(torch.sort(scores)[0][-top_patch_count:]).item())


def _summary(rows: list[dict[str, Any]], threshold: float = 0.0) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["synthetic_score"]) for row in rows]
    predicted_fake = [score > threshold for score in scores]
    tp = sum(flag and label == 1 for flag, label in zip(predicted_fake, labels))
    fp = sum(flag and label == 0 for flag, label in zip(predicted_fake, labels))
    tn = sum(not flag and label == 0 for flag, label in zip(predicted_fake, labels))
    fn = sum(not flag and label == 1 for flag, label in zip(predicted_fake, labels))
    return {
        "count": len(rows),
        "roc_auc_synthetic_score": float(roc_auc_score(labels, scores)),
        "official_decision_threshold": threshold,
        "accuracy": (tp + tn) / len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "true_positive_rate": tp / (tp + fn),
        "false_positive_rate": fp / (fp + tn),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    if not 1 <= args.top_patch_count <= 800:
        raise ValueError("top-patch-count must be within [1, 800]")
    if not 1 <= args.batch_size <= 128:
        raise ValueError("batch-size must be within [1, 128]")

    model = _load_model(args.source_dir, args.weights_path)
    image_ids = _parse_ids(args.ids)
    samples = list(_iter_samples(args.data_root, args.generator, image_ids))
    if not samples:
        raise ValueError("No samples were requested")

    rows: list[dict[str, Any]] = []
    for index, (scene, image_id, path, label) in enumerate(samples, start=1):
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark image is missing: {path}")
        score = _score_image(model, path, args.top_patch_count, args.batch_size)
        rows.append(
            {
                "scene": scene,
                "id": image_id,
                "label": label,
                "input_sha256": _sha256(path),
                "synthetic_score": score,
            }
        )
        print(f"scored={index}/{len(samples)} scene={scene} id={image_id} label={label}", flush=True)

    report = {
        "schema_version": "polimi-synthetic-detection-screen-v1",
        "purpose": "Isolated external-detector screen; never a P0 geometry evaluation or product calibration.",
        "upstream": {"repo": UPSTREAM_REPO, "revision": UPSTREAM_REVISION},
        "model": {
            "architecture": "EfficientNet-B4",
            "weights_sha256": _sha256(args.weights_path),
            "device": "cpu",
            "torch_threads": args.torch_threads,
            "forward_batch_size": args.batch_size,
            "released_patch_sampler": "fixed_seed=21, random_crop=96x96, patches=800",
            "aggregation": f"mean of top {args.top_patch_count} class-1 raw logits",
        },
        "dataset": {
            "source": "https://huggingface.co/datasets/amitabh3/Projective-Geometry",
            "generator": args.generator,
            "sample_ids": image_ids,
            "scenes": DEFAULT_SCENES,
        },
        "evaluation": _summary(rows),
        "by_scene": {
            scene: _summary([row for row in rows if row["scene"] == scene])
            for scene in sorted({str(row["scene"]) for row in rows})
        },
        "rows": rows,
        "limitations": [
            "Scores originate from a pixel-forensic detector, not projective geometry.",
            "The official zero threshold is evaluated unchanged; a pilot result cannot justify product integration.",
            "The script runs in a short-lived CPU process and does not load a model in the website service.",
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
    parser.add_argument("--source-dir", type=Path, default=Path("data/vendor/synthetic-image-detection"))
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("weights/polimi-synthetic-image-detection/synth_vs_real.pth"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/polimi_synthetic_detection_pilot_v1"))
    parser.add_argument("--generator", default="SDXL")
    parser.add_argument("--ids", default="426:430")
    parser.add_argument("--top-patch-count", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report["evaluation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
