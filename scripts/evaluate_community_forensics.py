"""Evaluate the official Community Forensics ViT on isolated SDXL pairs.

The upstream training/evaluation entry points require CUDA and distributed
training.  This harness performs the repository's documented *test*
preprocessing (resize, center crop and ImageNet normalization), loads the
official ``model.safetensors`` directly, and runs one CPU image at a time.
It is an external-detector screen: scores never flow into Demirror's product
pipeline unless this pre-registered held-out evaluation clears its gate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from torchvision import transforms


DEFAULT_SCENES = ("Indoor", "Outdoor")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ids(text: str) -> tuple[int, ...]:
    """Parse an inclusive range (``426:500``) or comma-separated IDs."""

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


def _parse_generators(text: str) -> tuple[str, ...]:
    generators = tuple(part.strip() for part in text.split(",") if part.strip())
    if not generators:
        raise ValueError("At least one generator is required")
    return generators


def _sample_paths(data_root: Path, generator: str, scene: str, image_id: int) -> list[tuple[Path, int]]:
    root = data_root / f"Recent_{generator}_{scene}" / f"Recent_{generator}_{scene}" / "test"
    return [(root / "real" / f"{image_id}.jpg", 0), (root / "gen" / f"{image_id}.jpg", 1)]


def _test_transform(input_size: int) -> transforms.Compose:
    if input_size == 224:
        resize_size = 256
    elif input_size == 384:
        resize_size = 440
    else:
        raise ValueError(f"Unsupported Community Forensics input size: {input_size}")
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_model(source_dir: Path, weights_path: Path, config: dict[str, Any]) -> torch.nn.Module:
    """Instantiate upstream ViT without fetching unrelated ImageNet weights."""

    sys.path.insert(0, str(source_dir.resolve()))
    import models  # type: ignore[import-not-found]
    import timm

    original_create_model = timm.create_model

    def no_pretrained(name: str, pretrained: bool = False, **kwargs: Any) -> torch.nn.Module:
        del pretrained
        return original_create_model(name, pretrained=False, **kwargs)

    timm.create_model = no_pretrained
    try:
        model = models.ViTClassifier(
            model_size=str(config["model_size"]),
            input_size=int(config["input_size"]),
            patch_size=int(config["patch_size"]),
            freeze_backbone=bool(config.get("freeze_backbone", False)),
            device="cpu",
            dtype=torch.float32,
        )
    finally:
        timm.create_model = original_create_model
    model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    return model.eval()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["ai_probability"]) for row in rows]
    predicted_fake = [score >= 0.5 for score in scores]
    true_positive = sum(flag and label == 1 for flag, label in zip(predicted_fake, labels))
    false_positive = sum(flag and label == 0 for flag, label in zip(predicted_fake, labels))
    true_negative = sum(not flag and label == 0 for flag, label in zip(predicted_fake, labels))
    false_negative = sum(not flag and label == 1 for flag, label in zip(predicted_fake, labels))
    return {
        "count": len(rows),
        "roc_auc_ai_probability": float(roc_auc_score(labels, scores)),
        "decision_threshold": 0.5,
        "accuracy": (true_positive + true_negative) / len(rows),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "true_positive_rate": true_positive / (true_positive + false_negative),
        "false_positive_rate": false_positive / (false_positive + true_negative),
    }


def _iter_samples(
    data_root: Path, generators: Iterable[str], image_ids: Iterable[int]
) -> Iterable[tuple[str, str, int, Path, int]]:
    for generator in generators:
        for scene in DEFAULT_SCENES:
            test_dir = data_root / f"Recent_{generator}_{scene}" / f"Recent_{generator}_{scene}" / "test"
            if not test_dir.is_dir():
                continue
            for image_id in image_ids:
                for path, label in _sample_paths(data_root, generator, scene, image_id):
                    yield generator, scene.lower(), image_id, path, label


def _model_input(path: Path, transform: transforms.Compose, jpeg_quality: int | None) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if jpeg_quality is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as recompressed:
            return transform(recompressed.convert("RGB")).unsqueeze(0)
    return transform(image).unsqueeze(0)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Community Forensics source directory does not exist: {args.source_dir}")
    if not args.weights_path.is_file():
        raise FileNotFoundError(f"Community Forensics weights do not exist: {args.weights_path}")
    config = json.loads(args.config_path.read_text(encoding="utf-8"))
    model = _load_model(args.source_dir, args.weights_path, config)
    transform = _test_transform(int(config["input_size"]))
    rows: list[dict[str, Any]] = []
    generators = _parse_generators(args.generators)
    image_ids = _parse_ids(args.ids)
    samples = list(_iter_samples(args.data_root, generators, image_ids))
    if not samples:
        raise ValueError("No requested generator/scene test folders were found")
    with torch.inference_mode():
        for index, (generator, scene, image_id, path, label) in enumerate(samples, start=1):
            if not path.is_file():
                raise FileNotFoundError(f"Required held-out image is missing: {path}")
            inputs = _model_input(path, transform, args.jpeg_quality)
            probability = float(torch.sigmoid(model(inputs)).item())
            rows.append(
                {
                    "generator": generator,
                    "scene": scene,
                    "id": image_id,
                    "label": label,
                    "input_sha256": _sha256(path),
                    "ai_probability": probability,
                }
            )
            print(
                f"scored={index}/{len(samples)} generator={generator} scene={scene} id={image_id} label={label}",
                flush=True,
            )
    report = {
        "schema_version": "community-forensics-sdxl-evaluation-v1",
        "purpose": "Isolated external-detector screen; not a product calibration.",
        "model": {
            "repo": "OwensLab/commfor-model-224",
            "source_dir": str(args.source_dir),
            "weights_sha256": _sha256(args.weights_path),
            "config_sha256": _sha256(args.config_path),
            "input_size": config["input_size"],
            "device": "cpu",
        },
        "generators": generators,
        "sample_ids": image_ids,
        "input_transform": (
            "original_decode" if args.jpeg_quality is None else f"jpeg_reencode_quality={args.jpeg_quality}"
        ),
        "evaluation": _summary(rows),
        "by_generator": {
            generator: _summary([row for row in rows if row["generator"] == generator])
            for generator in generators
            if any(row["generator"] == generator for row in rows)
        },
        "rows": rows,
        "limitations": [
            "SDXL is held out from Demirror's geometry classifier selection but this is still only one generator family.",
            "No score from this external model is integrated or calibrated for the website by this script.",
            "Product integration additionally requires robustness checks after realistic JPEG recompression and an isolated threshold-calibration split.",
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
    parser.add_argument("--source-dir", type=Path, default=Path("data/vendor/community-forensics"))
    parser.add_argument("--weights-path", type=Path, default=Path("weights/community-forensics-224/model.safetensors"))
    parser.add_argument("--config-path", type=Path, default=Path("weights/community-forensics-224/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/community_forensics_sdxl_v1"))
    parser.add_argument("--ids", default="426:500", help="Inclusive range, e.g. 426:500, or comma-separated IDs.")
    parser.add_argument("--generators", default="SDXL", help="Comma-separated dataset generator names, e.g. SDXL,Pixart.")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="Re-encode every input to this JPEG quality before scoring.")
    args = parser.parse_args()
    if args.jpeg_quality is not None and not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be within [1, 100]")
    report = evaluate(args)
    print(json.dumps(report["evaluation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
