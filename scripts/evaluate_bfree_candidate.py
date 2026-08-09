"""Run the released B-Free detector as an isolated, CPU-bounded candidate screen.

This is deliberately an evaluation harness rather than product integration.
It preserves the publisher's five-crop 504px inference and reports raw logits
on a frozen subset of the local paired benchmark.  A report produced here is
evidence for (or against) a later integration gate; its score is never read by
the website.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


DEFAULT_SCENES = ("Indoor", "Outdoor")
DEFAULT_GENERATORS = ("Deepfloyd", "Kandinsky", "Pixart", "SDXL")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ids(text: str) -> tuple[int, ...]:
    """Parse an inclusive range (``426:429``) or a comma-separated list."""

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
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("At least one generator is required")
    unknown = sorted(set(values).difference(DEFAULT_GENERATORS))
    if unknown:
        raise ValueError(f"Unknown local generator folders: {', '.join(unknown)}")
    return values


def _iter_samples(
    data_root: Path, generators: Iterable[str], image_ids: Iterable[int]
) -> Iterable[tuple[str, str, int, Path, int]]:
    """Yield paired local inputs without silently omitting a requested image."""

    for generator in generators:
        for scene in DEFAULT_SCENES:
            root = data_root / f"Recent_{generator}_{scene}" / f"Recent_{generator}_{scene}" / "test"
            if not root.is_dir():
                raise FileNotFoundError(f"Missing requested benchmark folder: {root}")
            for image_id in image_ids:
                for split, label in (("real", 0), ("gen", 1)):
                    path = root / split / f"{image_id}.jpg"
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing requested benchmark image: {path}")
                    yield generator, scene.lower(), image_id, path, label


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["ai_logit"]) for row in rows], dtype=np.float64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Both real and AI labels are required for a candidate screen")
    predicted_fake = scores > 0.0  # upstream's documented default decision.
    true_positive = int(np.sum(predicted_fake & (labels == 1)))
    false_positive = int(np.sum(predicted_fake & (labels == 0)))
    true_negative = int(np.sum(~predicted_fake & (labels == 0)))
    false_negative = int(np.sum(~predicted_fake & (labels == 1)))
    return {
        "count": int(len(rows)),
        "roc_auc_ai_logit": float(roc_auc_score(labels, scores)),
        "upstream_default_logit_threshold": 0.0,
        "accuracy_at_upstream_default": (true_positive + true_negative) / len(rows),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "true_positive_rate": true_positive / (true_positive + false_negative),
        "false_positive_rate": false_positive / (false_positive + true_negative),
    }


def _load_model(source_dir: Path, weights_dir: Path, model_name: str) -> tuple[Any, Any, str]:
    """Load B-Free exactly once; caller must be a short-lived process."""

    import torch
    from torchvision.transforms import Compose

    sys.path.insert(0, str(source_dir.resolve()))
    from main_bfree_single import get_config  # type: ignore[import-not-found]
    from networks import get_network, load_weights  # type: ignore[import-not-found]
    from utils.normalization import get_list_norm  # type: ignore[import-not-found]

    _, model_path, arch, norm_type = get_config(model_name, weights_dir=str(weights_dir))
    model = load_weights(get_network(arch), model_path).to("cpu").eval()
    return model, Compose(get_list_norm(norm_type)), str(model_path)


def _model_input(path: Path, jpeg_quality: int | None) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if jpeg_quality is None:
        return image
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        return recompressed.convert("RGB")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a bounded frozen sample; process exit releases all model memory."""

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"B-Free source directory does not exist: {args.source_dir}")
    config_path = args.weights_dir / args.model_name / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Verified B-Free config does not exist: {config_path}")

    import torch

    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(1)
    generators = _parse_generators(args.generators)
    image_ids = _parse_ids(args.ids)
    samples = list(_iter_samples(args.data_root, generators, image_ids))
    model, transform, model_path_text = _load_model(args.source_dir, args.weights_dir, args.model_name)
    model_path = Path(model_path_text)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, (generator, scene, image_id, path, label) in enumerate(samples, start=1):
            image = _model_input(path, args.jpeg_quality)
            values = transform(image).unsqueeze(0)
            output = model(values).detach().cpu().numpy()
            if output.shape != (1, 1):
                raise ValueError(f"Unexpected B-Free output shape: {output.shape}")
            rows.append(
                {
                    "generator": generator,
                    "scene": scene,
                    "id": image_id,
                    "label": label,
                    "input_sha256": _sha256(path),
                    "ai_logit": float(output[0, 0]),
                }
            )
            print(f"scored={index}/{len(samples)} generator={generator} scene={scene} id={image_id} label={label}", flush=True)

    report = {
        "schema_version": "bfree-candidate-screen-v1",
        "purpose": "External candidate screen only; this report cannot modify product decisions.",
        "model": {
            "upstream_repository": "https://github.com/grip-unina/B-Free",
            "source_dir": str(args.source_dir),
            "model_name": args.model_name,
            "model_sha256": _sha256(model_path),
            "config_sha256": _sha256(config_path),
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
            "preprocessing": "upstream five-crop 504px wrapper with published normalization",
        },
        "generators": generators,
        "sample_ids": image_ids,
        "input_transform": "original_decode" if args.jpeg_quality is None else f"jpeg_reencode_quality={args.jpeg_quality}",
        "evaluation": _summary(rows),
        "by_generator": {
            generator: _summary([row for row in rows if row["generator"] == generator]) for generator in generators
        },
        "rows": rows,
        "limitations": [
            "This is a candidate evaluation, not a threshold-calibration or product-integration result.",
            "The paired local corpus alone cannot establish performance on screenshots, WebP, camera uploads, or new generator families.",
            "A release decision requires a frozen calibration split plus independently held-out transformed-real and cross-generator data.",
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
    parser.add_argument("--source-dir", type=Path, default=Path("data/vendor/B-Free/code"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights/b-free"))
    parser.add_argument("--model-name", default="BFREE_dino2reg4")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bfree_candidate_v1"))
    parser.add_argument("--ids", default="426:427", help="Small frozen pilot range; inclusive, e.g. 426:427.")
    parser.add_argument("--generators", default="Deepfloyd,Kandinsky,Pixart,SDXL")
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.jpeg_quality is not None and not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be within [1, 100]")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be at least 1")
    report = evaluate(args)
    print(json.dumps(report["evaluation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    raise SystemExit(main())
