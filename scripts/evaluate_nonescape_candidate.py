"""Score the upstream Nonescape Mini candidate on fixed Projective Geometry slices.

This is a candidate-screen harness, not product inference.  It keeps the
upstream source unmodified, runs one bounded CPU worker, and emits raw class-1
scores only.  A separate calibration cohort must establish score direction and
threshold before a held-out cohort is interpreted.
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
from sklearn.metrics import roc_auc_score


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


def _iter_samples(
    data_root: Path, generators: Iterable[str], image_ids: Iterable[int]
) -> Iterable[tuple[str, str, int, Path, int]]:
    for generator in generators:
        for scene in DEFAULT_SCENES:
            root = data_root / f"Recent_{generator}_{scene}" / f"Recent_{generator}_{scene}" / "test"
            if not root.is_dir():
                continue
            for image_id in image_ids:
                for path, label in _sample_paths(data_root, generator, scene, image_id):
                    yield generator, scene.lower(), image_id, path, label


def _input_transform_label(jpeg_quality: int | None, webp_quality: int | None, resize_scale: float) -> str:
    parts: list[str] = []
    if resize_scale != 1.0:
        parts.append(f"lanczos_scale={resize_scale:g}")
    if jpeg_quality is not None:
        parts.append(f"jpeg_reencode_quality={jpeg_quality}")
    elif webp_quality is not None:
        parts.append(f"webp_reencode_quality={webp_quality}")
    return "+".join(parts) if parts else "original_decode"


def _open_variant(
    path: Path,
    jpeg_quality: int | None,
    webp_quality: int | None,
    resize_scale: float,
) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if resize_scale != 1.0:
        image = image.resize(
            (max(1, round(image.width * resize_scale)), max(1, round(image.height * resize_scale))),
            Image.Resampling.LANCZOS,
        )
    if jpeg_quality is None and webp_quality is None:
        return image
    buffer = io.BytesIO()
    if jpeg_quality is not None:
        image.save(buffer, format="JPEG", quality=jpeg_quality)
    else:
        image.save(buffer, format="WEBP", quality=webp_quality, method=6)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        return recompressed.convert("RGB")


def _load_model(source_dir: Path, weights_path: Path) -> tuple[torch.nn.Module, Any]:
    """Load the published safetensors weights without changing upstream code."""

    sys.path.insert(0, str(source_dir.resolve()))
    from nonescape import NonescapeClassifierMini, preprocess_image  # type: ignore[import-not-found]

    return NonescapeClassifierMini.from_pretrained(str(weights_path)).eval(), preprocess_image


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["class_1_probability"]) for row in rows]
    return {
        "count": len(rows),
        "real_count": labels.count(0),
        "generated_count": labels.count(1),
        "roc_auc_class_1_probability": float(roc_auc_score(labels, scores)),
        "mean_real_class_1_probability": float(
            sum(score for score, label in zip(scores, labels) if label == 0) / labels.count(0)
        ),
        "mean_generated_class_1_probability": float(
            sum(score for score, label in zip(scores, labels) if label == 1) / labels.count(1)
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Nonescape source directory does not exist: {args.source_dir}")
    if not args.weights_path.is_file():
        raise FileNotFoundError(f"Nonescape weights do not exist: {args.weights_path}")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    model, preprocess_image = _load_model(args.source_dir, args.weights_path)
    generators = _parse_generators(args.generators)
    image_ids = _parse_ids(args.ids)
    samples = list(_iter_samples(args.data_root, generators, image_ids))
    if not samples:
        raise ValueError("No requested generator/scene test folders were found")
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, (generator, scene, image_id, path, label) in enumerate(samples, start=1):
            if not path.is_file():
                raise FileNotFoundError(f"Required image is missing: {path}")
            image = _open_variant(path, args.jpeg_quality, args.webp_quality, args.resize_scale)
            probability = float(model(preprocess_image(image).unsqueeze(0))[0, 1].item())
            rows.append(
                {
                    "generator": generator,
                    "scene": scene,
                    "id": image_id,
                    "label": label,
                    "input_sha256": _sha256(path),
                    "class_1_probability": probability,
                }
            )
            print(
                f"scored={index}/{len(samples)} generator={generator} scene={scene} id={image_id} label={label}",
                flush=True,
            )
    report = {
        "schema_version": "nonescape-candidate-screen-v1",
        "purpose": "Uncalibrated external-detector candidate screen; it does not alter product decisions.",
        "model": {
            "repository": "https://github.com/e3ntity/nonescape",
            "source_dir": str(args.source_dir),
            "weights_sha256": _sha256(args.weights_path),
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
            "upstream_class_mapping": "not published; class-1 direction must be confirmed only on calibration data",
        },
        "generators": generators,
        "sample_ids": image_ids,
        "input_transform": _input_transform_label(args.jpeg_quality, args.webp_quality, args.resize_scale),
        "evaluation": _summary(rows),
        "by_generator": {
            generator: _summary([row for row in rows if row["generator"] == generator])
            for generator in generators
            if any(row["generator"] == generator for row in rows)
        },
        "rows": rows,
        "limitations": [
            "This harness reports raw upstream class-1 scores; it does not choose a threshold.",
            "A threshold and class direction must be frozen from an independent calibration split before any held-out decision metric is reported.",
            "No candidate score is integrated into the product by this script.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["evaluation"], ensure_ascii=False, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/p3_aigc_v2/extracted"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/vendor/nonescape/python"))
    parser.add_argument("--weights-path", type=Path, default=Path("weights/nonescape/nonescape-mini-v0.safetensors"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nonescape_candidate_screen_v1"))
    parser.add_argument("--ids", default="426:500")
    parser.add_argument("--generators", default="SDXL")
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--webp-quality", type=int, default=None)
    parser.add_argument("--resize-scale", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.jpeg_quality is not None and not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be within [1, 100]")
    if args.webp_quality is not None and not 1 <= args.webp_quality <= 100:
        parser.error("--webp-quality must be within [1, 100]")
    if args.jpeg_quality is not None and args.webp_quality is not None:
        parser.error("Only one output codec may be selected")
    if not 0.1 <= args.resize_scale <= 4.0:
        parser.error("--resize-scale must be within [0.1, 4.0]")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
