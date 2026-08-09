"""Compression-stable, high-threshold CLIP forensic complement.

The local checkpoint is reconstructed without network access and evaluated in
an ephemeral worker process.  Its raw output is a detector score, not a
population probability of AI generation.  The registered threshold was fixed
on PixArt controls and then tested unchanged on held-out SDXL originals and
JPEG 75 re-encodings.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "weights" / "wkaandemir-ai-detector"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "models" / "ai_likelihood_forensic_clip_v1.json"
FORENSIC_CLIP_HIGH_CONFIDENCE_THRESHOLD = 0.9925177097320557
FORENSIC_CLIP_LIMITED_REVIEW_THRESHOLD = 0.9919478297233582
FORENSIC_CLIP_WORKER_TIMEOUT_SECONDS = 75


class ForensicClipUnavailableError(RuntimeError):
    """The optional local forensic CLIP runtime could not produce a score."""


@dataclass(frozen=True)
class ForensicClipScore:
    """Raw AI-oriented detector score, where larger values are more AI-like."""

    score: float
    preprocessing: str


def score_forensic_clip_isolated(
    input_path: Path,
    *,
    model_root: Path = DEFAULT_MODEL_ROOT,
    timeout_seconds: int = FORENSIC_CLIP_WORKER_TIMEOUT_SECONDS,
) -> ForensicClipScore:
    """Score one image in a short-lived CPU worker."""

    config_path = model_root / "config.json"
    checkpoint_path = model_root / "model.safetensors"
    if not config_path.is_file():
        raise ForensicClipUnavailableError("forensic_clip_config_not_available")
    if not checkpoint_path.is_file():
        raise ForensicClipUnavailableError("forensic_clip_checkpoint_not_available")
    if not input_path.is_file():
        raise ForensicClipUnavailableError("forensic_clip_input_not_available")
    with tempfile.TemporaryDirectory(prefix="demirror-forensic-clip-") as temporary_directory:
        output_path = Path(temporary_directory) / "result.json"
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "FORENSIC_CLIP_CPU_THREADS": "4",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "image_trust.ai_likelihood.forensic_clip",
            "--worker",
            "--input",
            str(input_path),
            "--model-root",
            str(model_root),
            "--output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            raise ForensicClipUnavailableError("forensic_clip_worker_timed_out") from error
        if completed.returncode != 0 or not output_path.is_file():
            raise ForensicClipUnavailableError("forensic_clip_worker_failed")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            score = float(raw["score"])
            preprocessing = str(raw["preprocessing"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ForensicClipUnavailableError("forensic_clip_worker_result_invalid") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ForensicClipUnavailableError("forensic_clip_worker_score_out_of_range")
    return ForensicClipScore(score=score, preprocessing=preprocessing)


def _worker(input_path: Path, model_root: Path, output_path: Path) -> None:
    """Reconstruct the audited local checkpoint without a network request."""

    import timm
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from torchvision import transforms

    threads = int(os.environ.get("FORENSIC_CLIP_CPU_THREADS", "4"))
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    image_size = int(config["image_size"])
    temperature = float(config["temperature"])
    if image_size != 256 or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("forensic_clip_config_not_registered")
    model = timm.create_model(
        str(config["backbone"]),
        pretrained=False,
        num_classes=1,
        img_size=image_size,
    )
    state = load_file(str(model_root / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"forensic_clip_checkpoint_state_mismatch:{missing}:{unexpected}")
    del state
    gc.collect()
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(config["normalization_mean"]),
                std=list(config["normalization_std"]),
            ),
        ]
    )
    with Image.open(input_path) as source:
        tensor = transform(source.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        real_probability = torch.sigmoid(model(tensor).reshape(-1)[0] / temperature)
        score = float((1.0 - real_probability).item())
    output_path.write_text(
        json.dumps(
            {
                "score": score,
                "preprocessing": "resize_256x256_clip_normalization_temperature_0.594889",
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker or args.input is None or args.model_root is None or args.output is None:
        parser.error("worker mode requires --input, --model-root, and --output")
    _worker(args.input, args.model_root, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in an isolated process
    raise SystemExit(main())
