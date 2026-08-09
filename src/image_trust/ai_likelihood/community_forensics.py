"""Isolated Community Forensics pixel-detector signal.

The upstream project publishes CUDA/DDP training code.  This module recreates
only its documented ViT-Small test inference on CPU, from the separately
audited safetensors checkpoint.  It never contacts a model hub at runtime and
always releases the model in a short-lived worker process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "weights" / "community-forensics-224"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "models" / "ai_likelihood_community_forensics_v1.json"
COMMUNITY_FORENSICS_HIGH_CONFIDENCE_THRESHOLD = 0.8866265416145325
COMMUNITY_FORENSICS_LIMITED_REVIEW_THRESHOLD = 0.5
COMMUNITY_FORENSICS_WORKER_TIMEOUT_SECONDS = 45


class CommunityForensicsUnavailableError(RuntimeError):
    """The optional Community Forensics runtime could not produce a score."""


@dataclass(frozen=True)
class CommunityForensicsScore:
    """Raw detector score where larger values are more AI-like."""

    score: float
    preprocessing: str


def score_community_forensics_isolated(
    input_path: Path,
    *,
    model_root: Path = DEFAULT_MODEL_ROOT,
    timeout_seconds: int = COMMUNITY_FORENSICS_WORKER_TIMEOUT_SECONDS,
) -> CommunityForensicsScore:
    """Score one image with the audited local checkpoint in a CPU worker."""

    config_path = model_root / "config.json"
    checkpoint_path = model_root / "model.safetensors"
    if not config_path.is_file():
        raise CommunityForensicsUnavailableError("community_forensics_config_not_available")
    if not checkpoint_path.is_file():
        raise CommunityForensicsUnavailableError("community_forensics_checkpoint_not_available")
    if not input_path.is_file():
        raise CommunityForensicsUnavailableError("community_forensics_input_not_available")
    with tempfile.TemporaryDirectory(prefix="demirror-community-forensics-") as temporary_directory:
        output_path = Path(temporary_directory) / "result.json"
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "COMMUNITY_FORENSICS_CPU_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "image_trust.ai_likelihood.community_forensics",
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
            raise CommunityForensicsUnavailableError("community_forensics_worker_timed_out") from error
        if completed.returncode != 0 or not output_path.is_file():
            raise CommunityForensicsUnavailableError("community_forensics_worker_failed")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            score = float(raw["score"])
            preprocessing = str(raw["preprocessing"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CommunityForensicsUnavailableError("community_forensics_worker_result_invalid") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise CommunityForensicsUnavailableError("community_forensics_worker_score_out_of_range")
    return CommunityForensicsScore(score=score, preprocessing=preprocessing)


def _worker(input_path: Path, model_root: Path, output_path: Path) -> None:
    """Run the released 224-input ViT without upstream CUDA/DDP code."""

    import torch
    import torch.nn as nn
    import timm
    from PIL import Image
    from safetensors.torch import load_file
    from torchvision import transforms

    threads = int(os.environ.get("COMMUNITY_FORENSICS_CPU_THREADS", "2"))
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    if (
        str(config.get("model_size")) != "small"
        or int(config.get("input_size", 0)) != 224
        or int(config.get("patch_size", 0)) != 16
    ):
        raise ValueError("community_forensics_config_not_registered")

    class _ReleasedViT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vit = timm.create_model(
                "vit_small_patch16_224.augreg_in21k_ft_in1k",
                pretrained=False,
            )
            self.vit.head = nn.Linear(in_features=384, out_features=1, bias=True)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.vit(values)

    model = _ReleasedViT()
    state = load_file(str(model_root / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    with Image.open(input_path) as source:
        tensor = transform(source.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        score = float(torch.sigmoid(model(tensor).reshape(-1)[0]).item())
    output_path.write_text(
        json.dumps(
            {
                "score": score,
                "preprocessing": "resize_shorter_256_center_crop_224_imagenet_normalization",
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
