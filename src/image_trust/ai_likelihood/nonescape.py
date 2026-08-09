"""Isolated Nonescape Mini pixel-detector signal.

The model layout is compatible with the Apache-2.0 Nonescape Mini release
(``e3ntity/nonescape``).  Only the published EfficientNet-v2-S architecture
and released safetensors checkpoint are used.  Each score runs in a bounded
CPU child process so its model memory is released before the next detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "weights" / "nonescape" / "nonescape-mini-v0.safetensors"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "models" / "ai_likelihood_nonescape_mini_v1.json"
NONESCAPE_MINI_CHECKPOINT_SHA256 = "7a0d0740c813ce199bc32ed16a5f4f4915895c4c9fdee0a98bdbeedd4f3631fd"
NONESCAPE_MINI_HIGH_CONFIDENCE_THRESHOLD = 0.9260923266410828
NONESCAPE_MINI_WORKER_TIMEOUT_SECONDS = 45


class NonescapeMiniUnavailableError(RuntimeError):
    """The optional Nonescape Mini worker could not produce an audited score."""


@dataclass(frozen=True)
class NonescapeMiniScore:
    """Raw class-1 probability from the released detector."""

    score: float
    preprocessing: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_nonescape_mini_isolated(
    input_path: Path,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    timeout_seconds: int = NONESCAPE_MINI_WORKER_TIMEOUT_SECONDS,
) -> NonescapeMiniScore:
    """Score one local image with a short-lived, hash-checked CPU worker."""

    if not checkpoint_path.is_file():
        raise NonescapeMiniUnavailableError("nonescape_mini_checkpoint_not_available")
    if not input_path.is_file():
        raise NonescapeMiniUnavailableError("nonescape_mini_input_not_available")
    if _sha256(checkpoint_path) != NONESCAPE_MINI_CHECKPOINT_SHA256:
        raise NonescapeMiniUnavailableError("nonescape_mini_checkpoint_hash_mismatch")
    with tempfile.TemporaryDirectory(prefix="demirror-nonescape-mini-") as temporary_directory:
        output_path = Path(temporary_directory) / "result.json"
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "NONESCAPE_MINI_CPU_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "image_trust.ai_likelihood.nonescape",
            "--worker",
            "--input",
            str(input_path),
            "--checkpoint",
            str(checkpoint_path),
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
            raise NonescapeMiniUnavailableError("nonescape_mini_worker_timed_out") from error
        if completed.returncode != 0 or not output_path.is_file():
            raise NonescapeMiniUnavailableError("nonescape_mini_worker_failed")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            score = float(raw["score"])
            preprocessing = str(raw["preprocessing"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise NonescapeMiniUnavailableError("nonescape_mini_worker_result_invalid") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise NonescapeMiniUnavailableError("nonescape_mini_worker_score_out_of_range")
    return NonescapeMiniScore(score=score, preprocessing=preprocessing)


def _worker(input_path: Path, checkpoint_path: Path, output_path: Path) -> None:
    """Load the public Mini architecture without importing the vendor checkout."""

    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from torch import nn
    import torchvision.models as models
    import torchvision.transforms.v2 as transforms

    threads = int(os.environ.get("NONESCAPE_MINI_CPU_THREADS", "2"))
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)

    class _NonescapeMini(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = models.efficientnet_v2_s(
                weights=None,
                num_classes=1024,
                dropout=0.2,
            )
            self.head = nn.Linear(1024, 2)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            embedding = self.backbone(values)
            return torch.softmax(self.head(embedding), dim=-1)

    model = _NonescapeMini()
    state = load_file(str(checkpoint_path), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.JPEG(quality=100),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(input_path) as source:
        tensor = transform(source.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        score = float(model(tensor)[0, 1].item())
    output_path.write_text(
        json.dumps(
            {
                "score": score,
                "preprocessing": "resize_256_center_crop_224_jpeg100_imagenet_normalization",
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker or args.input is None or args.checkpoint is None or args.output is None:
        parser.error("worker mode requires --input, --checkpoint, and --output")
    _worker(args.input, args.checkpoint, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in an isolated process
    raise SystemExit(main())
