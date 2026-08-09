"""Isolated high-threshold SAFE signal for lossless generator artifacts.

The small SAFE network is deliberately scoped as a complementary signal.  It
performed strongly on local unmodified ChatGPT controls but failed after JPEG
re-encoding and did not generalize to the registered SDXL/PixArt benchmark.
The worker is short-lived so optional PyTorch/wavelet memory is always freed.
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
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "weights/aigibench-safe/SAFE-main/checkpoint-best.pth"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "models/ai_likelihood_safe_v1.json"
SAFE_HIGH_CONFIDENCE_THRESHOLD = 0.90
SAFE_WORKER_TIMEOUT_SECONDS = 45


class SafeUnavailableError(RuntimeError):
    """The optional local SAFE runtime could not produce a score."""


@dataclass(frozen=True)
class SafeScore:
    score: float
    preprocessing: str


def score_safe_isolated(
    input_path: Path,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    timeout_seconds: int = SAFE_WORKER_TIMEOUT_SECONDS,
) -> SafeScore:
    if not checkpoint_path.is_file():
        raise SafeUnavailableError("safe_checkpoint_not_available")
    if not input_path.is_file():
        raise SafeUnavailableError("safe_input_not_available")
    # Avoid loading the wavelet model when the registered crop is impossible.
    try:
        from PIL import Image

        with Image.open(input_path) as source:
            if min(source.size) < 256:
                raise SafeUnavailableError("safe_input_too_small")
    except SafeUnavailableError:
        raise
    except (OSError, ValueError) as error:
        raise SafeUnavailableError("safe_input_not_decodable") from error
    with tempfile.TemporaryDirectory(prefix="demirror-safe-") as temporary_directory:
        output_path = Path(temporary_directory) / "result.json"
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "SAFE_CPU_THREADS": "4"})
        command = [
            sys.executable,
            "-m",
            "image_trust.ai_likelihood.safe",
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
            raise SafeUnavailableError("safe_worker_timed_out") from error
        if completed.returncode != 0 or not output_path.is_file():
            raise SafeUnavailableError("safe_worker_failed")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            score = float(raw["score"])
            preprocessing = str(raw["preprocessing"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SafeUnavailableError("safe_worker_result_invalid") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise SafeUnavailableError("safe_worker_score_out_of_range")
    return SafeScore(score=score, preprocessing=preprocessing)


def _worker(input_path: Path, checkpoint_path: Path, output_path: Path) -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from PIL import Image
    from pytorch_wavelets import DWTForward

    threads = int(os.environ.get("SAFE_CPU_THREADS", "4"))
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Any = None) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, value: Any) -> Any:
            identity = value
            value = self.relu(self.bn1(self.conv1(value)))
            value = self.relu(self.bn2(self.conv2(value)))
            value = self.bn3(self.conv3(value))
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.relu(value + identity)

    class SafeResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 3)
            self.layer2 = self._make_layer(128, 4, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(512, 2)
            self.dwt = DWTForward(J=1, mode="symmetric", wave="bior1.3")

        def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
            downsample = None
            if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * Bottleneck.expansion),
                )
            layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * Bottleneck.expansion
            layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, value: Any) -> Any:
            _, high = self.dwt(value)
            value = functional.interpolate(
                high[0][:, :, 2],
                size=value.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            value = self.maxpool(self.relu(self.bn1(self.conv1(value))))
            value = self.layer2(self.layer1(value))
            value = self.avgpool(value).flatten(1)
            return self.fc1(value)

    model = SafeResNet()
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("safe_checkpoint_model_state_missing")
    # DWT filters are buffers in this implementation but were constructed on
    # every forward pass in the upstream model, so they are absent upstream.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("dwt.") for name in missing):
        raise ValueError(f"safe_checkpoint_state_mismatch:{missing}:{unexpected}")
    del checkpoint
    gc.collect()
    model.eval()
    with Image.open(input_path) as source:
        image = source.convert("RGB")
    if min(image.size) < 256:
        raise ValueError("safe_input_too_small")
    left = (image.width - 256) // 2
    top = (image.height - 256) // 2
    image = image.crop((left, top, left + 256, top + 256))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        score = float(torch.softmax(model(tensor), dim=1)[0, 1].item())
    output_path.write_text(
        json.dumps({"score": score, "preprocessing": "center_crop_256_rgb_to_tensor_dwt_bior1.3"}),
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


if __name__ == "__main__":
    raise SystemExit(main())
