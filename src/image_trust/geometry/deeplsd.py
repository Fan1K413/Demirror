"""Resource-bounded adapter for the official DeepLSD line detector.

DeepLSD uses a native line-detection extension.  The model is deliberately
loaded and used only in a disposable child process so that native allocations
cannot accumulate in the long-lived local web service.  Runtime inputs accept
only a locally converted safetensors state dict; the original PyTorch pickle
checkpoint is never deserialized while serving an uploaded image.
"""

from __future__ import annotations

import importlib.metadata
import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_trust.geometry.line_backend import BackendUnavailableError, LineExtraction, RawLine
from image_trust.schemas import LineBackendConfig


def _configure_threads(thread_count: int) -> None:
    """Limit native and PyTorch parallelism before importing the model."""

    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)


def _resize_for_model(grayscale: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    height, width = grayscale.shape
    longest = max(height, width)
    if longest <= max_side:
        return grayscale, 1.0, 1.0
    scale = max_side / longest
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(grayscale, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, width / resized_width, height / resized_height


def _detect_lines_worker(
    grayscale: np.ndarray,
    weights_path: str,
    max_side: int,
    thread_count: int,
    result_queue: Any,
) -> None:
    """Load one model, extract one image, serialize only coordinates, and exit."""

    try:
        _configure_threads(thread_count)
        import torch
        from safetensors.torch import load_file
        from deeplsd.models.deeplsd_inference import DeepLSD

        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(1)
        model = DeepLSD(
            {
                "detect_lines": True,
                "line_detection_params": {
                    "merge": False,
                    "filtering": "normal",
                    "grad_thresh": 3,
                    "grad_nfa": True,
                },
            }
        )
        model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
        model.eval()
        model_input, scale_x, scale_y = _resize_for_model(grayscale, max_side)
        normalized = model_input.astype(np.float32) / 255.0
        with torch.inference_mode():
            output = model({"image": torch.from_numpy(normalized[None, None])})
        lines = np.asarray(output["lines"][0], dtype=np.float64).reshape(-1, 2, 2)
        lines[..., 0] *= scale_x
        lines[..., 1] *= scale_y
        result_queue.put({"ok": True, "lines": lines.reshape(-1, 4).tolist()})
    except BaseException as error:  # communicate worker crashes without keeping the model alive
        result_queue.put({"ok": False, "error": f"{type(error).__name__}:{error}"})


class DeepLSDBackend:
    backend_id = "deeplsd"

    def __init__(self, config: LineBackendConfig) -> None:
        self.config = config
        self.fallback_reason: str | None = None
        weight_path = Path(config.deeplsd_weights) if config.deeplsd_weights else None
        if weight_path is None:
            raise BackendUnavailableError(
                self.backend_id,
                "DeepLSD was explicitly requested but deeplsd_weights is not configured. "
                "Set it to the locally converted official .safetensors state dict.",
            )
        if weight_path.suffix.lower() != ".safetensors":
            raise BackendUnavailableError(
                self.backend_id,
                "DeepLSD runtime accepts only .safetensors weights. Convert the trusted official "
                "checkpoint with scripts/convert_deeplsd_checkpoint_to_safetensors.py first.",
            )
        if not weight_path.is_file():
            raise BackendUnavailableError(
                self.backend_id,
                f"DeepLSD weights were not found at: {weight_path}",
            )
        try:
            import deeplsd  # noqa: F401
            import safetensors  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                self.backend_id,
                "DeepLSD safetensors weights are configured but the official deeplsd or safetensors "
                "dependency is unavailable.",
            ) from exc
        self.weights_path = weight_path

    def extract(self, grayscale: np.ndarray) -> LineExtraction:
        if grayscale.ndim != 2 or grayscale.dtype != np.uint8:
            raise ValueError("DeepLSD expects one uint8 grayscale analysis image")
        context = mp.get_context("spawn")
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_detect_lines_worker,
            args=(
                grayscale,
                str(self.weights_path),
                self.config.deeplsd_max_side,
                self.config.deeplsd_threads,
                result_queue,
            ),
        )
        process.start()
        try:
            # Read before join: a dense scene can contain enough serialized
            # line coordinates to fill the multiprocessing pipe.  Joining
            # first would then wait for a worker blocked in Queue.put().
            message = result_queue.get(timeout=self.config.deeplsd_timeout_seconds)
        except queue.Empty as error:
            if process.is_alive():
                process.terminate()
                process.join()
                raise RuntimeError("deeplsd_extraction_timeout") from error
            raise RuntimeError(f"deeplsd_worker_exited_without_result:{process.exitcode}") from error
        finally:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join()
            result_queue.close()
            result_queue.join_thread()
        if not bool(message.get("ok")):
            raise RuntimeError(f"deeplsd_extraction_failed:{message.get('error', 'unknown')}")
        raw_lines = np.asarray(message["lines"], dtype=np.float64).reshape(-1, 4)
        lines = [
            RawLine(
                p1=(float(row[0]), float(row[1])),
                p2=(float(row[2]), float(row[3])),
                quality=1.0,
                backend_features={"deep_score": None},
            )
            for row in raw_lines
        ]
        try:
            version = importlib.metadata.version("deeplsd")
        except importlib.metadata.PackageNotFoundError:
            version = "official-source"
        return LineExtraction(
            backend_id=self.backend_id,
            backend_version=version,
            lines=lines,
            warnings=[
                "deeplsd_cpu_child_process_per_image",
                "deeplsd_quality_is_unavailable_and_not_a_probability",
            ],
        )
