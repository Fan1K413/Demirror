"""Isolated, audited ensemble of complementary AI pixel signals.

This module intentionally reports only a *high-confidence* AI indication from
the official Dual Data Alignment (DDA) checkpoint.  Scores below the
pre-registered threshold remain indeterminate; they are never interpreted as
evidence that an image is camera-made.

Each checkpoint runs in a short-lived worker process instead of remaining in
the web server.  This keeps peak memory isolated and guarantees that it is
released after every local analysis job.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_trust.ai_likelihood.contracts import AiLikelihoodResult, AiSignal
from image_trust.ai_likelihood.forensic_clip import (
    DEFAULT_AUDIT_PATH as DEFAULT_FORENSIC_CLIP_AUDIT_PATH,
    DEFAULT_MODEL_ROOT as DEFAULT_FORENSIC_CLIP_MODEL_ROOT,
    FORENSIC_CLIP_HIGH_CONFIDENCE_THRESHOLD,
    FORENSIC_CLIP_LIMITED_REVIEW_THRESHOLD,
    ForensicClipUnavailableError,
    score_forensic_clip_isolated,
)
from image_trust.ai_likelihood.safe import (
    DEFAULT_AUDIT_PATH as DEFAULT_SAFE_AUDIT_PATH,
    DEFAULT_CHECKPOINT_PATH as DEFAULT_SAFE_CHECKPOINT_PATH,
    SAFE_HIGH_CONFIDENCE_THRESHOLD,
    SafeUnavailableError,
    score_safe_isolated,
)
from image_trust.provenance.contracts import C2paRecord, C2paSignatureValidationStatus


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "weights" / "dda-v1" / "DDA_ckpt.pth"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "models" / "ai_likelihood_dda_v1.json"
DDA_HIGH_CONFIDENCE_THRESHOLD = 0.94
DDA_WORKER_TIMEOUT_SECONDS = 150
_EXPLICIT_AI_SOURCE_TYPES = {
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
}


class DdaUnavailableError(RuntimeError):
    """The optional local DDA runtime could not produce a score."""


@dataclass(frozen=True)
class DdaScore:
    """Raw official DDA score, not a population AI-generation probability."""

    score: float
    preprocessing: str


def assess_high_confidence_ai(
    input_path: Path,
    c2pa_record: C2paRecord,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    safe_checkpoint_path: Path = DEFAULT_SAFE_CHECKPOINT_PATH,
    safe_audit_path: Path = DEFAULT_SAFE_AUDIT_PATH,
    forensic_clip_model_root: Path = DEFAULT_FORENSIC_CLIP_MODEL_ROOT,
    forensic_clip_audit_path: Path = DEFAULT_FORENSIC_CLIP_AUDIT_PATH,
) -> AiLikelihoodResult:
    """Return verified provenance or complementary high-confidence pixel signals."""

    provenance = _provenance_signal(c2pa_record)
    if provenance.status == "available":
        return AiLikelihoodResult(
            status="available",
            probability=0.995,
            risk_band="high",
            reliability=1.0,
            reliability_label="high",
            target_definition="Verified local C2PA declares trained algorithmic media.",
            model_version="verified-c2pa-policy-v1",
            decision_threshold=0.5,
            signals=[provenance],
            limitations=["c2pa_verified_ai_declaration_is_a_provenance_claim_not_pixel_forensics"],
        )
    signals = [provenance]
    limitations: list[str] = []
    evaluation: dict[str, object] = {}
    available_scores: list[tuple[str, float, float]] = []
    primary_threshold: float | None = None
    limited_ai_signal = False

    try:
        dda_audit = _load_audit(audit_path, DDA_HIGH_CONFIDENCE_THRESHOLD)
        dda_threshold = float(dda_audit["high_confidence_threshold"])
        primary_threshold = dda_threshold
        limitations.extend(dda_audit.get("limitations", []))
        evaluation["dda"] = dict(dda_audit.get("evaluation", {}))
        scored = score_dda_isolated(input_path, checkpoint_path=checkpoint_path)
        available_scores.append(("dda", scored.score, dda_threshold))
        signals.append(
            AiSignal(
                name="dda_pixel_detector",
                status="available",
                value=scored.score,
                interpretation=(
                    "The raw DDA score crosses its registered high-confidence threshold."
                    if scored.score >= dda_threshold
                    else "The raw DDA score is below its registered threshold; this is indeterminate, not camera evidence."
                ),
                details={
                    "high_confidence_threshold": dda_threshold,
                    "preprocessing": scored.preprocessing,
                    "checkpoint_sha256": dda_audit.get("checkpoint_sha256"),
                },
                limitations=list(dda_audit.get("limitations", [])),
            )
        )
        if scored.score < dda_threshold:
            limitations.append("dda_no_high_confidence_pixel_signal_is_not_camera_evidence")
    except (OSError, ValueError, KeyError, TypeError, DdaUnavailableError) as error:
        code = str(error) if isinstance(error, DdaUnavailableError) else f"dda_audit_record_unavailable:{type(error).__name__}"
        limitations.append(code)
        signals.append(
            AiSignal(
                name="dda_pixel_detector",
                status="unavailable",
                interpretation="The DDA pixel detector did not produce a score.",
                limitations=[code],
            )
        )

    try:
        safe_audit = _load_audit(safe_audit_path, SAFE_HIGH_CONFIDENCE_THRESHOLD)
        safe_threshold = float(safe_audit["high_confidence_threshold"])
        limitations.extend(safe_audit.get("limitations", []))
        evaluation["safe"] = dict(safe_audit.get("evaluation", {}))
        safe_scored = score_safe_isolated(input_path, checkpoint_path=safe_checkpoint_path)
        available_scores.append(("safe", safe_scored.score, safe_threshold))
        if primary_threshold is None:
            primary_threshold = safe_threshold
        signals.append(
            AiSignal(
                name="safe_pixel_detector",
                status="available",
                value=safe_scored.score,
                interpretation=(
                    "The SAFE wavelet score crosses its scoped high-confidence threshold."
                    if safe_scored.score >= safe_threshold
                    else "The SAFE wavelet score is below its scoped threshold; this is indeterminate, not camera evidence."
                ),
                details={
                    "high_confidence_threshold": safe_threshold,
                    "preprocessing": safe_scored.preprocessing,
                    "checkpoint_sha256": safe_audit.get("checkpoint_sha256"),
                    "scope": "lossless_or_unmodified_uploads",
                },
                limitations=list(safe_audit.get("limitations", [])),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, SafeUnavailableError) as error:
        code = str(error) if isinstance(error, SafeUnavailableError) else f"safe_audit_record_unavailable:{type(error).__name__}"
        limitations.append(code)
        signals.append(
            AiSignal(
                name="safe_pixel_detector",
                status="unavailable",
                interpretation="The SAFE pixel detector did not produce a score.",
                limitations=[code],
            )
        )

    try:
        forensic_audit = _load_audit(
            forensic_clip_audit_path,
            FORENSIC_CLIP_HIGH_CONFIDENCE_THRESHOLD,
        )
        forensic_threshold = float(forensic_audit["high_confidence_threshold"])
        forensic_limited_threshold = float(forensic_audit["limited_review_threshold"])
        if forensic_limited_threshold >= forensic_threshold:
            raise ValueError("forensic_clip_limited_threshold_not_below_high_threshold")
        limitations.extend(forensic_audit.get("limitations", []))
        evaluation["forensic_clip"] = dict(forensic_audit.get("evaluation", {}))
        forensic_scored = score_forensic_clip_isolated(
            input_path,
            model_root=forensic_clip_model_root,
        )
        available_scores.append(("forensic_clip", forensic_scored.score, forensic_threshold))
        limited_ai_signal = forensic_scored.score >= forensic_limited_threshold
        if primary_threshold is None:
            primary_threshold = forensic_threshold
        signals.append(
            AiSignal(
                name="forensic_clip_detector",
                status="available",
                value=forensic_scored.score,
                interpretation=(
                    "The compression-stable forensic score crosses its registered high-confidence threshold."
                    if forensic_scored.score >= forensic_threshold
                    else (
                        "The compression-stable forensic score crosses its recall-oriented limited-review threshold."
                        if forensic_scored.score >= forensic_limited_threshold
                        else "The compression-stable forensic score is below its thresholds; this is indeterminate, not camera evidence."
                    )
                ),
                details={
                    "high_confidence_threshold": forensic_threshold,
                    "limited_review_threshold": forensic_limited_threshold,
                    "preprocessing": forensic_scored.preprocessing,
                    "checkpoint_sha256": forensic_audit.get("checkpoint_sha256"),
                    "config_sha256": forensic_audit.get("config_sha256"),
                    "scope": "low_recall_jpeg_stable_complement",
                },
                limitations=list(forensic_audit.get("limitations", [])),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, ForensicClipUnavailableError) as error:
        code = (
            str(error)
            if isinstance(error, ForensicClipUnavailableError)
            else f"forensic_clip_audit_record_unavailable:{type(error).__name__}"
        )
        limitations.append(code)
        signals.append(
            AiSignal(
                name="forensic_clip_detector",
                status="unavailable",
                interpretation="The compression-stable forensic detector did not produce a score.",
                limitations=[code],
            )
        )

    if not available_scores:
        return AiLikelihoodResult(
            status="unavailable",
            target_definition="No configured local high-confidence pixel detector produced a score.",
            signals=signals,
            limitations=sorted(set(limitations)),
        )
    high_confidence = any(score >= threshold for _, score, threshold in available_scores)
    if not high_confidence:
        limitations.append("no_high_confidence_pixel_signal_is_not_camera_evidence")
    return AiLikelihoodResult(
        status="available",
        # The raw DDA output is deliberately kept in the signal rather than
        # being presented as a calibrated real-world probability.
        probability=None,
        risk_band="high" if high_confidence else ("medium" if limited_ai_signal else "unknown"),
        reliability=0.85 if high_confidence else (0.65 if limited_ai_signal else 0.5),
        reliability_label="high" if high_confidence else "limited",
        target_definition="Complementary registered high-confidence DDA, SAFE, and compression-stable forensic pixel signals.",
        model_version="dda-safe-forensic-clip-high-confidence-v1",
        decision_threshold=primary_threshold,
        signals=signals,
        evaluation=evaluation,
        limitations=sorted(set(limitations)),
    )


def score_dda_isolated(
    input_path: Path,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    timeout_seconds: int = DDA_WORKER_TIMEOUT_SECONDS,
) -> DdaScore:
    """Score one image in an ephemeral CPU worker and return its raw output."""

    if not checkpoint_path.is_file():
        raise DdaUnavailableError("dda_checkpoint_not_available")
    if not input_path.is_file():
        raise DdaUnavailableError("dda_input_not_available")
    # Reject undersized inputs before constructing the large DINOv2 model.
    try:
        from PIL import Image

        with Image.open(input_path) as source:
            if min(source.size) < 336:
                raise DdaUnavailableError("dda_input_too_small")
    except DdaUnavailableError:
        raise
    except (OSError, ValueError) as error:
        raise DdaUnavailableError("dda_input_not_decodable") from error
    with tempfile.TemporaryDirectory(prefix="demirror-dda-") as temporary_directory:
        output_path = Path(temporary_directory) / "result.json"
        environment = os.environ.copy()
        environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "DDA_CPU_THREADS": "4"})
        command = [
            sys.executable,
            "-m",
            "image_trust.ai_likelihood.dda",
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
            raise DdaUnavailableError("dda_worker_timed_out") from error
        if completed.returncode != 0 or not output_path.is_file():
            raise DdaUnavailableError("dda_worker_failed")
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            score = float(raw["score"])
            preprocessing = str(raw["preprocessing"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise DdaUnavailableError("dda_worker_result_invalid") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise DdaUnavailableError("dda_worker_score_out_of_range")
    return DdaScore(score=score, preprocessing=preprocessing)


def _load_audit(path: Path, expected_threshold: float) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dda_audit_not_an_object")
    if float(raw.get("high_confidence_threshold")) != expected_threshold:
        raise ValueError("pixel_detector_audit_threshold_mismatch")
    return raw


def _provenance_signal(record: C2paRecord) -> AiSignal:
    source_types = {_source_type_slug(value) for value in record.declared_digital_source_types}
    if (
        record.signature_validation_status is C2paSignatureValidationStatus.VALID
        and source_types & _EXPLICIT_AI_SOURCE_TYPES
    ):
        return AiSignal(
            name="verified_c2pa",
            status="available",
            value=1.0,
            interpretation="A locally verified C2PA manifest explicitly declares trained algorithmic media.",
            details={"digital_source_types": sorted(source_types)},
            limitations=list(record.limitations),
        )
    return AiSignal(
        name="verified_c2pa",
        status="neutral",
        interpretation="No locally verified explicit AI-origin declaration was observed; absence is not evidence of a camera photo.",
        details={"manifest_present": record.manifest_present, "signature_status": record.signature_validation_status.value},
        limitations=list(record.limitations),
    )


def _source_type_slug(value: str) -> str:
    candidate = value.strip().rstrip("/")
    candidate = candidate.rsplit("/", 1)[-1]
    candidate = candidate.rsplit(":", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", candidate.lower())


def _worker(input_path: Path, checkpoint_path: Path, output_path: Path) -> None:
    """Load the official architecture and checkpoint without a network request."""

    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms

    threads = int(os.environ.get("DDA_CPU_THREADS", "4"))
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)
    source_path = _local_dinov2_source(torch)
    model = _DdaModel(torch, nn, source_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("dda_checkpoint_model_state_missing")
    model.load_state_dict(state, strict=True)
    del checkpoint
    gc.collect()
    model.eval()
    with Image.open(input_path) as source:
        image = source.convert("RGB")
    if min(image.size) < 336:
        raise ValueError("dda_input_too_small")
    transform = transforms.Compose(
        [
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    with torch.inference_mode():
        score = float(torch.sigmoid(model(transform(image).unsqueeze(0))).item())
    output_path.write_text(
        json.dumps({"score": score, "preprocessing": "center_crop_336_clip_normalization"}),
        encoding="utf-8",
    )


def _local_dinov2_source(torch_module: Any) -> Path:
    hub_directory = Path(torch_module.hub.get_dir())
    candidates = sorted(hub_directory.glob("facebookresearch_dinov2_*"))
    for candidate in reversed(candidates):
        if (candidate / "hubconf.py").is_file():
            return candidate
    raise DdaUnavailableError("dda_dinov2_source_not_initialized")


class _DdaModel:  # constructed lazily so importing the web server stays light
    def __new__(cls, torch_module: Any, nn_module: Any, source_path: Path) -> Any:
        class LoRALayer(nn_module.Module):
            def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 1.0) -> None:
                super().__init__()
                self.alpha = alpha
                self.rank = rank
                self.lora_A = nn_module.Parameter(torch_module.zeros((rank, in_dim)))
                self.lora_B = nn_module.Parameter(torch_module.zeros((out_dim, rank)))
                nn_module.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn_module.init.zeros_(self.lora_B)

            def forward(self, value: Any) -> Any:
                low_rank = torch_module.einsum("...d, rd -> ...r", value, self.lora_A)
                return torch_module.einsum("...r, or -> ...o", low_rank, self.lora_B) * (self.alpha / self.rank)

        class LoRALinear(nn_module.Module):
            def __init__(self, original_layer: Any) -> None:
                super().__init__()
                self.original_layer = original_layer
                for parameter in self.original_layer.parameters():
                    parameter.requires_grad = False
                self.lora = LoRALayer(original_layer.in_features, original_layer.out_features)

            def forward(self, value: Any) -> Any:
                return self.original_layer(value) + self.lora(value)

        class DINOv2Model(nn_module.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch_module.hub.load(
                    str(source_path), "dinov2_vitl14", source="local", pretrained=False
                )
                self.fc = nn_module.Linear(1024, 1)

            def forward(self, value: Any) -> Any:
                features = self.model.forward_features(value)["x_norm_clstoken"]
                return self.fc(features)

        class DINOv2ModelWithLoRA(nn_module.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_model = DINOv2Model()
                targets = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
                for name, module in list(self.base_model.model.named_modules()):
                    if not isinstance(module, nn_module.Linear) or not any(target in name for target in targets):
                        continue
                    parent_name, _, child_name = name.rpartition(".")
                    parent = self.base_model.model.get_submodule(parent_name) if parent_name else self.base_model.model
                    setattr(parent, child_name, LoRALinear(module))

            def forward(self, value: Any) -> Any:
                return self.base_model(value)

        return DINOv2ModelWithLoRA()


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
