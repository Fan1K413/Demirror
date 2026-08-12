"""Isolated, audited ensemble of complementary AI pixel signals.

The registered detectors produce high-confidence or limited-review AI
indications.  Scores below their pre-registered thresholds remain
indeterminate; they are never interpreted as evidence that an image is
camera-made.

Each checkpoint runs in a short-lived worker process instead of remaining in
the web server.  This keeps peak memory isolated and guarantees that it is
released after every local analysis job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from image_trust.ai_likelihood.contracts import AiLikelihoodResult, AiSignal
from image_trust.ai_likelihood.community_forensics import (
    COMMUNITY_FORENSICS_HIGH_CONFIDENCE_THRESHOLD,
    DEFAULT_AUDIT_PATH as DEFAULT_COMMUNITY_FORENSICS_AUDIT_PATH,
    DEFAULT_MODEL_ROOT as DEFAULT_COMMUNITY_FORENSICS_MODEL_ROOT,
    CommunityForensicsUnavailableError,
    score_community_forensics_isolated,
)
from image_trust.ai_likelihood.forensic_clip import (
    DEFAULT_AUDIT_PATH as DEFAULT_FORENSIC_CLIP_AUDIT_PATH,
    DEFAULT_MODEL_ROOT as DEFAULT_FORENSIC_CLIP_MODEL_ROOT,
    FORENSIC_CLIP_HIGH_CONFIDENCE_THRESHOLD,
    FORENSIC_CLIP_LIMITED_REVIEW_THRESHOLD,
    ForensicClipUnavailableError,
    score_forensic_clip_isolated,
)
from image_trust.ai_likelihood.nonescape import (
    DEFAULT_AUDIT_PATH as DEFAULT_NONESCAPE_MINI_AUDIT_PATH,
    DEFAULT_CHECKPOINT_PATH as DEFAULT_NONESCAPE_MINI_CHECKPOINT_PATH,
    NONESCAPE_MINI_HIGH_CONFIDENCE_THRESHOLD,
    NonescapeMiniUnavailableError,
    score_nonescape_mini_isolated,
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


@dataclass(frozen=True)
class _PixelDetectorSpec:
    """One registered detector in the sequential, memory-isolated ensemble."""

    key: str
    signal_name: str
    progress_stage: str
    progress_percent: int
    audit_path: Path
    expected_high_threshold: float
    scorer: Callable[[], Any]
    unavailable_error: type[Exception]
    high_interpretation: str
    static_webp_high_interpretation: str
    low_interpretation: str
    unavailable_interpretation: str
    limited_interpretation: str | None = None
    scope: str | None = None
    audit_detail_keys: tuple[str, ...] = ("checkpoint_sha256",)
    below_threshold_limitation: str | None = None
    reserve_primary_threshold_after_audit: bool = False


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
    community_forensics_model_root: Path = DEFAULT_COMMUNITY_FORENSICS_MODEL_ROOT,
    community_forensics_audit_path: Path = DEFAULT_COMMUNITY_FORENSICS_AUDIT_PATH,
    nonescape_mini_checkpoint_path: Path = DEFAULT_NONESCAPE_MINI_CHECKPOINT_PATH,
    nonescape_mini_audit_path: Path = DEFAULT_NONESCAPE_MINI_AUDIT_PATH,
    progress_callback: Callable[[str, int], None] | None = None,
) -> AiLikelihoodResult:
    """Return verified provenance or complementary high-confidence pixel signals."""

    provenance = _provenance_signal(c2pa_record)
    if provenance.status == "available":
        if progress_callback is not None:
            progress_callback("ai_provenance", 82)
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
    static_webp_limited_review = _is_static_webp(input_path)
    available_scores: list[tuple[str, float, float, bool]] = []
    primary_threshold: float | None = None
    limited_ai_signal = False
    if static_webp_limited_review:
        limitations.append(
            "static_webp_pixel_high_scores_are_limited_review_only_without_format_calibration"
        )

    detector_specs = _pixel_detector_specs(
        input_path=input_path,
        checkpoint_path=checkpoint_path,
        audit_path=audit_path,
        safe_checkpoint_path=safe_checkpoint_path,
        safe_audit_path=safe_audit_path,
        forensic_clip_model_root=forensic_clip_model_root,
        forensic_clip_audit_path=forensic_clip_audit_path,
        community_forensics_model_root=community_forensics_model_root,
        community_forensics_audit_path=community_forensics_audit_path,
        nonescape_mini_checkpoint_path=nonescape_mini_checkpoint_path,
        nonescape_mini_audit_path=nonescape_mini_audit_path,
    )
    for spec in detector_specs:
        if progress_callback is not None:
            progress_callback(spec.progress_stage, spec.progress_percent)
        try:
            audit = _load_audit(spec.audit_path, spec.expected_high_threshold)
            high_threshold = float(audit["high_confidence_threshold"])
            limited_threshold = _limited_review_threshold(audit, spec)
            if primary_threshold is None and spec.reserve_primary_threshold_after_audit:
                primary_threshold = high_threshold
            audit_limitations = list(audit.get("limitations", []))
            limitations.extend(audit_limitations)
            evaluation[spec.key] = dict(audit.get("evaluation", {}))
            scored = spec.scorer()
            eligible_for_high = not static_webp_limited_review
            available_scores.append(
                (spec.key, scored.score, high_threshold, eligible_for_high)
            )
            if primary_threshold is None:
                primary_threshold = high_threshold
            detector_limited = (
                limited_threshold is not None and scored.score >= limited_threshold
            ) or (
                static_webp_limited_review and scored.score >= high_threshold
            )
            limited_ai_signal = limited_ai_signal or detector_limited
            details: dict[str, object] = {
                "high_confidence_threshold": high_threshold,
                "high_confidence_eligible": eligible_for_high,
                "preprocessing": scored.preprocessing,
            }
            if limited_threshold is not None:
                details["limited_review_threshold"] = limited_threshold
            for key in spec.audit_detail_keys:
                details[key] = audit.get(key)
            if spec.scope is not None:
                details["scope"] = spec.scope
            signals.append(
                AiSignal(
                    name=spec.signal_name,
                    status="available",
                    value=scored.score,
                    interpretation=_pixel_interpretation(
                        spec,
                        score=scored.score,
                        high_threshold=high_threshold,
                        limited_threshold=limited_threshold,
                        static_webp=static_webp_limited_review,
                    ),
                    details=details,
                    limitations=audit_limitations,
                )
            )
            if (
                spec.below_threshold_limitation is not None
                and scored.score < high_threshold
            ):
                limitations.append(spec.below_threshold_limitation)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            spec.unavailable_error,
        ) as error:
            code = (
                str(error)
                if isinstance(error, spec.unavailable_error)
                else f"{spec.key}_audit_record_unavailable:{type(error).__name__}"
            )
            limitations.append(code)
            signals.append(
                AiSignal(
                    name=spec.signal_name,
                    status="unavailable",
                    interpretation=spec.unavailable_interpretation,
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
    high_confidence = any(
        score >= threshold and eligible_for_high
        for _, score, threshold, eligible_for_high in available_scores
    )
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
        target_definition="Complementary registered DDA, SAFE, forensic CLIP, Community Forensics, and Nonescape Mini pixel signals.",
        model_version="dda-safe-forensic-clip-community-forensics-nonescape-mini-v1",
        decision_threshold=primary_threshold,
        signals=signals,
        evaluation=evaluation,
        limitations=sorted(set(limitations)),
    )


def _pixel_detector_specs(
    *,
    input_path: Path,
    checkpoint_path: Path,
    audit_path: Path,
    safe_checkpoint_path: Path,
    safe_audit_path: Path,
    forensic_clip_model_root: Path,
    forensic_clip_audit_path: Path,
    community_forensics_model_root: Path,
    community_forensics_audit_path: Path,
    nonescape_mini_checkpoint_path: Path,
    nonescape_mini_audit_path: Path,
) -> tuple[_PixelDetectorSpec, ...]:
    return (
        _PixelDetectorSpec(
            key="dda",
            signal_name="dda_pixel_detector",
            progress_stage="ai_dda",
            progress_percent=30,
            audit_path=audit_path,
            expected_high_threshold=DDA_HIGH_CONFIDENCE_THRESHOLD,
            scorer=lambda: score_dda_isolated(
                input_path,
                checkpoint_path=checkpoint_path,
            ),
            unavailable_error=DdaUnavailableError,
            high_interpretation="The raw DDA score crosses its registered high-confidence threshold.",
            static_webp_high_interpretation="The raw DDA score crosses its registered high-confidence threshold, but static WebP is limited to review strength.",
            low_interpretation="The raw DDA score is below its registered threshold; this is indeterminate, not camera evidence.",
            unavailable_interpretation="The DDA pixel detector did not produce a score.",
            below_threshold_limitation="dda_no_high_confidence_pixel_signal_is_not_camera_evidence",
            reserve_primary_threshold_after_audit=True,
        ),
        _PixelDetectorSpec(
            key="safe",
            signal_name="safe_pixel_detector",
            progress_stage="ai_safe",
            progress_percent=42,
            audit_path=safe_audit_path,
            expected_high_threshold=SAFE_HIGH_CONFIDENCE_THRESHOLD,
            scorer=lambda: score_safe_isolated(
                input_path,
                checkpoint_path=safe_checkpoint_path,
            ),
            unavailable_error=SafeUnavailableError,
            high_interpretation="The SAFE wavelet score crosses its scoped high-confidence threshold.",
            static_webp_high_interpretation="The SAFE wavelet score crosses its scoped high-confidence threshold, but static WebP is limited to review strength.",
            low_interpretation="The SAFE wavelet score is below its scoped threshold; this is indeterminate, not camera evidence.",
            unavailable_interpretation="The SAFE pixel detector did not produce a score.",
            scope="lossless_or_unmodified_uploads",
        ),
        _PixelDetectorSpec(
            key="forensic_clip",
            signal_name="forensic_clip_detector",
            progress_stage="ai_forensic_clip",
            progress_percent=54,
            audit_path=forensic_clip_audit_path,
            expected_high_threshold=FORENSIC_CLIP_HIGH_CONFIDENCE_THRESHOLD,
            scorer=lambda: score_forensic_clip_isolated(
                input_path,
                model_root=forensic_clip_model_root,
            ),
            unavailable_error=ForensicClipUnavailableError,
            high_interpretation="The compression-stable forensic score crosses its registered high-confidence threshold.",
            static_webp_high_interpretation="The forensic score crosses its registered high-confidence threshold, but static WebP is limited to review strength.",
            limited_interpretation="The compression-stable forensic score crosses its recall-oriented limited-review threshold.",
            low_interpretation="The compression-stable forensic score is below its thresholds; this is indeterminate, not camera evidence.",
            unavailable_interpretation="The compression-stable forensic detector did not produce a score.",
            scope="low_recall_jpeg_stable_complement",
            audit_detail_keys=("checkpoint_sha256", "config_sha256"),
        ),
        _PixelDetectorSpec(
            key="community_forensics",
            signal_name="community_forensics_detector",
            progress_stage="ai_community_forensics",
            progress_percent=66,
            audit_path=community_forensics_audit_path,
            expected_high_threshold=COMMUNITY_FORENSICS_HIGH_CONFIDENCE_THRESHOLD,
            scorer=lambda: score_community_forensics_isolated(
                input_path,
                model_root=community_forensics_model_root,
            ),
            unavailable_error=CommunityForensicsUnavailableError,
            high_interpretation="The Community Forensics score crosses its registered high-confidence threshold.",
            static_webp_high_interpretation="The Community Forensics score crosses its registered high-confidence threshold, but static WebP is limited to review strength.",
            limited_interpretation="The Community Forensics score crosses its recall-oriented limited-review threshold.",
            low_interpretation="The Community Forensics score is below its thresholds; this is indeterminate, not camera evidence.",
            unavailable_interpretation="The Community Forensics pixel detector did not produce a score.",
            scope="cross_generator_pixel_detector_with_jpeg85_audit",
            audit_detail_keys=("checkpoint_sha256", "config_sha256"),
        ),
        _PixelDetectorSpec(
            key="nonescape_mini",
            signal_name="nonescape_mini_detector",
            progress_stage="ai_nonescape_mini",
            progress_percent=78,
            audit_path=nonescape_mini_audit_path,
            expected_high_threshold=NONESCAPE_MINI_HIGH_CONFIDENCE_THRESHOLD,
            scorer=lambda: score_nonescape_mini_isolated(
                input_path,
                checkpoint_path=nonescape_mini_checkpoint_path,
            ),
            unavailable_error=NonescapeMiniUnavailableError,
            high_interpretation="The Nonescape Mini score crosses its registered high-confidence threshold.",
            static_webp_high_interpretation="The Nonescape Mini score crosses its registered high-confidence threshold, but static WebP is limited to review strength.",
            low_interpretation="The Nonescape Mini score is below its registered threshold; this is indeterminate, not camera evidence.",
            unavailable_interpretation="The Nonescape Mini detector did not produce a score.",
            scope="strict-zero-new-false-positive-complement",
        ),
    )


def _limited_review_threshold(
    audit: dict[str, Any],
    spec: _PixelDetectorSpec,
) -> float | None:
    if spec.limited_interpretation is None:
        return None
    threshold = float(audit["limited_review_threshold"])
    high_threshold = float(audit["high_confidence_threshold"])
    if threshold >= high_threshold:
        raise ValueError(f"{spec.key}_limited_threshold_not_below_high_threshold")
    return threshold


def _pixel_interpretation(
    spec: _PixelDetectorSpec,
    *,
    score: float,
    high_threshold: float,
    limited_threshold: float | None,
    static_webp: bool,
) -> str:
    if score >= high_threshold:
        return (
            spec.static_webp_high_interpretation
            if static_webp
            else spec.high_interpretation
        )
    if limited_threshold is not None and score >= limited_threshold:
        assert spec.limited_interpretation is not None
        return spec.limited_interpretation
    return spec.low_interpretation


def _is_static_webp(input_path: Path) -> bool:
    """Identify WebP from container magic, rather than trusting the extension."""
    try:
        with input_path.open("rb") as source:
            header = source.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"


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
            if "dda_low_memory_torch_api_unavailable" in completed.stderr:
                raise DdaUnavailableError("dda_low_memory_torch_api_unavailable")
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
    model = _load_dda_model_low_memory(torch, nn, checkpoint_path)
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


def _load_dda_model_low_memory(
    torch_module: Any,
    nn_module: Any,
    checkpoint_path: Path,
    *,
    source_path: Path | None = None,
) -> Any:
    """Build DDA without materializing a second full parameter copy.

    DINOv2-L is constructed on the ``meta`` device, then the registered CPU
    checkpoint is memory-mapped and assigned into that skeleton.  If the
    installed PyTorch cannot provide either operation, the detector fails
    closed instead of falling back to the previous high-memory loader.
    """

    resolved_source = source_path or _local_dinov2_source(torch_module)
    with torch_module.device("meta"):
        model = _DdaModel(torch_module, nn_module, resolved_source)
    try:
        checkpoint = torch_module.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError as error:
        raise ValueError("dda_low_memory_torch_api_unavailable") from error
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("dda_checkpoint_model_state_missing")
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except TypeError as error:
        raise ValueError("dda_low_memory_torch_api_unavailable") from error
    if any(getattr(parameter, "is_meta", False) for parameter in model.parameters()):
        raise ValueError("dda_model_contains_unmaterialized_parameters")
    return model


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
