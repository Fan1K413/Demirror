"""Explicit, non-downloading adapters for P1 camera-estimation backends."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from image_trust.camera.contracts import (
    CameraBackendConfig,
    CameraBackendProvenance,
    CameraEstimate,
    CameraEstimateStatus,
    CameraModel,
    CameraUncertainty,
    CropSpec,
    FieldOfViewOrFocal,
    HorizonLine,
    IntrinsicKind,
)
from image_trust.schemas import Point


@dataclass(frozen=True)
class CameraBackendInput:
    """Runtime request passed to a backend; crop metadata is optional context."""

    image_rgb: np.ndarray
    canonical_size: tuple[int, int]
    crop: CropSpec | None = None


class CameraBackend(Protocol):
    backend_id: str

    def estimate(self, request: CameraBackendInput) -> CameraEstimate:
        """Return a measurement or an explicit unavailable/not-applicable result."""


class _UnavailableCameraBackend:
    """Readiness adapter that never downloads packages or weights at import/run."""

    backend_id: str

    def __init__(self, config: CameraBackendConfig) -> None:
        self.config = config
        self.backend_id = config.name
        self._module_available = _module_is_available(config.module_name)
        self._backend_version = _installed_version(config.module_name)
        self._weights_sha256 = _weights_hash(config.weights_path)
        self._readiness = self._readiness_limitations()

    def estimate(self, request: CameraBackendInput) -> CameraEstimate:
        del request
        started = time.perf_counter()
        return self._unavailable_estimate(started)

    def _unavailable_estimate(self, started: float) -> CameraEstimate:
        return CameraEstimate(
            status=CameraEstimateStatus.UNAVAILABLE,
            camera_model=CameraModel.UNKNOWN,
            applicability=0.0,
            coverage=0.0,
            limitations=self._readiness,
            provenance=self._provenance(started, inference_device="not_started"),
        )

    def _readiness_limitations(self) -> list[str]:
        return ["p1_backend_inference_not_implemented", *self._availability_limitations()]

    def _availability_limitations(self) -> list[str]:
        limitations: list[str] = []
        if not self._module_available:
            limitations.append(f"dependency_not_installed:{self.config.module_name}")
        if self.config.weights_path is None:
            limitations.append("weights_path_not_configured")
        elif not Path(self.config.weights_path).is_file():
            limitations.append("weights_file_not_found")
        if self.config.model_commit is None:
            limitations.append("model_commit_not_recorded")
        return sorted(limitations)

    def _provenance(
        self,
        started: float,
        *,
        inference_device: str,
    ) -> CameraBackendProvenance:
        return CameraBackendProvenance(
            backend_id=self.backend_id,
            backend_version=self._backend_version,
            model_commit=self.config.model_commit,
            weights_sha256=self._weights_sha256,
            weights_license=self.config.weights_license,
            inference_device=inference_device,
            requested_inference_device=self.config.inference_device,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )


class PerspectiveFieldsBackend(_UnavailableCameraBackend):
    """Local-weight Perspective Fields adapter with no implicit downloads."""

    backend_id = "perspective_fields"

    def __init__(self, config: CameraBackendConfig) -> None:
        super().__init__(config)
        self._model = None
        self._resolved_device: str | None = None

    def _readiness_limitations(self) -> list[str]:
        limitations = self._availability_limitations()
        if self.config.weights_license is None:
            limitations.append("weights_license_not_recorded")
        if self.config.expected_weights_sha256 is None:
            limitations.append("expected_weights_sha256_not_recorded")
        elif self._weights_sha256 != self.config.expected_weights_sha256.lower():
            limitations.append("weights_sha256_mismatch")
        return sorted(set(limitations))

    def estimate(self, request: CameraBackendInput) -> CameraEstimate:
        started = time.perf_counter()
        if self._readiness:
            return self._unavailable_estimate(started)
        try:
            model, device, cv2 = self._load_model()
            image_bgr = cv2.cvtColor(request.image_rgb, cv2.COLOR_RGB2BGR)
            prediction = model.inference(img_bgr=image_bgr)
            return self._to_camera_estimate(prediction, request, started, device)
        except Exception as exc:
            return CameraEstimate(
                status=CameraEstimateStatus.FAILED,
                camera_model=CameraModel.UNKNOWN,
                applicability=0.0,
                coverage=0.0,
                limitations=[f"perspective_fields_inference_failed:{type(exc).__name__}"],
                provenance=self._provenance(
                    started,
                    inference_device=self._resolved_device or "failed",
                ),
            )

    def _load_model(self):
        if self._model is not None and self._resolved_device is not None:
            import cv2

            return self._model, self._resolved_device, cv2
        import cv2
        import torch
        import perspective2d.perspectivefields as perspectivefields

        device = _resolve_device(self.config.inference_device, torch)
        original_loader = torch.hub.load_state_dict_from_url

        def local_loader(*_args, **_kwargs):
            # PerspectiveFields otherwise fetches the model-zoo URL in __init__.
            return torch.load(self.config.weights_path, map_location="cpu")

        torch.hub.load_state_dict_from_url = local_loader
        try:
            model = perspectivefields.PerspectiveFields(
                self.config.perspective_fields_model_version
            ).to(device).eval()
        finally:
            torch.hub.load_state_dict_from_url = original_loader
        self._model = model
        self._resolved_device = device
        return model, device, cv2

    def _to_camera_estimate(
        self,
        prediction,
        request: CameraBackendInput,
        started: float,
        device: str,
    ) -> CameraEstimate:
        width = float(request.image_rgb.shape[1])
        height = float(request.image_rgb.shape[0])
        roll = math.radians(_scalar(prediction["pred_roll"]))
        pitch = math.radians(_scalar(prediction["pred_pitch"]))
        vfov_deg = _scalar(prediction["pred_general_vfov"])
        relative_cx = _scalar(prediction["pred_rel_cx"])
        relative_cy = _scalar(prediction["pred_rel_cy"])
        return CameraEstimate(
            status=CameraEstimateStatus.OK,
            camera_model=CameraModel.PINHOLE,
            roll=roll,
            pitch=pitch,
            vfov_or_focal=FieldOfViewOrFocal(
                kind=IntrinsicKind.VFOV_DEG,
                value=vfov_deg,
                reference="camera",
            ),
            principal_point=Point(
                x=(relative_cx + 0.5) * width,
                y=(relative_cy + 0.5) * height,
            ),
            horizon=None,
            uncertainty=CameraUncertainty(),
            applicability=1.0,
            coverage=1.0,
            limitations=[
                "perspective_fields_uncentered_model_expected_cropped_input",
                "perspective_fields_native_uncertainty_not_exposed",
                "perspective_fields_horizon_not_emitted_by_backend",
                "perspective_fields_parameters_are_not_source_confidence",
            ],
            provenance=self._provenance(started, inference_device=device),
        )


class GeoCalibBackend(_UnavailableCameraBackend):
    """Local-weight GeoCalib inference adapter with no implicit downloads."""

    backend_id = "geocalib"

    def __init__(self, config: CameraBackendConfig) -> None:
        super().__init__(config)
        self._model = None
        self._resolved_device: str | None = None

    def _readiness_limitations(self) -> list[str]:
        """GeoCalib is implemented, but only with a compliant local checkpoint."""
        limitations = self._availability_limitations()
        if self.config.weights_license is None:
            limitations.append("weights_license_not_recorded")
        if self.config.expected_weights_sha256 is None:
            limitations.append("expected_weights_sha256_not_recorded")
        elif self._weights_sha256 != self.config.expected_weights_sha256.lower():
            limitations.append("weights_sha256_mismatch")
        return sorted(set(limitations))

    def estimate(self, request: CameraBackendInput) -> CameraEstimate:
        started = time.perf_counter()
        if self._readiness:
            return self._unavailable_estimate(started)
        try:
            model, device, torch = self._load_model()
            image_rgb, input_was_downscaled = _resize_geocalib_input(
                request.image_rgb,
                self.config.geocalib_max_input_edge,
            )
            image = torch.from_numpy(image_rgb).permute(2, 0, 1)
            image = image.to(device=device, dtype=torch.float32).div(255.0)
            result = model.calibrate(
                image,
                camera_model=self.config.geocalib_camera_model,
            )
            return self._to_camera_estimate(
                result,
                torch,
                started,
                device,
                output_size=(request.image_rgb.shape[1], request.image_rgb.shape[0]),
                input_was_downscaled=input_was_downscaled,
            )
        except Exception as exc:
            return CameraEstimate(
                status=CameraEstimateStatus.FAILED,
                camera_model=CameraModel.UNKNOWN,
                applicability=0.0,
                coverage=0.0,
                limitations=[f"geocalib_inference_failed:{type(exc).__name__}"],
                provenance=self._provenance(
                    started,
                    inference_device=self._resolved_device or "failed",
                ),
            )

    def _load_model(self):
        if self._model is not None and self._resolved_device is not None:
            import torch

            return self._model, self._resolved_device, torch
        import torch
        from geocalib import GeoCalib

        device = _resolve_device(self.config.inference_device, torch)
        # Passing a path explicitly avoids GeoCalib's built-in URL downloader.
        model = GeoCalib(weights=str(self.config.weights_path)).to(device).eval()
        self._model = model
        self._resolved_device = device
        return model, device, torch

    def _to_camera_estimate(
        self,
        result,
        torch,
        started: float,
        device: str,
        *,
        output_size: tuple[int, int] | None = None,
        input_was_downscaled: bool = False,
    ) -> CameraEstimate:
        camera = result["camera"]
        gravity = result["gravity"]
        model_width = _scalar(camera.size[0, 0])
        model_height = _scalar(camera.size[0, 1])
        width, height = output_size or (round(model_width), round(model_height))
        scale_x = width / model_width
        scale_y = height / model_height
        principal_point = Point(
            x=_scalar(camera.c[0, 0]) * scale_x,
            y=_scalar(camera.c[0, 1]) * scale_y,
        )
        roll = _scalar(gravity.rp[0, 0])
        pitch = _scalar(gravity.rp[0, 1])
        vfov_deg = math.degrees(_scalar(camera.vfov[0]))
        horizon = _geocalib_horizon(camera, gravity, torch)[0]
        dense_diagnostics = _geocalib_dense_diagnostics(result, torch)
        uncertainty = _geocalib_uncertainty(result, width, height, focal_scale=scale_y)
        limitations = [
            "geocalib_horizon_derived_from_camera_and_gravity",
            "geocalib_principal_point_is_assumed_center_not_optimized",
            "geocalib_prediction_confidence_is_not_source_confidence",
            "geocalib_dense_confidence_is_an_optimizer_weight_not_a_probability",
            f"geocalib_camera_model:{self.config.geocalib_camera_model}",
        ]
        if input_was_downscaled:
            limitations.append(
                f"geocalib_input_downscaled_to_max_edge:{self.config.geocalib_max_input_edge}"
            )
        return CameraEstimate(
            status=CameraEstimateStatus.OK,
            camera_model=_geocalib_camera_model(self.config.geocalib_camera_model),
            roll=roll,
            pitch=pitch,
            vfov_or_focal=FieldOfViewOrFocal(
                kind=IntrinsicKind.VFOV_DEG,
                value=vfov_deg,
                reference="camera",
            ),
            principal_point=principal_point,
            horizon=HorizonLine(
                p1=Point(x=0.0, y=_scalar(horizon[0]) * scale_y),
                p2=Point(x=width, y=_scalar(horizon[1]) * scale_y),
            ),
            uncertainty=uncertainty,
            # GeoCalib's dense confidence maps are per-pixel optimisation
            # weights.  They commonly live far below 0.5 and are not a
            # calibrated applicability probability.  A successful finite
            # camera/gravity solution is therefore applicable; non-finite
            # dense fields still reduce the recorded coverage.
            applicability=1.0,
            coverage=dense_diagnostics["geocalib_dense_finite_coverage"],
            backend_diagnostics=dense_diagnostics,
            limitations=limitations,
            provenance=self._provenance(started, inference_device=device),
        )


def resolve_camera_backend(config: CameraBackendConfig) -> CameraBackend:
    """Resolve exactly the requested backend; no cross-model fallback exists."""

    if config.name == "perspective_fields":
        return PerspectiveFieldsBackend(config)
    if config.name == "geocalib":
        return GeoCalibBackend(config)
    raise ValueError(f"Unsupported camera backend '{config.name}'.")


def _installed_version(module_name: str) -> str | None:
    try:
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_is_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _resolve_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested_cuda_unavailable")
    return requested


def _scalar(value) -> float:
    return float(value.detach().cpu().reshape(-1)[0].item())


def _geocalib_dense_diagnostics(result, torch) -> dict[str, float]:
    """Describe GeoCalib's dense optimiser weights without treating them as probability.

    GeoCalib uses these values to weight residuals in its Levenberg--Marquardt
    optimisation.  The package visualises them on a log scale, so a 0.5
    probability-like threshold excludes normal successful calibrations.  Keep
    their distribution for audit while relying on the backend's parameter
    uncertainty for the P1 quality gate.
    """

    up = result.get("up_confidence")
    latitude = result.get("latitude_confidence")
    if up is None or latitude is None:
        return {
            "geocalib_dense_confidence_mean": 0.0,
            "geocalib_dense_confidence_p10": 0.0,
            "geocalib_dense_confidence_p90": 0.0,
            "geocalib_dense_finite_coverage": 0.0,
        }
    confidence = torch.minimum(up.detach(), latitude.detach())
    finite = torch.isfinite(confidence)
    finite_coverage = float(finite.float().mean().cpu().item())
    values = confidence[finite]
    if values.numel() == 0:
        return {
            "geocalib_dense_confidence_mean": 0.0,
            "geocalib_dense_confidence_p10": 0.0,
            "geocalib_dense_confidence_p90": 0.0,
            "geocalib_dense_finite_coverage": finite_coverage,
        }
    values = values.clamp_min(0.0)
    return {
        "geocalib_dense_confidence_mean": float(values.mean().cpu().item()),
        "geocalib_dense_confidence_p10": float(torch.quantile(values, 0.10).cpu().item()),
        "geocalib_dense_confidence_p90": float(torch.quantile(values, 0.90).cpu().item()),
        "geocalib_dense_finite_coverage": finite_coverage,
    }


def _geocalib_horizon(camera, gravity, torch):
    """Compute horizon intersections using GeoCalib's documented projection formula.

    The upstream helper indexes a batched tensor as though it were unbatched.
    Keeping the batch axis here makes the adapter work for GeoCalib's normal
    single-image batch output while retaining the same camera/gravity formula.
    """

    horizon_midpoint = camera.K @ gravity.R @ camera.new_tensor([0.0, 0.0, 1.0])
    midpoint = horizon_midpoint[..., :2] / horizon_midpoint[..., 2:].clamp_min(1e-12)
    tangent = torch.tan(gravity.roll)
    left = midpoint[..., 1] + midpoint[..., 0] * tangent
    right = midpoint[..., 1] - (camera.size[..., 0] - midpoint[..., 0]) * tangent
    return torch.stack([left, right], dim=-1)


def _geocalib_uncertainty(
    result,
    width: float,
    height: float,
    *,
    focal_scale: float = 1.0,
) -> CameraUncertainty:
    roll = _optional_scalar(result.get("roll_uncertainty"))
    pitch = _optional_scalar(result.get("pitch_uncertainty"))
    focal = _optional_scalar(result.get("focal_uncertainty"))
    if focal is not None:
        focal *= focal_scale
    diagonal = math.hypot(width, height)
    normalized = [
        min(1.0, value / math.pi) for value in (roll, pitch) if value is not None
    ]
    if focal is not None:
        normalized.append(min(1.0, focal / diagonal))
    return CameraUncertainty(
        overall=max(normalized) if normalized else None,
        roll_rad=roll,
        pitch_rad=pitch,
        focal_px=focal,
    )


def _optional_scalar(value) -> float | None:
    if value is None:
        return None
    scalar = _scalar(value)
    return scalar if math.isfinite(scalar) and scalar >= 0.0 else None


def _geocalib_camera_model(name: str) -> CameraModel:
    if name == "pinhole":
        return CameraModel.PINHOLE
    if name in {"simple_radial", "radial"}:
        return CameraModel.RADIAL
    if name == "simple_divisional":
        return CameraModel.DIVISIONAL
    raise ValueError(f"Unsupported GeoCalib camera model '{name}'.")


def _weights_hash(path_string: str | None) -> str | None:
    if path_string is None:
        return None
    path = Path(path_string)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resize_geocalib_input(
    image_rgb: np.ndarray,
    max_edge: int,
) -> tuple[np.ndarray, bool]:
    """Bound GeoCalib's full-resolution post-processing memory use.

    GeoCalib resizes its network input internally, but then expands dense fields
    back to the size supplied by the caller.  Passing a many-megapixel photo
    therefore creates large full-resolution tensors despite the small network
    input.  Resize only the backend copy and map its scalar geometry back to
    the original request coordinate space in ``_to_camera_estimate``.
    """

    height, width = image_rgb.shape[:2]
    largest_edge = max(width, height)
    if largest_edge <= max_edge:
        return image_rgb, False
    scale = max_edge / largest_edge
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = Image.fromarray(image_rgb).resize(
        resized_size,
        resample=Image.Resampling.LANCZOS,
    )
    return np.asarray(resized).copy(), True
