"""OpenCV LineSegmentDetector baseline."""

from __future__ import annotations

import math

import cv2
import numpy as np

from image_trust.geometry.line_backend import LineExtraction, RawLine
from image_trust.schemas import LineBackendConfig


class OpenCVLSDBackend:
    backend_id = "opencv_lsd"

    def __init__(
        self,
        config: LineBackendConfig,
        fallback_reason: str | None = None,
    ) -> None:
        self.config = config
        self.fallback_reason = fallback_reason

    def extract(self, grayscale: np.ndarray) -> LineExtraction:
        refine_options = {
            "none": cv2.LSD_REFINE_NONE,
            "std": cv2.LSD_REFINE_STD,
            "adv": cv2.LSD_REFINE_ADV,
        }
        try:
            refine = refine_options[self.config.opencv_refine.lower()]
        except KeyError as exc:
            raise ValueError(
                "line_backend.opencv_refine must be one of none, std, or adv."
            ) from exc

        detector = cv2.createLineSegmentDetector(refine)
        detections = detector.detect(grayscale)
        line_array = detections[0] if detections else None
        widths = detections[1] if len(detections) > 1 else None
        precisions = detections[2] if len(detections) > 2 else None
        nfas = detections[3] if len(detections) > 3 else None
        lines: list[RawLine] = []
        if line_array is not None:
            for index, raw_line in enumerate(line_array.reshape(-1, 4)):
                x1, y1, x2, y2 = (float(value) for value in raw_line)
                width = _optional_scalar(widths, index)
                precision = _optional_scalar(precisions, index)
                nfa = _optional_scalar(nfas, index)
                lines.append(
                    RawLine(
                        p1=(x1, y1),
                        p2=(x2, y2),
                        quality=_quality(nfa=nfa, precision=precision),
                        backend_features={
                            "width": width,
                            "precision": precision,
                            "nfa": nfa,
                        },
                    )
                )
        warnings: list[str] = [
            "opencv_lsd_quality_is_backend_relative_not_probability"
        ]
        if self.fallback_reason:
            warnings.append(f"fallback_reason:{self.fallback_reason}")
        return LineExtraction(
            backend_id=self.backend_id,
            backend_version=cv2.__version__,
            lines=lines,
            warnings=warnings,
        )


def _optional_scalar(values: np.ndarray | None, index: int) -> float | None:
    if values is None:
        return None
    value = float(np.asarray(values).reshape(-1)[index])
    return value if math.isfinite(value) else None


def _quality(nfa: float | None, precision: float | None) -> float:
    if nfa is not None:
        return float(1.0 - math.exp(-max(nfa, 0.0) / 5.0))
    if precision is not None:
        return float(math.exp(-max(precision, 0.0) / 2.0))
    return 1.0
