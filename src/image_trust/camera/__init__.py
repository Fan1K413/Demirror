"""P1 camera-parameter measurement contracts and experiment helpers.

This namespace is deliberately separate from the P0 projection-geometry
evidence pipeline.  Its outputs are camera measurements only; they are not
source-authenticity or AI-generation conclusions.
"""

from image_trust.camera.contracts import CameraEstimate, CameraExperimentResult

__all__ = ["CameraEstimate", "CameraExperimentResult"]
