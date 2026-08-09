from pathlib import Path

import numpy as np
import pytest

from image_trust.geometry.deeplsd import DeepLSDBackend, _resize_for_model
from image_trust.geometry.line_backend import BackendUnavailableError
from image_trust.schemas import LineBackendConfig


def test_resize_for_model_preserves_coordinate_mapping() -> None:
    image = np.zeros((1200, 600), dtype=np.uint8)
    resized, scale_x, scale_y = _resize_for_model(image, max_side=768)
    assert resized.shape == (768, 384)
    assert scale_x == pytest.approx(600 / 384)
    assert scale_y == pytest.approx(1200 / 768)


def test_deeplsd_runtime_rejects_pickle_weights_before_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "untrusted.tar"
    checkpoint.write_bytes(b"not-a-safe-checkpoint")
    with pytest.raises(BackendUnavailableError, match="only .safetensors"):
        DeepLSDBackend(LineBackendConfig(name="deeplsd", deeplsd_weights=str(checkpoint)))
