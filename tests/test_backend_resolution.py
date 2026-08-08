import pytest

from image_trust.geometry.line_backend import BackendUnavailableError, resolve_backend
from image_trust.schemas import LineBackendConfig


def test_auto_backend_records_explicit_opencv_fallback() -> None:
    backend = resolve_backend(LineBackendConfig(name="auto", allow_fallback=True))
    assert backend.backend_id == "opencv_lsd"
    assert backend.fallback_reason == "deeplsd_unavailable"


def test_auto_backend_without_fallback_is_unavailable() -> None:
    with pytest.raises(BackendUnavailableError) as raised:
        resolve_backend(LineBackendConfig(name="auto", allow_fallback=False))
    assert raised.value.backend == "deeplsd"


def test_unknown_backend_is_unavailable() -> None:
    with pytest.raises(BackendUnavailableError) as raised:
        resolve_backend(LineBackendConfig(name="unknown"))
    assert raised.value.backend == "unknown"
