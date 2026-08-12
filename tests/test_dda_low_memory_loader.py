from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from image_trust.ai_likelihood import dda


class _DeviceContext:
    def __init__(self, torch: "_FakeTorch", name: str) -> None:
        self.torch = torch
        self.name = name

    def __enter__(self) -> None:
        assert self.name == "meta"
        self.torch.in_meta_context = True

    def __exit__(self, *_args: object) -> None:
        self.torch.in_meta_context = False


class _FakeTorch:
    def __init__(self, checkpoint: object) -> None:
        self.checkpoint = checkpoint
        self.in_meta_context = False
        self.load_call: tuple[Path, dict[str, object]] | None = None

    def device(self, name: str) -> _DeviceContext:
        return _DeviceContext(self, name)

    def load(self, path: Path, **kwargs: object) -> object:
        self.load_call = (path, kwargs)
        return self.checkpoint


class _UnsupportedFakeTorch(_FakeTorch):
    def load(self, path: Path, **kwargs: object) -> object:
        raise TypeError("mmap is not supported")


class _FakeParameter:
    def __init__(self, *, is_meta: bool = False) -> None:
        self.is_meta = is_meta


class _FakeModel:
    def __init__(self, *, remaining_meta: bool = False) -> None:
        self.remaining_meta = remaining_meta
        self.load_call: tuple[object, dict[str, object]] | None = None

    def load_state_dict(self, state: object, **kwargs: object) -> None:
        self.load_call = (state, kwargs)

    def parameters(self) -> list[_FakeParameter]:
        return [_FakeParameter(is_meta=self.remaining_meta)]


def test_low_memory_loader_maps_and_assigns_registered_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"weight": object()}
    torch = _FakeTorch({"model": state})
    model = _FakeModel()
    source_path = Path("dinov2-source")
    checkpoint_path = Path("DDA_ckpt.pth")

    def build_model(torch_module: Any, _nn_module: Any, source: Path) -> _FakeModel:
        assert torch_module is torch
        assert torch.in_meta_context is True
        assert source == source_path
        return model

    monkeypatch.setattr(dda, "_DdaModel", build_model)

    loaded = dda._load_dda_model_low_memory(
        torch,
        object(),
        checkpoint_path,
        source_path=source_path,
    )

    assert loaded is model
    assert torch.load_call == (
        checkpoint_path,
        {"map_location": "cpu", "weights_only": True, "mmap": True},
    )
    assert model.load_call == (state, {"strict": True, "assign": True})


def test_low_memory_loader_rejects_checkpoint_without_model_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch({"unexpected": {}})
    monkeypatch.setattr(dda, "_DdaModel", lambda *_args: _FakeModel())

    with pytest.raises(ValueError, match="checkpoint_model_state_missing"):
        dda._load_dda_model_low_memory(
            torch,
            object(),
            Path("DDA_ckpt.pth"),
            source_path=Path("dinov2-source"),
        )


def test_low_memory_loader_rejects_unmaterialized_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch({"model": {"weight": object()}})
    monkeypatch.setattr(
        dda,
        "_DdaModel",
        lambda *_args: _FakeModel(remaining_meta=True),
    )

    with pytest.raises(ValueError, match="unmaterialized_parameters"):
        dda._load_dda_model_low_memory(
            torch,
            object(),
            Path("DDA_ckpt.pth"),
            source_path=Path("dinov2-source"),
        )


def test_low_memory_loader_fails_closed_when_torch_api_is_too_old(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _UnsupportedFakeTorch({"model": {"weight": object()}})
    monkeypatch.setattr(dda, "_DdaModel", lambda *_args: _FakeModel())

    with pytest.raises(ValueError, match="low_memory_torch_api_unavailable"):
        dda._load_dda_model_low_memory(
            torch,
            object(),
            Path("DDA_ckpt.pth"),
            source_path=Path("dinov2-source"),
        )


def test_isolated_scorer_reports_known_low_memory_api_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "asset.png"
    checkpoint_path = tmp_path / "checkpoint.pth"
    Image.new("RGB", (336, 336), "white").save(input_path)
    checkpoint_path.write_bytes(b"registered-by-caller")

    class _Completed:
        returncode = 1
        stderr = "ValueError: dda_low_memory_torch_api_unavailable"

    monkeypatch.setattr(dda.subprocess, "run", lambda *_args, **_kwargs: _Completed())

    with pytest.raises(
        dda.DdaUnavailableError,
        match="dda_low_memory_torch_api_unavailable",
    ):
        dda.score_dda_isolated(input_path, checkpoint_path=checkpoint_path)
