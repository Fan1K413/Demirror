from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preflight_lota_candidate", ROOT / "scripts" / "preflight_lota_candidate.py"
)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


def test_fixed_smoke_image_is_deterministic_rgb() -> None:
    first = np.asarray(preflight._fixed_smoke_image())
    second = np.asarray(preflight._fixed_smoke_image())

    assert first.shape == (320, 448, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first[..., 0], first[..., 1])


def test_require_hash_rejects_modified_registered_source(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("first\n", encoding="utf-8")
    expected = preflight._sha256(source)
    preflight._require_hash(source, expected)

    source.write_text("second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preflight._require_hash(source, expected)


def test_require_hash_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        preflight._require_hash(tmp_path / "missing.py", "0" * 64)


def test_cpu_smoke_protocol_uses_the_published_low_bit_configuration() -> None:
    import json

    protocol = json.loads(
        (ROOT / "research/records/2026-08-19/pixel/lota_cpu_preflight_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )

    smoke = protocol["cpu_smoke"]
    assert smoke == {
        "cpu_threads": 2,
        "seed": 20260819,
        "img_height": 256,
        "bit_mode": "scaling",
        "patch_size": 32,
        "patch_mode": "max",
        "expected_parameter_count": 23510081,
    }
    assert protocol["policy"]["no_checkpoint_no_detector_scoring"] is True


def test_model_zoo_guard_forces_no_pretrained_download_and_restores_symbol() -> None:
    calls: list[bool] = []

    def resnet50(*, pretrained: bool = True) -> dict[str, bool]:
        calls.append(pretrained)
        return {"pretrained": pretrained}

    def construct() -> dict[str, bool]:
        return module.resnet50(pretrained=True)

    module = types.SimpleNamespace(resnet50=resnet50, model=construct)
    original = module.resnet50

    result = preflight._build_model_without_model_zoo(module)

    assert result == {"pretrained": False}
    assert calls == [False]
    assert module.resnet50 is original
