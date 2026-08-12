from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_sidbench_candidates",
    ROOT / "scripts" / "screen_sidbench_candidates.py",
)
assert SPEC is not None and SPEC.loader is not None
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def test_wilson_interval_and_percentile_are_deterministic() -> None:
    interval = screen._wilson95(0, 40)

    assert interval["rate"] == 0.0
    assert interval["wilson_95_lower"] == 0.0
    assert interval["wilson_95_upper"] == pytest.approx(0.0876216012)
    assert screen._percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_patchcraft_adapter_repeats_exactly_for_a_fixed_seed() -> None:
    values = np.arange(96 * 128 * 3, dtype=np.uint32).reshape(96, 128, 3)
    image = Image.fromarray((values % 251).astype(np.uint8), mode="RGB")

    screen._set_seed(17)
    first = screen._patchcraft_input(image)
    screen._set_seed(17)
    second = screen._patchcraft_input(image)
    screen._set_seed(43)
    third = screen._patchcraft_input(image)

    assert first.shape == (1, 2, 3, 224, 224)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_seed_stability_reports_unanimity_and_score_range() -> None:
    rows = []
    for profile in screen.PROFILES:
        for seed, scores in ((17, (0.2, 0.7)), (43, (0.3, 0.8)), (101, (0.6, 0.9))):
            for asset, score in zip(("a", "b"), scores, strict=True):
                rows.append(
                    {
                        "profile": profile,
                        "seed": seed,
                        "asset_sha256": asset,
                        "score": score,
                    }
                )

    stability = screen._seed_stability({"seeds": [17, 43, 101], "rows": rows}, 0.5)

    assert stability is not None
    assert stability["original_decode"]["unanimous_binary_count"] == 1
    assert stability["original_decode"]["binary_agreement_rate"] == 0.5
    assert stability["original_decode"]["maximum_score_range"] == pytest.approx(0.4)


def test_binary_agreement_requires_identical_assets() -> None:
    rows = [
        {"seed": 42, "profile": "original_decode", "asset_sha256": "a", "score": 0.8},
        {
            "seed": 42,
            "profile": "jpeg_reencode_quality=85",
            "asset_sha256": "b",
            "score": 0.8,
        },
    ]

    with pytest.raises(ValueError, match="identical assets"):
        screen._binary_agreement({"rows": rows}, 42, 0.5)


def test_npr_checkpoint_adapter_removes_only_registered_prefix() -> None:
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])

    normalized = screen._strict_npr_state_dict(
        {"model": {"module.first": first, "module.second": second}}
    )

    assert normalized == {"first": first, "second": second}
    with pytest.raises(ValueError, match="module"):
        screen._strict_npr_state_dict({"model": {"first": first}})
