"""Unit coverage for the isolated FerretNet CPU research probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_ferretnet_candidate", ROOT / "scripts" / "screen_ferretnet_candidate.py"
)
assert SPEC and SPEC.loader
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def test_normalise_state_dict_strips_uniform_module_prefix() -> None:
    first, second = object(), object()
    assert screen._normalise_state_dict({"model": {"module.first": first, "module.second": second}}) == {
        "first": first,
        "second": second,
    }


def test_normalise_state_dict_rejects_mixed_prefixes() -> None:
    with pytest.raises(ValueError, match="mixes"):
        screen._normalise_state_dict({"model": {"module.first": object(), "second": object()}})


def test_variants_are_rgb_and_resize_profile_restores_size(tmp_path: Path) -> None:
    source = tmp_path / "sample.png"
    Image.new("RGB", (1800, 900), (20, 80, 140)).save(source)

    restored = screen._decode_variant(source, "resize_longest=1024_restore")
    screenshot = screen._decode_variant(source, "screenshot_raster_png_longest=1600")
    jpeg = screen._decode_variant(source, "jpeg_reencode_quality=85")

    assert restored.mode == screenshot.mode == jpeg.mode == "RGB"
    assert restored.size == (1800, 900)
    assert screenshot.size == (1600, 800)


def test_percentile_interpolates() -> None:
    assert screen._percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_wilson_interval_is_bounded() -> None:
    interval = screen._wilson95(0, 40)
    assert interval["rate"] == 0.0
    assert 0.0 <= interval["wilson_95_lower"] <= interval["wilson_95_upper"] <= 1.0


def test_tier1_parser_does_not_require_a_limit() -> None:
    parser = screen._parser()
    args = parser.parse_args(
        [
            "audit-tier1",
            "--original-score", "original.json",
            "--jpeg85-score", "jpeg.json",
            "--output", "audit.json",
        ]
    )
    assert not hasattr(args, "limit")


def test_score_parser_redacts_source_paths_by_default() -> None:
    parser = screen._parser()
    args = parser.parse_args(
        [
            "score",
            "--profile", "original_decode",
            "--output", "score.json",
        ]
    )
    assert args.include_relative_path is False
