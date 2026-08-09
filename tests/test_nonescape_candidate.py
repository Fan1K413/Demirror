from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/evaluate_nonescape_candidate.py")
    spec = importlib.util.spec_from_file_location("evaluate_nonescape_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parses_inclusive_ids_and_rejects_reverse_ranges() -> None:
    subject = _module()

    assert subject._parse_ids("426:428") == (426, 427, 428)
    try:
        subject._parse_ids("428:426")
    except ValueError as error:
        assert "end" in str(error)
    else:
        raise AssertionError("reverse range must fail")


def test_transform_label_is_explicit_about_codec_and_scale() -> None:
    subject = _module()

    assert subject._input_transform_label(None, None, 1.0) == "original_decode"
    assert subject._input_transform_label(85, None, 1.25) == "lanczos_scale=1.25+jpeg_reencode_quality=85"
    assert subject._input_transform_label(None, 80, 1.0) == "webp_reencode_quality=80"
