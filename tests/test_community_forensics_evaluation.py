from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image


def _module():
    path = Path("scripts/evaluate_community_forensics.py")
    spec = importlib.util.spec_from_file_location("evaluate_community_forensics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_ids_rejects_reversed_range() -> None:
    module = _module()
    assert module._parse_ids("426:428") == (426, 427, 428)
    assert module._parse_ids("426, 428") == (426, 428)
    with pytest.raises(ValueError, match="end"):
        module._parse_ids("428:426")


def test_parse_generators_and_paths_preserve_generator_name(tmp_path: Path) -> None:
    module = _module()
    assert module._parse_generators("SDXL, Pixart") == ("SDXL", "Pixart")
    assert module._sample_paths(tmp_path, "Pixart", "Indoor", 426) == [
        (tmp_path / "Recent_Pixart_Indoor" / "Recent_Pixart_Indoor" / "test" / "real" / "426.jpg", 0),
        (tmp_path / "Recent_Pixart_Indoor" / "Recent_Pixart_Indoor" / "test" / "gen" / "426.jpg", 1),
    ]


def test_summary_uses_fixed_half_probability_threshold() -> None:
    module = _module()
    summary = module._summary(
        [
            {"label": 0, "ai_probability": 0.1},
            {"label": 0, "ai_probability": 0.9},
            {"label": 1, "ai_probability": 0.8},
            {"label": 1, "ai_probability": 0.95},
        ]
    )
    assert summary["roc_auc_ai_probability"] == 0.75
    assert summary["accuracy"] == 0.75
    assert summary["true_positive_rate"] == 1.0
    assert summary["false_positive_rate"] == 0.5


def test_test_transform_matches_upstream_sizes() -> None:
    module = _module()
    transform = module._test_transform(224)
    assert transform.transforms[0].size == 256
    assert transform.transforms[1].size == (224, 224)


def test_webp_resize_transform_is_explicit_and_decodable(tmp_path: Path) -> None:
    module = _module()
    image_path = tmp_path / "input.png"
    Image.new("RGB", (32, 48), (64, 128, 192)).save(image_path)
    values = module._model_input(
        image_path,
        module._test_transform(224),
        jpeg_quality=None,
        webp_quality=80,
        resize_scale=1.25,
    )
    assert tuple(values.shape) == (1, 3, 224, 224)
    assert module._input_transform_label(None, 80, 1.25) == "lanczos_scale=1.25+webp_reencode_quality=80"


def test_registered_threshold_metrics_do_not_reselect_threshold() -> None:
    module = _module()
    metrics = module._metrics_at_threshold(
        [
            {"label": 0, "ai_probability": 0.1},
            {"label": 0, "ai_probability": 0.95},
            {"label": 1, "ai_probability": 0.90},
            {"label": 1, "ai_probability": 0.99},
        ],
        0.95,
    )
    assert metrics["decision_threshold"] == 0.95
    assert metrics["true_positive_rate"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
