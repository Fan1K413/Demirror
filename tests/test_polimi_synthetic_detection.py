"""Unit coverage for the resource-bounded Polimi screening harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_polimi_synthetic_detection.py"
    spec = importlib.util.spec_from_file_location("polimi_synthetic_screen", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_ids_accepts_ranges_and_lists() -> None:
    module = _module()
    assert module._parse_ids("426:428") == (426, 427, 428)
    assert module._parse_ids("426, 430") == (426, 430)


def test_summary_uses_the_released_positive_score_direction() -> None:
    module = _module()
    report = module._summary(
        [
            {"label": 0, "synthetic_score": -0.2},
            {"label": 1, "synthetic_score": 0.3},
            {"label": 0, "synthetic_score": -0.1},
            {"label": 1, "synthetic_score": 0.2},
        ]
    )
    assert report["accuracy"] == 1.0
    assert report["roc_auc_synthetic_score"] == 1.0
    assert report["true_positive_rate"] == 1.0
    assert report["false_positive_rate"] == 0.0
