from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("scripts/evaluate_bfree_candidate.py")
    spec = importlib.util.spec_from_file_location("evaluate_bfree_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_ids_and_generators_are_bounded() -> None:
    module = _load_module()
    assert module._parse_ids("4:6") == (4, 5, 6)
    assert module._parse_ids("9, 11") == (9, 11)
    assert module._parse_generators("SDXL,Pixart") == ("SDXL", "Pixart")


def test_summary_uses_upstream_zero_logit_threshold() -> None:
    module = _load_module()
    summary = module._summary(
        [
            {"label": 0, "ai_logit": -0.5},
            {"label": 0, "ai_logit": 0.2},
            {"label": 1, "ai_logit": 0.4},
            {"label": 1, "ai_logit": -0.1},
        ]
    )
    assert summary["upstream_default_logit_threshold"] == 0.0
    assert summary["true_positive"] == 1
    assert summary["false_positive"] == 1
    assert summary["true_negative"] == 1
    assert summary["false_negative"] == 1
