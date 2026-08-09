from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/evaluate_fsd_sdxl_pilot.py")
    spec = importlib.util.spec_from_file_location("evaluate_fsd_sdxl_pilot", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_keeps_default_threshold_metrics_separate_from_auc() -> None:
    module = _module()
    summary = module._summary(
        [
            {"label": 0, "z_score": 1.0},
            {"label": 0, "z_score": -3.0},
            {"label": 1, "z_score": -4.0},
            {"label": 1, "z_score": -1.0},
        ],
        threshold=-2.0,
    )
    assert summary["roc_auc_ai_negative_z"] == 0.75
    assert summary["true_positive"] == 1
    assert summary["false_positive"] == 1
    assert summary["true_negative"] == 1
    assert summary["false_negative"] == 1


def test_sample_paths_are_fixed_to_the_sdxl_test_pairs(tmp_path: Path) -> None:
    module = _module()
    paths = module._sample_paths(tmp_path, "Outdoor", 426)
    assert paths == [
        (tmp_path / "Recent_SDXL_Outdoor" / "Recent_SDXL_Outdoor" / "test" / "real" / "426.jpg", 0),
        (tmp_path / "Recent_SDXL_Outdoor" / "Recent_SDXL_Outdoor" / "test" / "gen" / "426.jpg", 1),
    ]
