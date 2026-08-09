from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "geometry_cache_classifier_screen",
    REPOSITORY_ROOT / "scripts" / "benchmark_geometry_cache_classifiers.py",
)
assert SPEC is not None and SPEC.loader is not None
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def _samples() -> list[object]:
    rows = []
    groups = {
        "train": (("deepfloyd", "kandinsky"), range(1, 9)),
        "calibration": (("pixart",), range(351, 359)),
        "test": (("sdxl",), range(426, 434)),
    }
    for split, (generators, identifiers) in groups.items():
        for generator in generators:
            for identifier in identifiers:
                for label in (0, 1):
                    rows.append(
                        screen.CachedSample(
                            archive=f"Recent_{generator}_{identifier}",
                            generator=generator,
                            identifier=identifier,
                            label=label,
                            scene="indoor" if identifier % 2 else "outdoor",
                            split=split,
                        )
                    )
    return rows


def test_evaluate_uses_isolated_splits_and_reports_held_out_metrics() -> None:
    samples = _samples()
    rng = np.random.default_rng(4)
    relations = rng.normal(size=(len(samples), 6))
    for index, sample in enumerate(samples):
        relations[index, 0] += sample.label * 1.5

    report, predictions = screen.evaluate(relations, samples, target_fpr=0.25)

    assert report["selection_protocol"]["untouched_final_test"] == "SDXL IDs 426-500"
    assert report["selected_candidate"] in {
        "linear_l2_c_0_1",
        "linear_l2_c_1",
        "rbf_svm_c_0_5",
        "extra_trees_leaf_5",
        "hist_gradient_leaf_7",
    }
    assert report["held_out_test"]["count"] == 16
    assert len(predictions) == 16


def test_protocol_indices_rejects_wrong_generator_assignment() -> None:
    samples = _samples()
    invalid = list(samples)
    first = invalid[0]
    invalid[0] = screen.CachedSample(
        archive=first.archive,
        generator="sdxl",
        identifier=first.identifier,
        label=first.label,
        scene=first.scene,
        split=first.split,
    )

    try:
        screen.protocol_indices(invalid)
    except ValueError as exc:
        assert "generators" in str(exc)
    else:
        raise AssertionError("Expected wrong generator assignment to be rejected")
