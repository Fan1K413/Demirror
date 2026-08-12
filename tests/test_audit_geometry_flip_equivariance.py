from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_geometry_flip_equivariance",
    ROOT / "scripts/audit_geometry_flip_equivariance.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_threshold_maximizes_recall_under_fpr_constraint_and_breaks_ties_high() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.9, 0.25, 0.4, 0.8, 0.95])

    threshold, rates = audit.select_threshold(
        labels,
        probabilities,
        maximum_false_positive_rate=0.25,
    )

    assert threshold == 0.4
    assert rates["true_positive_rate"] == 0.75
    assert rates["false_positive_rate"] == 0.25


def test_valid_shard_requires_exact_sample_identity_and_feature_order(tmp_path: Path) -> None:
    sample = audit.Sample(
        archive="Recent_Pixart_Indoor",
        generator="pixart",
        identifier=351,
        label=0,
        path=str((tmp_path / "351.jpg").resolve()),
        scene="indoor",
        sha256="a" * 64,
        split="calibration",
    )
    features = ("first", "second")
    path = tmp_path / "part.json"
    audit._atomic_write_json(
        path,
        {
            "schema_version": "demirror-geometry-flip-equivariance-shard-v1",
            "rows": [
                {
                    "sample": audit.asdict(sample),
                    "metrics": {"first": 1.0, "second": 2.0},
                }
            ],
        },
    )

    assert audit._valid_shard(path, [sample], features)
    assert not audit._valid_shard(path, [sample], tuple(reversed(features)))


def test_rates_keep_generated_positive_and_real_negative() -> None:
    rates = audit._rates(
        np.asarray([0, 0, 1, 1]),
        np.asarray([False, True, True, False]),
    )
    assert rates == {
        "true_positive_count": 1,
        "positive_count": 2,
        "true_positive_rate": 0.5,
        "false_positive_count": 1,
        "negative_count": 2,
        "false_positive_rate": 0.5,
    }


def test_incomplete_state_preserves_previous_peak_for_same_protocol(tmp_path: Path) -> None:
    state = tmp_path / "incomplete_state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": "demirror-geometry-flip-equivariance-incomplete-v1",
                "protocol_sha256": "a" * 64,
                "resource": {
                    "maximum_sampled_worker_working_set_bytes": 1234,
                },
            }
        ),
        encoding="utf-8",
    )
    resource: dict[str, int | float] = {
        "maximum_sampled_worker_working_set_bytes": 1000,
    }

    audit._preserve_previous_peak(
        state,
        protocol_sha256="a" * 64,
        resource=resource,
    )

    assert resource["maximum_sampled_worker_working_set_bytes"] == 1234


def test_default_protocol_path_follows_the_research_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["audit_geometry_flip_equivariance.py"])

    args = audit._parse_args()

    assert args.protocol == (
        ROOT
        / "research/records/2026-08-12/geometry/geometry_flip_equivariance_protocol_v1.json"
    )
