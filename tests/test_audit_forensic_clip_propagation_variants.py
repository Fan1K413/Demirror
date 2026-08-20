from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/audit_forensic_clip_propagation_variants.py")


def _module():
    spec = importlib.util.spec_from_file_location("audit_forensic_clip_propagation_variants", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_partial_rows_reject_labels_and_unregistered_keys(tmp_path: Path) -> None:
    module = _module()
    checkpoint = tmp_path / "scores.partial.json"
    _write_json(
        checkpoint,
        {
            "protocol_sha256": "registered",
            "rows": [
                {
                    "source_manifest_index": 0,
                    "source_asset_sha256": "asset",
                    "profile": "original_decode",
                    "artifact_sha256": "artifact",
                    "score": 0.4,
                    "label": "fake",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="label-free"):
        module._completed_rows(checkpoint, "registered")


def test_summary_keeps_high_and_limited_thresholds_frozen(tmp_path: Path) -> None:
    module = _module()
    audit_path = tmp_path / "forensic.json"
    _write_json(
        audit_path,
        {"high_confidence_threshold": 0.9, "limited_review_threshold": 0.7},
    )
    protocol = {"model": {"audit_path": str(audit_path)}}
    source_records = [
        {"asset_sha256": "fake", "label": "fake"},
        {"asset_sha256": "real", "label": "real"},
    ]
    scores = []
    for profile in module.PROFILES:
        scores.extend(
            [
                {"source_asset_sha256": "fake", "profile": profile, "score": 0.8},
                {"source_asset_sha256": "real", "profile": profile, "score": 0.1},
            ]
        )
    summary = module._summarize(protocol, scores, source_records)
    assert summary["high_confidence_threshold"] == 0.9
    assert summary["limited_review_threshold"] == 0.7
    assert summary["profile_metrics"]["original_decode"]["generated_high_threshold"]["hits"] == 0
    assert summary["profile_metrics"]["original_decode"]["generated_limited_threshold"]["hits"] == 1
    assert summary["profile_metrics"]["webp_reencode_quality=85"]["product_high_eligible"] is False


def test_protocol_binds_the_research_runner_and_product_runtime() -> None:
    protocol_path = Path("research/records/2026-08-19/pixel/forensic_clip_propagation_audit_protocol_v1.json")
    assert protocol_path.is_file()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    implementation = protocol["implementation"]
    assert hashlib.sha256(Path(implementation["audit_script_path"]).read_bytes()).hexdigest() == implementation["audit_script_sha256"]
    assert hashlib.sha256(Path(implementation["runtime_adapter_path"]).read_bytes()).hexdigest() == implementation["runtime_adapter_sha256"]
    assert implementation["equivalence_probe"]["absolute_difference_maximum"] == 1e-6


def test_runtime_keeps_product_temperature_score_direction_and_checkpoint_checks() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "model.load_state_dict(state, strict=False)" in source
    assert "if missing or unexpected:" in source
    assert "return 1.0 - real_probability" in source
    assert "score_forensic_clip_isolated" in source
