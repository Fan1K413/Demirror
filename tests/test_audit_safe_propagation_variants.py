from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/audit_safe_propagation_variants.py")


def _module():
    spec = importlib.util.spec_from_file_location("audit_safe_propagation_variants", SCRIPT)
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
                    "status": "available",
                    "score": 0.4,
                    "label": "fake",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="label-free"):
        module._completed_rows(checkpoint, "registered")


def test_summary_allows_product_high_only_for_original_decode(tmp_path: Path) -> None:
    module = _module()
    audit_path = tmp_path / "safe.json"
    _write_json(audit_path, {"high_confidence_threshold": 0.9})
    protocol = {"model": {"audit_path": str(audit_path)}}
    sources = [{"asset_sha256": "fake", "label": "fake"}, {"asset_sha256": "real", "label": "real"}]
    scores = [
        {"source_asset_sha256": source, "profile": profile, "status": "available", "score": score}
        for profile in module.PROFILES
        for source, score in (("fake", 0.95), ("real", 0.05))
    ]
    summary = module._summarize(protocol, scores, sources)
    assert summary["profile_metrics"]["original_decode"]["product_high_eligible"] is True
    assert summary["profile_metrics"]["jpeg_reencode_quality=85"]["product_high_eligible"] is False


def test_summary_excludes_unavailable_views_but_reports_them(tmp_path: Path) -> None:
    module = _module()
    audit_path = tmp_path / "safe.json"
    _write_json(audit_path, {"high_confidence_threshold": 0.9})
    protocol = {"model": {"audit_path": str(audit_path)}}
    sources = [{"asset_sha256": "fake", "label": "fake"}, {"asset_sha256": "real", "label": "real"}]
    scores = [
        {"source_asset_sha256": source, "profile": profile, "status": "available", "score": score}
        for profile in module.PROFILES
        for source, score in (("fake", 0.95), ("real", 0.05))
    ]
    scores.append(
        {
            "source_asset_sha256": "fake",
            "profile": "original_decode",
            "status": "unavailable",
            "reason": "safe_input_too_small",
        }
    )
    summary = module._summarize(protocol, scores, sources)
    metric = summary["profile_metrics"]["original_decode"]
    assert metric["available_count"] == 2
    assert metric["unavailable_count"] == 1
    assert metric["unavailable_reasons"] == {"safe_input_too_small": 1}


def test_protocol_binds_the_research_runner_and_product_runtime() -> None:
    protocol_path = Path("research/records/2026-08-19/pixel/safe_propagation_audit_protocol_v2.json")
    assert protocol_path.is_file()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    implementation = protocol["implementation"]
    assert hashlib.sha256(Path(implementation["audit_script_path"]).read_bytes()).hexdigest() == implementation["audit_script_sha256"]
    assert hashlib.sha256(Path(implementation["runtime_adapter_path"]).read_bytes()).hexdigest() == implementation["runtime_adapter_sha256"]
    assert implementation["equivalence_probe"]["absolute_difference_maximum"] == 1e-6


def test_runtime_keeps_product_wavelet_crop_and_checkpoint_checks() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'DWTForward(J=1, mode="symmetric", wave="bior1.3")' in source
    assert "image.crop((left, top, left + 256, top + 256))" in source
    assert 'any(not name.startswith("dwt.") for name in missing)' in source
    assert "score_safe_isolated" in source
