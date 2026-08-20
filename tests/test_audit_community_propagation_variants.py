from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_community_propagation_variants", ROOT / "scripts" / "audit_community_propagation_variants.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_rate_uses_fixed_threshold_without_reselection() -> None:
    result = audit._rate([{"score": 0.5}, {"score": 0.9}, {"score": 0.2}], 0.5)
    assert result == {"hits": 2, "total": 3, "rate": pytest.approx(2 / 3)}


def test_partial_checkpoint_requires_matching_protocol(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scores.partial.json"
    checkpoint.write_text('{"protocol_sha256":"different","rows":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="different protocol"):
        audit._completed_rows(checkpoint, "registered")


def test_summary_marks_webp_as_not_product_high_eligible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text('{"high_confidence_threshold":0.9,"limited_review_threshold":0.5}', encoding="utf-8")
    protocol = {"model": {"audit_path": str(audit_path)}}
    sources = [
        {"asset_sha256": "a", "label": "fake"},
        {"asset_sha256": "b", "label": "real"},
    ]
    rows = [
        {"source_asset_sha256": asset, "profile": profile, "score": score}
        for profile in audit.PROFILES
        for asset, score in (("a", 0.95), ("b", 0.2))
    ]
    result = audit._summarize(protocol, rows, sources)
    assert result["profile_metrics"]["webp_reencode_quality=85"]["product_high_eligible"] is False
    assert result["profile_metrics"]["original_decode"]["generated_high_threshold"]["hits"] == 1


def test_protocol_registers_a_strict_runtime_equivalence_probe() -> None:
    import json

    protocol = json.loads(
        (ROOT / "research/records/2026-08-19/pixel/community_propagation_audit_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )
    probe = protocol["implementation"]["equivalence_probe"]
    assert probe["profile"] == "screenshot_raster_png_longest=1600"
    assert probe["absolute_difference_maximum"] == pytest.approx(1e-6)
