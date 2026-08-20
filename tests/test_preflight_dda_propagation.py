from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/preflight_dda_propagation.py")


def _module():
    spec = importlib.util.spec_from_file_location("preflight_dda_propagation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_python_tree_hash_is_path_independent_and_detects_content(tmp_path: Path) -> None:
    module = _module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "pkg").mkdir(parents=True)
    (second / "pkg").mkdir(parents=True)
    (first / "pkg" / "a.py").write_text("value = 1\n", encoding="utf-8")
    (second / "pkg" / "a.py").write_text("value = 1\n", encoding="utf-8")
    assert module._python_tree_sha256(first) == module._python_tree_sha256(second)
    (second / "pkg" / "a.py").write_text("value = 2\n", encoding="utf-8")
    assert module._python_tree_sha256(first) != module._python_tree_sha256(second)


def test_variant_selection_rejects_hash_mismatch(tmp_path: Path) -> None:
    module = _module()
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(b"image")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "records": [
                {
                    "source_asset_sha256": "source",
                    "profile": "screen",
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "relative_path": "artifact.png",
                }
            ]
        },
    )
    protocol = {
        "variant_manifest": {
            "path": "manifest.json",
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "record_count": 1,
        },
        "input": {
            "source_asset_sha256": "source",
            "profile": "screen",
            "artifact_sha256": "wrong",
        },
    }
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        module._select_variant(tmp_path, protocol)


def test_prior_probe_requires_the_same_input(tmp_path: Path) -> None:
    module = _module()
    report = tmp_path / "probe.json"
    _write_json(report, {"input": {"different": True}, "rows": []})
    protocol = {
        "prior_probe": {
            "path": "probe.json",
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
        "input": {"source_asset_sha256": "source"},
    }
    with pytest.raises(ValueError, match="input differs"):
        module._prior_score(tmp_path, protocol)


def test_descendant_pids_include_transitive_children() -> None:
    module = _module()
    assert module._descendant_pids(10, {11: 10, 12: 11, 13: 99}) == {10, 11, 12}


def test_process_tree_working_set_sums_launcher_and_children(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_process_parent_map", lambda: {11: 10, 12: 11, 13: 99})
    monkeypatch.setattr(module, "_working_set_bytes", lambda pid: {10: 5, 11: 100, 12: 20}[pid])
    assert module._process_tree_working_set_bytes(10) == (125, 3)


def test_protocol_binds_preflight_runtime_and_two_gibibyte_stop() -> None:
    protocol_path = Path("research/records/2026-08-20/pixel/dda_propagation_preflight_protocol_v2.json")
    assert protocol_path.is_file()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    implementation = protocol["implementation"]
    assert hashlib.sha256(Path(implementation["preflight_script_path"]).read_bytes()).hexdigest() == implementation["preflight_script_sha256"]
    assert hashlib.sha256(Path(implementation["runtime_adapter_path"]).read_bytes()).hexdigest() == implementation["runtime_adapter_sha256"]
    assert protocol["execution"]["maximum_working_set_bytes"] == 2 * 1024 * 1024 * 1024
    assert protocol["execution"]["batch_size"] == 1
    assert protocol["execution"]["memory_sampling_scope"] == "launcher_and_all_descendants"
