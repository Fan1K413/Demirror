from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/audit_dda_propagation_variants.py")


def _module():
    spec = importlib.util.spec_from_file_location("audit_dda_propagation_variants", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _views() -> list[dict[str, object]]:
    return [
        {
            "source_manifest_index": 0,
            "source_asset_sha256": "a",
            "profile": "original_decode",
            "artifact_sha256": "a",
            "path": Path("a.png"),
        },
        {
            "source_manifest_index": 0,
            "source_asset_sha256": "a",
            "profile": "jpeg_reencode_quality=85",
            "artifact_sha256": "b",
            "path": Path("b.jpg"),
        },
    ]


def test_completed_rows_accepts_only_an_exact_label_free_prefix(tmp_path: Path) -> None:
    module = _module()
    checkpoint = tmp_path / "scores.partial.json"
    rows = [
        {
            "source_manifest_index": 0,
            "source_asset_sha256": "a",
            "profile": "original_decode",
            "artifact_sha256": "a",
            "status": "available",
            "score": 0.25,
        }
    ]
    _write_json(checkpoint, module._partial("protocol", rows))
    assert module._completed_rows(checkpoint, "protocol", _views()) == rows


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"profile": "jpeg_reencode_quality=85"}, "exact registered prefix"),
        ({"label": "fake"}, "label-free registered row"),
        ({"score": 1.1}, "outside"),
    ],
)
def test_completed_rows_rejects_reordering_labels_and_invalid_scores(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    module = _module()
    checkpoint = tmp_path / "scores.partial.json"
    row: dict[str, object] = {
        "source_manifest_index": 0,
        "source_asset_sha256": "a",
        "profile": "original_decode",
        "artifact_sha256": "a",
        "status": "available",
        "score": 0.25,
    }
    row.update(mutation)
    _write_json(checkpoint, module._partial("protocol", [row]))
    with pytest.raises(ValueError, match=message):
        module._completed_rows(checkpoint, "protocol", _views())


def test_rate_excludes_unavailable_views_from_denominator() -> None:
    module = _module()
    assert module._rate(
        [
            {"status": "available", "score": 0.95},
            {"status": "available", "score": 0.10},
            {"status": "unavailable", "reason": "dda_input_too_small"},
        ],
        0.94,
    ) == {
        "hits": 1,
        "available": 2,
        "unavailable": 1,
        "rate_over_available": 0.5,
    }


def test_failure_report_preserves_only_checkpoint_receipt(tmp_path: Path) -> None:
    module = _module()
    checkpoint = tmp_path / "scores.partial.json"
    _write_json(checkpoint, module._partial("protocol", [{"score": 0.1}]))
    report = module._failure_report(
        "protocol",
        {
            "stop_reason": "working_set_limit_exceeded",
            "returncode": 1,
            "warning_codes": [],
            "stderr_present_unclassified": False,
            "maximum_sampled_working_set_bytes": 3,
            "maximum_sampled_process_count": 2,
        },
        checkpoint,
    )
    assert report["completed_label_free_score_rows"] == 1
    assert report["result_interpretation_allowed"] is False
    assert "rows" not in report


def test_protocol_binds_runtime_monitor_preflight_and_two_gibibyte_stop() -> None:
    protocol_path = Path(
        "research/records/2026-08-20/pixel/dda_propagation_audit_protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    implementation = protocol["implementation"]
    for path_key, hash_key in (
        ("audit_script_path", "audit_script_sha256"),
        ("resource_monitor_path", "resource_monitor_sha256"),
        ("runtime_adapter_path", "runtime_adapter_sha256"),
    ):
        assert (
            hashlib.sha256(Path(implementation[path_key]).read_bytes()).hexdigest()
            == implementation[hash_key]
        )
    preflight = protocol["resource_preflight"]
    assert hashlib.sha256(Path(preflight["path"]).read_bytes()).hexdigest() == preflight["sha256"]
    execution = protocol["execution"]
    assert execution["maximum_working_set_bytes"] == 2 * 1024 * 1024 * 1024
    assert execution["batch_size"] == 1
    assert execution["detector_processes_in_parallel"] == 1
    assert execution["checkpoint_after_each_view"] is True
    assert execution["memory_sampling_scope"] == "launcher_and_all_descendants"
