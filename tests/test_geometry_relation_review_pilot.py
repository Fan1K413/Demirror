from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from image_trust.geometry_ai.relation_annotations import build_review_packet


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


annotation_contract_tests = _load_script(
    "geometry_relation_annotation_test_fixtures",
    "tests/test_geometry_relation_annotations.py",
)
_measurement = annotation_contract_tests._measurement

builder = _load_script(
    "geometry_relation_builder",
    "scripts/build_geometry_relation_review_pilot.py",
)
auditor = _load_script(
    "geometry_relation_auditor",
    "scripts/audit_geometry_relation_review_pilot.py",
)


def _registry_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for archive in ("pixart-indoor", "pixart-outdoor", "sdxl-indoor", "sdxl-outdoor"):
        for label in ("ai_generated", "camera_photo"):
            for index in range(6):
                sample_id = f"{archive}-{label}-{index}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "sha256": f"{len(rows):064x}",
                        "relative_path": f"private/{sample_id}.png",
                        "split": "development",
                        "declared_source_slice": {
                            "archive_name": archive,
                            "generator_family": archive.split("-")[0],
                            "label_name": label,
                            "must_not_be_used_as_geometry_label": True,
                        },
                    }
                )
    return rows


def test_default_pilot_selects_32_unique_images_and_four_hidden_duplicates() -> None:
    selected = builder.select_unique_rows(
        _registry_rows(),
        per_stratum=4,
        seed="fixed-seed",
    )
    duplicates = builder.select_duplicate_rows(selected, count=4, seed="fixed-seed")
    plan = builder.make_packet_plan(selected, duplicates, seed="fixed-seed")

    assert len(selected) == 32
    assert len({row["sample_id"] for row in selected}) == 32
    assert len(duplicates) == 4
    assert len(plan) == 36
    assert len({item["reviewer_id"] for item in plan}) == 36
    assert {sum(item["source_group"] == group for item in plan) for group in {item["source_group"] for item in plan}} == {1, 2}


def test_registered_protocol_closes_the_actual_source_set_and_implementation() -> None:
    registry = REPOSITORY_ROOT / "data/p0_geometry_anomaly_v1/cross_generator_review_registry.jsonl"
    protocol = REPOSITORY_ROOT / "research/records/2026-08-12/geometry/geometry_semantic_relation_pilot_protocol_v1.json"
    rows = builder._read_registry(registry)
    selected = builder.select_unique_rows(
        rows,
        per_stratum=4,
        seed=builder.DEFAULT_SELECTION_SEED,
    )

    validated = builder.validate_registered_protocol(
        protocol,
        project_root=REPOSITORY_ROOT,
        registry_path=registry,
        selected=selected,
        per_stratum=4,
        duplicate_count=4,
    )

    assert validated["origin_scoring_authorized"] is False


def test_blind_payload_scan_rejects_source_keys_and_values() -> None:
    builder.assert_blind_payload(
        {"reviewer_id": "grr-123", "packet": "packets/grr-123/review_packet.json"},
        {"ai_generated", "source-file.jpg"},
    )

    try:
        builder.assert_blind_payload(
            {"reviewer_id": "grr-123", "sample_id": "source-1"},
            set(),
        )
    except ValueError as error:
        assert "source field leaked" in str(error)
    else:
        raise AssertionError("source key leakage was accepted")


def test_incomplete_audit_does_not_open_posthoc_source_key(tmp_path: Path) -> None:
    blind_root = tmp_path / "blind"
    packet_dir = blind_root / "packets" / "grr-test"
    packet_dir.mkdir(parents=True)
    packet, annotation = build_review_packet("grr-test", _measurement())
    (packet_dir / "review_packet.json").write_text(
        packet.model_dump_json(indent=2), encoding="utf-8"
    )
    (packet_dir / "annotation.json").write_text(
        annotation.model_dump_json(indent=2), encoding="utf-8"
    )
    (blind_root / "review_manifest.jsonl").write_text(
        json.dumps(
            {
                "reviewer_id": "grr-test",
                "packet": "packets/grr-test/review_packet.json",
                "annotation": "packets/grr-test/annotation.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    missing_key = tmp_path / "posthoc" / "missing-review-key.jsonl"
    report_path = tmp_path / "report.json"

    report, exit_code = auditor.audit(blind_root, missing_key, report_path)

    assert exit_code == 2
    assert report["status"] == "incomplete"
    assert report["source_key_opened"] is False
    assert "posthoc_source_summary" not in report
    assert report_path.is_file()
