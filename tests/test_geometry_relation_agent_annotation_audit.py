from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from image_trust.geometry_ai.relation_annotations import build_review_packet


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixtures = _load(
    "agent_audit_relation_fixtures",
    "tests/test_geometry_relation_annotations.py",
)
auditor = _load(
    "agent_annotation_auditor",
    "scripts/audit_geometry_relation_agent_annotations.py",
)


def _setup(tmp_path: Path):
    blind = tmp_path / "blind"
    packets = blind / "packets"
    annotations = tmp_path / "agent_annotations"
    hashes: dict[str, str] = {}
    rows = []
    for index in range(36):
        reviewer_id = f"grr-{index:02d}"
        packet, annotation = build_review_packet(reviewer_id, fixtures._measurement())
        packet_dir = packets / reviewer_id
        packet_dir.mkdir(parents=True)
        packet_path = packet_dir / "review_packet.json"
        annotation_path = packet_dir / "annotation.json"
        packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
        annotation_path.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
        rows.append(
            {
                "reviewer_id": reviewer_id,
                "packet": f"packets/{reviewer_id}/review_packet.json",
                "annotation": f"packets/{reviewer_id}/annotation.json",
            }
        )
        hashes[reviewer_id] = auditor._sha256(annotation_path)
        output = annotations / f"{reviewer_id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
    (blind / "review_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    hashes_path = tmp_path / "template_hashes.json"
    hashes_path.write_text(json.dumps(hashes), encoding="utf-8")
    return blind, annotations, hashes_path


def test_agent_audit_preserves_source_blindness_and_human_templates(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["source_key_opened"] is False
    assert report["gates"]["all_36_annotations_present_once"] is True
    assert report["gates"]["human_templates_unchanged"] is True
    assert report["gates"]["all_annotations_frozen"] is False
    assert len(report["agent_annotation_closure_sha256"]) == 64
    assert len(report["semantic_annotation_closure_sha256"]) == 64


def test_agent_audit_accepts_bom_prefixed_template_hash_closure(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    hashes.write_text(hashes.read_text(encoding="utf-8"), encoding="utf-8-sig")

    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["gates"]["human_templates_unchanged"] is True
    assert report["validation_errors"] == []


def test_agent_audit_accepts_bom_prefixed_annotation(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    annotation = annotations / "grr-00.json"
    annotation.write_text(annotation.read_text(encoding="utf-8"), encoding="utf-8-sig")

    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["validated_annotation_count"] == 36
    assert report["validation_errors"] == []


def test_agent_audit_accepts_complete_contract_valid_blind_annotations(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    for annotation_path in annotations.glob("*.json"):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        payload["status"] = "completed"
        payload["surfaces"] = [
            {
                "surface_id": "surface-001",
                "surface_kind": "roof",
                "polygon_normalized": [
                    {"x": 0.05, "y": 0.05},
                    {"x": 0.95, "y": 0.05},
                    {"x": 0.95, "y": 0.95},
                ],
                "line_ids": ["m0001", "m0002", "m0003", "m0004"],
                "visibility": "clear",
                "note": "",
            }
        ]
        for review in payload["proposed_family_reviews"]:
            review["verdict"] = "coherent_within_surface"
            review["surface_ids"] = ["surface-001"]
        annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 0
    assert report["passed"] is True
    assert report["validated_annotation_count"] == 36
    assert report["source_key_opened"] is False


def test_agent_audit_rejects_source_fields_and_template_drift(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    annotation_path = annotations / "grr-00.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["sample_id"] = "forbidden"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    template_path = blind / "packets/grr-01/annotation.json"
    template_path.write_text("{}\n", encoding="utf-8")

    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["source_key_opened"] is False
    assert report["validated_annotation_count"] == 35
    assert report["template_drift_reviewer_ids"] == ["grr-01"]
    assert any("forbidden fields" in error for error in report["validation_errors"])


def test_agent_audit_rejects_decided_family_with_unexplained_members(
    tmp_path: Path,
) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    annotation_path = annotations / "grr-00.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["status"] = "completed"
    payload["surfaces"] = [
        {
            "surface_id": "surface-001",
            "surface_kind": "roof",
            "polygon_normalized": [
                {"x": 0.05, "y": 0.05},
                {"x": 0.95, "y": 0.05},
                {"x": 0.95, "y": 0.95},
            ],
            "line_ids": ["m0001", "m0003"],
            "visibility": "clear",
            "note": "",
        }
    ]
    for review in payload["proposed_family_reviews"]:
        review["verdict"] = "coherent_within_surface"
        review["surface_ids"] = ["surface-001"]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["validated_annotation_count"] == 35
    assert any(
        "unexplained by its reviewed surfaces" in error
        for error in report["validation_errors"]
    )


def test_agent_audit_rejects_duplicate_reviewer_files(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    duplicate = annotations / "duplicate" / "grr-00.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes((annotations / "grr-00.json").read_bytes())

    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["duplicate_reviewer_ids"] == ["grr-00"]
    assert report["gates"]["all_36_annotations_present_once"] is False


def test_agent_audit_rejects_source_label_text(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    annotation_path = annotations / "grr-00.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["review_note"] = "looks like an SDXL generated image"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["validated_annotation_count"] == 35
    assert any(
        "forbidden source-label text" in error
        for error in report["validation_errors"]
    )


def test_agent_audit_rejects_additional_relation_without_surface_closure(
    tmp_path: Path,
) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    annotation_path = annotations / "grr-00.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["status"] = "completed"
    payload["surfaces"] = [
        {
            "surface_id": "surface-main",
            "surface_kind": "roof",
            "polygon_normalized": [
                {"x": 0.05, "y": 0.05},
                {"x": 0.95, "y": 0.05},
                {"x": 0.95, "y": 0.95},
            ],
            "line_ids": ["m0001", "m0002", "m0003", "m0004"],
            "visibility": "clear",
            "note": "",
        },
        {
            "surface_id": "surface-secondary",
            "surface_kind": "roof",
            "polygon_normalized": [
                {"x": 0.10, "y": 0.10},
                {"x": 0.20, "y": 0.10},
                {"x": 0.20, "y": 0.20},
            ],
            "line_ids": ["m0001"],
            "visibility": "clear",
            "note": "",
        },
    ]
    for review in payload["proposed_family_reviews"]:
        review["verdict"] = "coherent_within_surface"
        review["surface_ids"] = ["surface-main"]
    payload["additional_relations"] = [
        {
            "relation_id": "relation-001",
            "relation_type": "parallel_family",
            "scope": "within_surface",
            "surface_ids": ["surface-secondary"],
            "member_line_ids": ["m0001", "m0002"],
            "verdict": "coherent",
            "outlier_line_ids": [],
            "note": "",
        }
    ]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["validated_annotation_count"] == 35
    assert any(
        "relation relation-001 leaves member lines unexplained" in error
        for error in report["validation_errors"]
    )


def test_agent_audit_rejects_registered_input_hash_drift(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "input": {
                    "review_manifest_sha256": "0" * 64,
                    "original_annotation_template_hash_closure_sha256": auditor._sha256(
                        hashes
                    ),
                    "packet_count": 36,
                },
                "output": {"root": annotations.name},
                "origin_scoring_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    report, exit_code = auditor.audit(
        blind,
        annotations,
        hashes,
        tmp_path / "report.json",
        protocol,
    )

    assert exit_code == 2
    assert report["protocol_verified"] is True
    assert report["gates"]["registered_input_closure_matches"] is False
    assert report["registration_errors"] == [
        "blind manifest hash differs from protocol"
    ]


def test_agent_audit_rejects_filename_identity_drift(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    annotation_path = annotations / "grr-00.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["reviewer_id"] = "grr-01"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    report, exit_code = auditor.audit(blind, annotations, hashes, tmp_path / "report.json")

    assert exit_code == 2
    assert report["validated_annotation_count"] == 35
    assert any(
        "reviewer ID must match its filename" in error
        for error in report["validation_errors"]
    )


def test_agent_audit_enforces_registered_role_directories(tmp_path: Path) -> None:
    blind, annotations, hashes = _setup(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "input": {
                    "review_manifest_sha256": auditor._sha256(
                        blind / "review_manifest.jsonl"
                    ),
                    "original_annotation_template_hash_closure_sha256": auditor._sha256(
                        hashes
                    ),
                    "packet_count": 36,
                },
                "roles": [
                    {
                        "manifest_positions": [1, 36],
                        "output_subdirectory": "registered-part",
                    }
                ],
                "output": {"root": annotations.name},
                "origin_scoring_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    report, exit_code = auditor.audit(
        blind,
        annotations,
        hashes,
        tmp_path / "report.json",
        protocol,
    )

    assert exit_code == 2
    assert report["gates"]["registered_input_closure_matches"] is False
    assert any(
        "outside its registered role directory" in error
        for error in report["registration_errors"]
    )
