"""Build a source-blind semantic-surface review pilot from the P0 registry.

The default pilot contains 32 unique images (four from each archive/declared
source stratum) and four hidden duplicate packets.  Source labels are used only
for deterministic balancing and are written exclusively to ``posthoc``.  The
measurement function receives only an image path and never receives a label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageOps

from image_trust.geometry_ai.measurement_types import GeometryMeasurementV2Result
from image_trust.geometry_ai.measurement_v2 import assess_geometry_measurement_v2
from image_trust.geometry_ai.relation_annotations import (
    build_review_packet,
    write_relation_review_overlays,
)


DEFAULT_SELECTION_SEED = "demirror-geometry-semantic-relation-pilot-2026-08-12-v1"
FORBIDDEN_BLIND_KEYS = {
    "archive_name",
    "declared_source_slice",
    "generator_family",
    "label_name",
    "original_relative_path",
    "relative_path",
    "sample_id",
    "source_label",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(seed: str, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{purpose}\0{value}".encode("utf-8")).hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_registry(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("review registry is empty")
    sample_ids = [str(row["sample_id"]) for row in rows]
    hashes = [str(row["sha256"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("review registry sample IDs and source hashes must be unique")
    return rows


def _stratum(row: dict[str, Any]) -> tuple[str, str]:
    source = dict(row["declared_source_slice"])
    return str(source["archive_name"]), str(source["label_name"])


def select_unique_rows(
    rows: list[dict[str, Any]],
    *,
    per_stratum: int,
    seed: str,
) -> list[dict[str, Any]]:
    """Select a fixed count per archive/source stratum without using geometry."""

    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_stratum(row)].append(row)
    if len(grouped) < 2:
        raise ValueError("pilot requires at least two source strata")
    selected: list[dict[str, Any]] = []
    for stratum, candidates in sorted(grouped.items()):
        if len(candidates) < per_stratum:
            raise ValueError(f"stratum {stratum!r} has fewer than {per_stratum} rows")
        ranked = sorted(
            candidates,
            key=lambda row: _rank(seed, f"stratum:{stratum}", str(row["sample_id"])),
        )
        selected.extend(ranked[:per_stratum])
    return selected


def select_duplicate_rows(
    selected: list[dict[str, Any]],
    *,
    count: int,
    seed: str,
) -> list[dict[str, Any]]:
    """Choose hidden duplicates across archives and source labels."""

    if count < 0 or count > len(selected):
        raise ValueError("duplicate count must be within selected unique rows")
    if count == 0:
        return []
    by_archive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_archive[_stratum(row)[0]].append(row)
    chosen: list[dict[str, Any]] = []
    for archive_index, archive in enumerate(sorted(by_archive)):
        if len(chosen) >= count:
            break
        labels = sorted({_stratum(row)[1] for row in by_archive[archive]})
        desired_label = labels[(archive_index + archive_index // 2) % len(labels)]
        candidates = [row for row in by_archive[archive] if _stratum(row)[1] == desired_label]
        chosen.append(
            min(
                candidates,
                key=lambda row: _rank(seed, f"duplicate:{archive}", str(row["sample_id"])),
            )
        )
    if len(chosen) < count:
        remaining = [row for row in selected if row not in chosen]
        remaining.sort(key=lambda row: _rank(seed, "duplicate:remainder", str(row["sample_id"])))
        chosen.extend(remaining[: count - len(chosen)])
    return chosen


def selected_source_set_sha256(selected: list[dict[str, Any]]) -> str:
    identities = sorted(f"{row['sample_id']}\t{row['sha256']}" for row in selected)
    return hashlib.sha256(("\n".join(identities) + "\n").encode("utf-8")).hexdigest()


def make_packet_plan(
    selected: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    *,
    seed: str,
) -> list[dict[str, Any]]:
    duplicate_ids = {str(row["sample_id"]) for row in duplicates}
    plan: list[dict[str, Any]] = []
    for row in selected:
        sample_id = str(row["sample_id"])
        occurrence_count = 2 if sample_id in duplicate_ids else 1
        source_group = _rank(seed, "source-group", sample_id)[:16]
        for occurrence in range(occurrence_count):
            reviewer_id = f"grr-{_rank(seed, 'packet', f'{sample_id}:{occurrence}')[:12]}"
            plan.append(
                {
                    "reviewer_id": reviewer_id,
                    "source_group": source_group,
                    "occurrence": occurrence,
                    "row": row,
                }
            )
    plan.sort(key=lambda item: _rank(seed, "blind-order", str(item["reviewer_id"])))
    return plan


def _write_clean_image(source_path: Path, destination: Path) -> None:
    with Image.open(source_path) as source:
        clean = ImageOps.exif_transpose(source).convert("RGB")
        clean.save(destination, format="PNG", optimize=True)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_measurement_artifacts(cache_dir: Path, packet_dir: Path) -> None:
    mapping = {
        "geometry_measurement_v2.json": "geometry_measurement_v2.json",
        "regions_overlay.png": "regions_overlay.png",
        "families_overlay.png": "local_families_overlay.png",
        "consistency_overlay.png": "consistency_overlay.png",
        "repeat_spacing_overlay.png": "repeat_spacing_overlay.png",
    }
    for source_name, destination_name in mapping.items():
        source = cache_dir / source_name
        if not source.is_file():
            raise ValueError(f"measurement artifact is missing: {source}")
        shutil.copyfile(source, packet_dir / destination_name)


def _find_forbidden_key(value: object, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in FORBIDDEN_BLIND_KEYS:
                return f"{path}.{key}"
            found = _find_forbidden_key(nested, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found = _find_forbidden_key(nested, f"{path}[{index}]")
            if found:
                return found
    return None


def assert_blind_payload(payload: object, forbidden_values: set[str]) -> None:
    forbidden_key = _find_forbidden_key(payload)
    if forbidden_key:
        raise ValueError(f"source field leaked into blind payload at {forbidden_key}")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    leaked = sorted(value for value in forbidden_values if value and value in serialized)
    if leaked:
        raise ValueError(f"source values leaked into blind payload: {leaked[:3]}")


def validate_registered_protocol(
    protocol_path: Path,
    *,
    project_root: Path,
    registry_path: Path,
    selected: list[dict[str, Any]],
    per_stratum: int,
    duplicate_count: int,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "demirror-geometry-semantic-relation-pilot-protocol-v1":
        raise ValueError("unexpected semantic-relation pilot protocol schema")
    if protocol.get("status") != "pre_registered_before_blind_packet_measurement":
        raise ValueError("protocol is not registered for packet measurement")
    if protocol.get("origin_scoring_authorized") is not False:
        raise ValueError("protocol must explicitly forbid origin scoring")
    cohort = dict(protocol.get("cohort", {}))
    expected = {
        "registry_sha256": _sha256(registry_path),
        "selected_source_set_sha256": selected_source_set_sha256(selected),
        "per_stratum": per_stratum,
        "hidden_duplicate_count": duplicate_count,
        "unique_source_count": len(selected),
        "packet_count": len(selected) + duplicate_count,
    }
    mismatches = {
        key: {"expected": value, "observed": cohort.get(key)}
        for key, value in expected.items()
        if cohort.get(key) != value
    }
    if mismatches:
        raise ValueError(f"protocol cohort closure failed: {mismatches}")
    for relative_path, expected_hash in dict(protocol.get("implementation_sha256", {})).items():
        observed_hash = _sha256(project_root / relative_path)
        if observed_hash != expected_hash:
            raise ValueError(f"registered implementation changed: {relative_path}")
    return protocol


def _build_readme(unique_count: int, packet_count: int) -> str:
    return f"""# 几何语义表面关系盲审包

本包包含 {unique_count} 张互不重复图片和隐藏重复项，共 {packet_count} 个任务。重复项不会在盲审清单中标出。

每个任务按以下顺序审核：

1. 查看 `image.png`，不要判断来源；
2. 参考 `regions_overlay.png` 与 `line_ids_overlay.png` 标出同一屋面、立面、道路或物体表面；
3. 每项最多复核 4 个全局族和 4 个局部族；拥挤时按 `review_packet.json` 中每个线族的 `detail_overlay` 单独查看线号；
4. 同方向但属于不同物体或表面的线必须拆开；
5. 完成时把 `status` 改为 `completed`。无法判断时使用 `unassessable` 并说明原因。

`review_manifest.jsonl` 只含匿名任务路径。来源标签、原文件名、重复映射和事后统计密钥不在本目录中。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument("--duplicates", type=int, default=4)
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output directory must be absent or empty; annotations are never overwritten")
    if not args.protocol.is_file():
        raise ValueError("registered protocol is missing")
    rows = _read_registry(args.registry)
    selected = select_unique_rows(rows, per_stratum=args.per_stratum, seed=args.selection_seed)
    duplicates = select_duplicate_rows(selected, count=args.duplicates, seed=args.selection_seed)
    plan = make_packet_plan(selected, duplicates, seed=args.selection_seed)
    validate_registered_protocol(
        args.protocol,
        project_root=args.project_root,
        registry_path=args.registry,
        selected=selected,
        per_stratum=args.per_stratum,
        duplicate_count=args.duplicates,
    )

    source_paths: dict[str, Path] = {}
    for row in selected:
        sample_id = str(row["sample_id"])
        source_path = args.project_root / str(row["relative_path"])
        if not source_path.is_file() or _sha256(source_path) != str(row["sha256"]):
            raise ValueError(f"source image is missing or changed: {sample_id}")
        source_paths[sample_id] = source_path

    blind_root = args.output_dir / "blind"
    packet_root = blind_root / "packets"
    posthoc_root = args.output_dir / "posthoc"
    cache_root = posthoc_root / "measurement_cache"
    packet_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    started = time.monotonic()
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    forbidden_values = {
        str(value)
        for row in selected
        for value in (
            row["sample_id"],
            row["relative_path"],
            Path(str(row["relative_path"])).name,
            *dict(row["declared_source_slice"]).values(),
        )
        if isinstance(value, (str, int, float))
    }

    for index, item in enumerate(plan, start=1):
        row = dict(item["row"])
        sample_id = str(row["sample_id"])
        reviewer_id = str(item["reviewer_id"])
        source_group = str(item["source_group"])
        source_path = source_paths[sample_id]
        cache_dir = cache_root / source_group
        cache_result = cache_dir / "geometry_measurement_v2.json"
        if not cache_result.is_file():
            measurement = assess_geometry_measurement_v2(source_path, output_dir=cache_dir)
        else:
            measurement = GeometryMeasurementV2Result.model_validate_json(
                cache_result.read_text(encoding="utf-8")
            )
        packet_dir = packet_root / reviewer_id
        packet_dir.mkdir(parents=True, exist_ok=False)
        image_path = packet_dir / "image.png"
        _write_clean_image(source_path, image_path)
        _copy_measurement_artifacts(cache_dir, packet_dir)
        write_relation_review_overlays(image_path, measurement, packet_dir)
        packet, annotation = build_review_packet(reviewer_id, measurement)
        packet_payload = packet.model_dump(mode="json")
        annotation_payload = annotation.model_dump(mode="json")
        assert_blind_payload(packet_payload, forbidden_values)
        assert_blind_payload(annotation_payload, forbidden_values)
        _write_json(packet_dir / "review_packet.json", packet_payload)
        _write_json(packet_dir / "annotation.json", annotation_payload)
        blind_rows.append(
            {
                "annotation": f"packets/{reviewer_id}/annotation.json",
                "packet": f"packets/{reviewer_id}/review_packet.json",
                "reviewer_id": reviewer_id,
            }
        )
        source = dict(row["declared_source_slice"])
        key_rows.append(
            {
                "reviewer_id": reviewer_id,
                "source_group": source_group,
                "hidden_duplicate": int(item["occurrence"]) > 0,
                "sample_id": sample_id,
                "split": row["split"],
                "original_relative_path": row["relative_path"],
                "original_sha256": row["sha256"],
                "anonymized_image_sha256": _sha256(image_path),
                "declared_source_slice": source,
            }
        )
        print(f"packet={index}/{len(plan)} reviewer_id={reviewer_id}", flush=True)

    assert_blind_payload(blind_rows, forbidden_values)
    _write_jsonl(blind_root / "review_manifest.jsonl", blind_rows)
    (blind_root / "README.md").write_text(
        _build_readme(len(selected), len(plan)), encoding="utf-8"
    )
    _write_jsonl(posthoc_root / "review_key.jsonl", key_rows)
    strata = Counter("/".join(_stratum(row)) for row in selected)
    report = {
        "schema_version": "geometry-semantic-relation-pilot-build-report-v1",
        "protocol_canonical_sha256": _canonical_json_sha256(args.protocol),
        "registry_sha256": _sha256(args.registry),
        "selection_seed_sha256": hashlib.sha256(args.selection_seed.encode("utf-8")).hexdigest(),
        "selected_source_set_sha256": selected_source_set_sha256(selected),
        "unique_source_count": len(selected),
        "packet_count": len(plan),
        "hidden_duplicate_count": len(duplicates),
        "selected_per_stratum": dict(sorted(strata.items())),
        "blind_payload_source_key_scan": "passed",
        "elapsed_seconds": time.monotonic() - started,
        "measurement_receives_source_label": False,
        "origin_scoring_authorized": False,
        "limitations": [
            "This is an annotation-feasibility pilot, not a source classifier evaluation.",
            "The source key must remain closed until all blind annotations are frozen.",
            "No geometry-v2 value may affect the web score from this pilot.",
        ],
    }
    _write_json(posthoc_root / "build_report.json", report)
    print(json.dumps({key: report[key] for key in ("unique_source_count", "packet_count", "hidden_duplicate_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
