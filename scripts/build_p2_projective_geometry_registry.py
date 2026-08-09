"""Build a deterministic P2 registry from a locally extracted official archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(pair_id: str) -> str:
    bucket = int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 2:
        return "test"
    if bucket < 4:
        return "calibration"
    return "train"


def build_registry(dataset_root: Path, source_archive: Path | None) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for class_name, label in (("real", 0), ("gen", 1)):
        paths = sorted(dataset_root.glob(f"**/test/{class_name}/*.jpg"), key=lambda path: int(path.stem))
        if not paths:
            raise ValueError(f"No test/{class_name}/*.jpg files found under {dataset_root}")
        for path in paths:
            pair_id = path.stem
            rows.append(
                {
                    "sample_id": f"projective_geometry_recent_deepfloyd_indoor_{class_name}_{pair_id}",
                    "relative_path": path.relative_to(dataset_root).as_posix(),
                    "label": label,
                    "label_name": "ai_generated" if label else "camera_photo",
                    "paired_scene_id": pair_id,
                    "split": _split(pair_id),
                    "sha256": _sha256(path),
                }
            )
    counts = Counter((str(row["split"]), str(row["label_name"])) for row in rows)
    archive = None
    if source_archive is not None:
        archive = {
            "filename": source_archive.name,
            "sha256": _sha256(source_archive),
            "size_bytes": source_archive.stat().st_size,
        }
    return {
        "schema_version": "p2-projective-geometry-registry-v1",
        "dataset_root": str(dataset_root.resolve()),
        "source": {
            "repository": "https://github.com/hanlinm2/projective-geometry/",
            "dataset": "https://huggingface.co/datasets/amitabh3/Projective-Geometry",
            "archive": archive,
            "subset": "Recent_Deepfloyd_Indoor/test/{real,gen}",
        },
        "split_protocol": "The same numeric paired_scene_id is always assigned to one split; SHA-256 deterministic bucket 0-1=test, 2-3=calibration, 4-9=train.",
        "class_counts": {f"{split}:{label}": count for (split, label), count in sorted(counts.items())},
        "entries": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    registry = build_registry(args.dataset_root, args.source_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"entries={len(registry['entries'])} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
