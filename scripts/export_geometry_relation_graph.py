"""Export one finalized, source-blind geometry relation graph and overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from image_trust.geometry_ai.relation_annotations import (
    GeometryRelationAnnotation,
    GeometryRelationReviewPacket,
)
from image_trust.geometry_ai.relation_graph import (
    RelationGraphArtifactManifest,
    export_relation_graph_diagnostics,
)


def export_packet_graph(
    packet_path: Path,
    annotation_path: Path,
    output_dir: Path,
) -> RelationGraphArtifactManifest:
    """Load only declared blind inputs and export their relationship graph."""

    _reject_posthoc_path(packet_path, "packet")
    _reject_posthoc_path(annotation_path, "annotation")
    _reject_posthoc_path(output_dir, "output")
    packet = GeometryRelationReviewPacket.model_validate(
        json.loads(packet_path.read_text(encoding="utf-8-sig"))
    )
    annotation = GeometryRelationAnnotation.model_validate(
        json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    )
    image_path = _resolve_declared_asset(packet_path, packet.assets.image)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    _, manifest = export_relation_graph_diagnostics(
        image,
        packet,
        annotation,
        output_dir,
    )
    return manifest


def _resolve_declared_asset(packet_path: Path, declared_path: str) -> Path:
    packet_root = packet_path.resolve().parent
    relative = Path(declared_path)
    if relative.is_absolute():
        raise ValueError("packet image asset must be a relative path")
    resolved = (packet_root / relative).resolve()
    try:
        resolved.relative_to(packet_root)
    except ValueError as error:
        raise ValueError("packet image asset escapes the packet directory") from error
    if not resolved.is_file():
        raise ValueError(f"packet image asset is missing: {declared_path}")
    _reject_posthoc_path(resolved, "packet image")
    return resolved


def _reject_posthoc_path(path: Path, label: str) -> None:
    if any(part.casefold() == "posthoc" for part in path.resolve().parts):
        raise ValueError(f"{label} path must not enter a posthoc directory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_packet_graph(args.packet, args.annotation, args.output_dir)
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
