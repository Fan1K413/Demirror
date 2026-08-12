"""Run the deterministic source-neutral surface baseline for one blind packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from image_trust.geometry_ai.deterministic_surfaces import (
    DeterministicSurfaceArtifactManifest,
    export_deterministic_surface_diagnostics,
)
from image_trust.geometry_ai.relation_annotations import GeometryRelationReviewPacket


def run_packet_baseline(
    packet_path: Path,
    output_dir: Path,
) -> DeterministicSurfaceArtifactManifest:
    """Read one packet and only its declared anonymous image."""

    _reject_posthoc_path(packet_path, "packet")
    _reject_posthoc_path(output_dir, "output")
    packet = GeometryRelationReviewPacket.model_validate(
        json.loads(packet_path.read_text(encoding="utf-8-sig"))
    )
    image_path = _resolve_declared_asset(packet_path, packet.assets.image)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    _, manifest = export_deterministic_surface_diagnostics(
        image,
        packet,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_packet_baseline(args.packet, args.output_dir)
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
