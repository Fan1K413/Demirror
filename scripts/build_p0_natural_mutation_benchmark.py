"""Build paired natural-background projective-line mutation fixtures for P0.

Each pair starts from a Projective Geometry *real* image.  The mutated partner
copies the local contrast profile of one long, stable parallel-family line to
a nearby line rotated by a fixed angle.  The matching clean partner is JPEG
re-encoded in exactly the same way.  The result is a controlled localization
benchmark with natural image clutter and exact target-line coordinates.

It measures only whether P0 can localize this deliberately injected
parallel-family violation.  It must not be used as an AI-origin benchmark:
both clean and mutated files derive from a real image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_trust.pipeline import analyze_image
from image_trust.utils.config import load_config


@dataclass(frozen=True)
class Mutation:
    family_id: str
    source_line_id: str
    target_start: np.ndarray
    target_end: np.ndarray
    core_bgr: tuple[int, int, int]
    side_bgr: tuple[int, int, int]


def baseline_is_eligible(result: Any, minimum_applicability: float) -> bool:
    """Require that the unmodified real image is already P0-measurable."""

    return (
        result.evidence.run_status.value == "ok"
        and result.evidence.applicability is not None
        and float(result.evidence.applicability) >= minimum_applicability
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_points(line: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    first = dict(line["p1_analysis"])
    second = dict(line["p2_analysis"])
    return (
        np.asarray([float(first["x"]), float(first["y"])], dtype=np.float64),
        np.asarray([float(second["x"]), float(second["y"])], dtype=np.float64),
    )


def _inside(points: list[np.ndarray], width: int, height: int, margin: float) -> bool:
    return all(
        margin <= point[0] < width - margin and margin <= point[1] < height - margin
        for point in points
    )


def choose_mutation(
    image: np.ndarray,
    lines: list[dict[str, Any]],
    families: list[dict[str, Any]],
    *,
    rotation_deg: float,
    offset_px: float,
    minimum_contrast: float,
) -> Mutation | None:
    """Select one long stable line and construct an in-bounds rotated duplicate."""

    height, width = image.shape[:2]
    line_by_id = {str(line["line_id"]): line for line in lines}
    angle = math.radians(rotation_deg)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    choices: list[tuple[float, Mutation]] = []
    for family in families:
        if not bool(family.get("stable")) or family.get("direction_analysis") is None:
            continue
        for line_id in family.get("member_line_ids", []):
            line = line_by_id.get(str(line_id))
            if line is None:
                continue
            start, end = _line_points(line)
            source_length = float(np.linalg.norm(end - start))
            if source_length < 52.0:
                continue
            axis = (end - start) / source_length
            normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
            midpoint = (start + end) / 2.0
            midpoint_int = np.round(midpoint).astype(int)
            plus_int = np.round(midpoint + normal * 4.0).astype(int)
            minus_int = np.round(midpoint - normal * 4.0).astype(int)
            if not _inside(
                [midpoint_int.astype(float), plus_int.astype(float), minus_int.astype(float)],
                width,
                height,
                2.0,
            ):
                continue
            core = image[midpoint_int[1], midpoint_int[0]].astype(np.float64)
            side_options = [
                image[plus_int[1], plus_int[0]].astype(np.float64),
                image[minus_int[1], minus_int[0]].astype(np.float64),
            ]
            side = max(side_options, key=lambda color: float(np.linalg.norm(color - core)))
            contrast = float(np.linalg.norm(side - core))
            if contrast < minimum_contrast:
                continue
            target_length = min(70.0, max(46.0, source_length * 0.60))
            target_axis = rotation @ axis
            target_midpoint = midpoint + normal * offset_px
            target_start = target_midpoint - target_axis * target_length / 2.0
            target_end = target_midpoint + target_axis * target_length / 2.0
            if not _inside([target_start, target_end], width, height, 3.0):
                continue
            choices.append(
                (
                    min(source_length, 100.0) + contrast,
                    Mutation(
                        family_id=str(family["family_id"]),
                        source_line_id=str(line_id),
                        target_start=target_start,
                        target_end=target_end,
                        core_bgr=tuple(int(value) for value in core),
                        side_bgr=tuple(int(value) for value in side),
                    ),
                )
            )
    return max(choices, key=lambda item: item[0])[1] if choices else None


def apply_mutation(image: np.ndarray, mutation: Mutation) -> np.ndarray:
    """Draw a two-width edge matching the selected source line's local contrast."""

    output = image.copy()
    start = tuple(int(value) for value in np.round(mutation.target_start))
    end = tuple(int(value) for value in np.round(mutation.target_end))
    cv2.line(output, start, end, mutation.side_bgr, thickness=5, lineType=cv2.LINE_AA)
    cv2.line(output, start, end, mutation.core_bgr, thickness=2, lineType=cv2.LINE_AA)
    return output


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write fixture image: {path}")


def bounded_source_paths(source_root: Path, minimum_id: int, maximum_id: int, limit: int) -> list[Path]:
    """Return numerically ordered images within one predeclared identifier range."""

    if minimum_id > maximum_id:
        raise ValueError("minimum source ID must not exceed maximum source ID")
    selected = [
        path
        for path in source_root.glob("*.jpg")
        if path.stem.isdigit() and minimum_id <= int(path.stem) <= maximum_id
    ]
    return sorted(selected, key=lambda path: int(path.stem))[:limit]


def _build_split(
    *,
    role: str,
    source_root: Path,
    count: int,
    output_root: Path,
    config_path: Path,
    rotation_deg: float,
    offset_px: float,
    minimum_contrast: float,
    maximum_sources: int,
    minimum_source_id: int,
    maximum_source_id: int,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    selected: list[dict[str, Any]] = []
    candidates = bounded_source_paths(
        source_root, minimum_source_id, maximum_source_id, maximum_sources
    )
    if not candidates:
        raise ValueError(f"No JPEG sources found: {source_root}")
    for path in candidates:
        if len(selected) >= count:
            break
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scratch = output_root / "_selection_artifacts" / role / path.stem
        result = analyze_image(path, config, scratch)
        if not baseline_is_eligible(result, config.applicability.anomaly_min_applicability):
            shutil.rmtree(scratch, ignore_errors=True)
            continue
        lines = json.loads((scratch / "lines.json").read_text(encoding="utf-8"))
        mutation = choose_mutation(
            image,
            lines,
            list(result.evidence.features.get("parallel_families", [])),
            rotation_deg=rotation_deg,
            offset_px=offset_px,
            minimum_contrast=minimum_contrast,
        )
        shutil.rmtree(scratch, ignore_errors=True)
        if mutation is None:
            continue
        index = len(selected) + 1
        base_name = f"{role}_{index:02d}_{path.stem}"
        clean_relative = Path("images") / f"{base_name}_clean.jpg"
        mutated_relative = Path("images") / f"{base_name}_mutated.jpg"
        clean_path = output_root / clean_relative
        mutated_path = output_root / mutated_relative
        _write_jpeg(clean_path, image)
        # The final clean control, not merely the uncompressed source, must be
        # measurable.  This checks no mutated pixels and therefore preserves
        # the holdout mutation as an untouched localization evaluation.
        clean_scratch = output_root / "_selection_artifacts" / role / f"{path.stem}_clean"
        clean_result = analyze_image(clean_path, config, clean_scratch)
        shutil.rmtree(clean_scratch, ignore_errors=True)
        if not baseline_is_eligible(clean_result, config.applicability.anomaly_min_applicability):
            clean_path.unlink(missing_ok=True)
            continue
        _write_jpeg(mutated_path, apply_mutation(image, mutation))
        selected.append(
            {
                "sample_id": f"natural-mutation-{role}-{index:02d}",
                "role": role,
                "source_relative_path": str(path).replace("\\", "/"),
                "source_sha256": _sha256(path),
                "clean_relative_path": str(clean_relative).replace("\\", "/"),
                "mutated_relative_path": str(mutated_relative).replace("\\", "/"),
                "family_id": mutation.family_id,
                "source_line_id": mutation.source_line_id,
                "target_segment": {
                    "p1": [float(value) for value in mutation.target_start],
                    "p2": [float(value) for value in mutation.target_end],
                    "tolerance_px": 7.0,
                },
                "mutation": {
                    "rotation_deg": rotation_deg,
                    "offset_px": offset_px,
                    "core_bgr": list(mutation.core_bgr),
                    "side_bgr": list(mutation.side_bgr),
                    "jpeg_quality": 95,
                },
            }
        )
        print(f"built={role}:{index}/{count} source={path.name}", flush=True)
    if len(selected) != count:
        raise RuntimeError(f"Could only construct {len(selected)} of {count} {role} fixtures")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count-per-split", type=int, default=12)
    parser.add_argument("--rotation-deg", type=float, default=22.0)
    parser.add_argument("--offset-px", type=float, default=13.0)
    parser.add_argument("--minimum-contrast", type=float, default=60.0)
    parser.add_argument("--maximum-sources", type=int, default=100)
    parser.add_argument("--development-min-id", type=int, default=351)
    parser.add_argument("--development-max-id", type=int, default=425)
    parser.add_argument("--holdout-min-id", type=int, default=426)
    parser.add_argument("--holdout-max-id", type=int, default=500)
    args = parser.parse_args()
    if args.count_per_split <= 0 or args.maximum_sources <= 0:
        raise ValueError("fixture counts must be positive")
    if not 8.0 <= abs(args.rotation_deg) <= 45.0:
        raise ValueError("rotation-deg magnitude must be within [8, 45]")
    if args.output_root.exists():
        raise ValueError(f"Output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    try:
        development = _build_split(
            role="development",
            source_root=args.development_root,
            count=args.count_per_split,
            output_root=args.output_root,
            config_path=args.config,
            rotation_deg=args.rotation_deg,
            offset_px=args.offset_px,
            minimum_contrast=args.minimum_contrast,
            maximum_sources=args.maximum_sources,
            minimum_source_id=args.development_min_id,
            maximum_source_id=args.development_max_id,
        )
        holdout = _build_split(
            role="holdout",
            source_root=args.holdout_root,
            count=args.count_per_split,
            output_root=args.output_root,
            config_path=args.config,
            rotation_deg=-args.rotation_deg,
            offset_px=args.offset_px,
            minimum_contrast=args.minimum_contrast,
            maximum_sources=args.maximum_sources,
            minimum_source_id=args.holdout_min_id,
            maximum_source_id=args.holdout_max_id,
        )
        manifest = {
            "schema_version": "p0-natural-line-mutation-benchmark-v1",
            "purpose": "Natural-background injected projective-line localization benchmark; not AI-origin classification.",
            "source_data": "Projective Geometry real images, source paths and SHA-256 are listed per fixture.",
            "protocol": {
                "development": f"PixArt real IDs {args.development_min_id}-{args.development_max_id}; used only for implementation development.",
                "holdout": f"SDXL real IDs {args.holdout_min_id}-{args.holdout_max_id}; not used for implementation changes or threshold selection.",
                "paired_control": "Clean and mutated images are both JPEG quality 95 from the same real source.",
            },
            "records": [*development, *holdout],
        }
        (args.output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"manifest={args.output_root / 'manifest.json'}")
        return 0
    except Exception:
        shutil.rmtree(args.output_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
