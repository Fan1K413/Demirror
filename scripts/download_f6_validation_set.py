"""Download a traceable, local-only F6 real-photo validation set from Wikimedia.

This script is deliberately separate from P0 analysis and is never called by the
package or tests. Downloaded images and review logs live under ``data/`` and are
ignored by Git. Each source page is the licensing and attribution record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


SOURCES = (
    {
        "image_id": "f6_01_railway_perspective",
        "filename": "Railway perspective.jpg",
        "author": "Bromskloss",
        "license": "GFDL; CC BY-SA 3.0",
        "scene_tags": ["railway", "one_point_perspective"],
    },
    {
        "image_id": "f6_02_railroad_tracks",
        "filename": "Railroad-Tracks-Perspective.jpg",
        "author": "MikeMalak",
        "license": "Public domain (PD-self)",
        "scene_tags": ["railway", "one_point_perspective"],
    },
    {
        "image_id": "f6_03_tuileries_rivoli",
        "filename": "Tuileries Rivoli Perspective.jpg",
        "author": "OperaJoeGreen (see Commons source page)",
        "license": "CC0 1.0",
        "scene_tags": ["urban", "garden", "one_point_perspective"],
    },
    {
        "image_id": "f6_04_washington_square",
        "filename": "Vanishing Point (16297157742).jpg",
        "author": "Billie Grace Ward",
        "license": "CC BY 2.0",
        "scene_tags": ["urban", "architecture", "vanishing_point"],
    },
    {
        "image_id": "f6_05_56_leonard",
        "filename": "Vanishing point (43199413571).jpg",
        "author": "Billie Grace Ward",
        "license": "CC BY 2.0",
        "scene_tags": ["urban", "architecture", "vanishing_point"],
    },
    {
        "image_id": "f6_06_california_zephyr",
        "filename": "Vanishing point from the California Zephyr.jpg",
        "author": "Mackensen",
        "license": "CC BY-SA 4.0",
        "scene_tags": ["railway", "landscape", "vanishing_point"],
    },
    {
        "image_id": "f6_07_springthorpe_road",
        "filename": "Vanishing point - geograph.org.uk - 4765490.jpg",
        "author": "Steve Fareham",
        "license": "CC BY-SA 2.0",
        "scene_tags": ["road", "one_point_perspective"],
    },
    {
        "image_id": "f6_08_aldgate_alley",
        "filename": "Vanishing Point (8444970097).jpg",
        "author": "Duncan",
        "license": "CC BY 2.0",
        "scene_tags": ["urban", "alley", "vanishing_point"],
    },
    {
        "image_id": "f6_09_moscow_street",
        "filename": "Moscow, Preobrazhenskaya Street perspective (30572890213).jpg",
        "author": "Unknown; attribution is recorded on the Commons source page",
        "license": "CC BY 2.0",
        "scene_tags": ["street", "urban", "one_point_perspective"],
    },
    {
        "image_id": "f6_10_bukchon_street",
        "filename": "Bukchon-ro 11-gil street with hanok houses and blue sky in Bukchon Hanok Village Seoul.jpg",
        "author": "Basile Morin",
        "license": "CC BY-SA 4.0",
        "scene_tags": ["street", "architecture", "one_point_perspective"],
    },
)


def _page_url(filename: str) -> str:
    return f"https://commons.wikimedia.org/wiki/File:{quote(filename.replace(' ', '_'))}"


def _download_url(filename: str) -> str:
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(filename)}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(output: Path, timeout_seconds: int, delay_seconds: float) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source in SOURCES:
        filename = str(source["filename"])
        target = images_dir / f"{source['image_id']}{Path(filename).suffix.lower()}"
        if not target.is_file():
            request = Request(
                _download_url(filename),
                headers={"User-Agent": "Demirror-P0-validation/0.1 (local research fixture)"},
            )
            with urlopen(request, timeout=timeout_seconds) as response:
                target.write_bytes(response.read())
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        record = {
            **source,
            "relative_path": str(target.relative_to(output)).replace("\\", "/"),
            "source_url": _page_url(filename),
            "download_url": _download_url(filename),
            "original_file_hash": _sha256(target),
            "redistribution": "not committed; verify source-page license before any reuse",
            "privacy_and_redistribution_flag": "local_validation_only",
        }
        records.append(record)
    manifest = {
        "schema_version": "p0-f6-source-manifest-v1",
        "downloaded_on": date.today().isoformat(),
        "fixtures": records,
    }
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    review_path = output / "validation_log.json"
    if not review_path.exists():
        review_path.write_text(
            json.dumps(
                {
                    "schema_version": "p0-visual-review-v1",
                    "instructions": [
                        "Review lines_overlay.png for geometric-edge alignment.",
                        "Review anomalous_lines_overlay.png for stable-family mixing and false candidates.",
                        "Record a human reviewer, date, findings, and disposition for every image.",
                        "The screenshot field must list existing project-relative or absolute paths; separate multiple paths with semicolons.",
                        "Do not describe any P0 anomaly as an AI conclusion.",
                    ],
                    "reviews": [
                        {
                            "image_id": source["image_id"],
                            "reviewer": None,
                            "review_date": None,
                            "overlay_alignment": None,
                            "family_mixing_blocker": None,
                            "anomaly_gate_correct": None,
                            "screenshot": None,
                            "findings": None,
                            "disposition": "pending_human_review",
                        }
                        for source in SOURCES
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/p0_f6_real"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait after each network download (default: 1.0).",
    )
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be non-negative.")
    manifest_path = download(args.output, args.timeout, args.delay)
    print(f"Downloaded {len(SOURCES)} F6 sources and wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
