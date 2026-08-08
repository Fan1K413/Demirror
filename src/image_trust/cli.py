"""Command-line entry point for P0."""

from __future__ import annotations

import argparse
from pathlib import Path

from image_trust.pipeline import analyze_image
from image_trust.schemas import RunStatus
from image_trust.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-trust",
        description="P0 projection-geometry measurement baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Analyze one local image.")
    analyze.add_argument("input", type=Path, help="PNG, JPEG, or static WebP input.")
    analyze.add_argument(
        "--config",
        type=Path,
        required=True,
        help="P0 YAML configuration.",
    )
    analyze.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Artifact directory to create or update.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "analyze":
        return 2
    config = load_config(args.config)
    result = analyze_image(args.input, config, args.output)
    print(
        f"run_status={result.evidence.run_status.value} "
        f"observation={result.evidence.observation.value} "
        f"output={args.output}"
    )
    return 0 if result.evidence.run_status in {RunStatus.OK, RunStatus.NOT_APPLICABLE} else 2


if __name__ == "__main__":
    raise SystemExit(main())

