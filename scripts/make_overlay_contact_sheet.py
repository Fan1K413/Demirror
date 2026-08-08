"""Build a compact contact sheet for manual P0 overlay review."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--artifact", default="anomalous_lines_overlay.png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--cell-height", type=int, default=300)
    args = parser.parse_args()
    if args.columns < 1 or args.cell_width < 80 or args.cell_height < 80:
        parser.error("columns, cell-width, and cell-height must be positive usable values.")
    paths = sorted(path for path in args.artifact_root.glob(f"*/{args.artifact}") if path.is_file())
    if not paths:
        parser.error(f"No '{args.artifact}' images found below {args.artifact_root}")

    label_height = 26
    rows = (len(paths) + args.columns - 1) // args.columns
    sheet = Image.new(
        "RGB",
        (args.columns * args.cell_width, rows * args.cell_height),
        (28, 28, 28),
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((args.cell_width, args.cell_height - label_height), Image.Resampling.LANCZOS)
        col = index % args.columns
        row = index // args.columns
        x = col * args.cell_width + (args.cell_width - image.width) // 2
        y = row * args.cell_height + label_height + (args.cell_height - label_height - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((col * args.cell_width + 8, row * args.cell_height + 6), path.parent.name, fill=(255, 255, 255))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Wrote contact sheet: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
