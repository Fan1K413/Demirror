"""Create a local, synthetic perspective image for the P0 CLI demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_input.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1280, 900), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    left_vp, right_vp = (210, 245), (1070, 245)
    for y in range(390, 880, 70):
        draw.line((left_vp[0], left_vp[1], 610, y), fill=(20, 20, 20), width=4)
        draw.line((right_vp[0], right_vp[1], 670, y), fill=(20, 20, 20), width=4)
    for x in range(250, 1060, 100):
        draw.line((x, 210, x, 870), fill=(45, 45, 45), width=4)
    draw.line((120, 850, 1160, 850), fill=(20, 20, 20), width=5)
    image.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

