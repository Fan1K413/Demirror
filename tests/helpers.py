from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def write_structured_image(path: Path, size: tuple[int, int] = (640, 480)) -> None:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    vp_left = (110, 155)
    vp_right = (540, 155)
    for y in (230, 275, 320, 365, 410, 455):
        draw.line((vp_left[0], vp_left[1], 300, y), fill=(25, 25, 25), width=3)
        draw.line((vp_right[0], vp_right[1], 340, y), fill=(25, 25, 25), width=3)
    for x in (130, 200, 270, 370, 440, 510):
        draw.line((x, 120, x, 455), fill=(50, 50, 50), width=3)
    draw.line((80, 440, 560, 440), fill=(20, 20, 20), width=4)
    image.save(path)


def write_solid_image(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    Image.new("RGB", size, (120, 120, 120)).save(path)

