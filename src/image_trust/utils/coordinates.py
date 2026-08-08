"""Coordinate mapping between canonical and resized analysis images."""

from __future__ import annotations

from dataclasses import dataclass

from image_trust.schemas import CoordinateTransformSummary, Point


@dataclass(frozen=True)
class CoordinateTransform:
    encoded_size: tuple[int, int]
    canonical_size: tuple[int, int]
    analysis_size: tuple[int, int]
    exif_orientation: int
    orientation_applied: bool

    @property
    def scale_x(self) -> float:
        return self.analysis_size[0] / self.canonical_size[0]

    @property
    def scale_y(self) -> float:
        return self.analysis_size[1] / self.canonical_size[1]

    def canonical_to_analysis(self, point: Point) -> Point:
        return Point(x=point.x * self.scale_x, y=point.y * self.scale_y)

    def analysis_to_canonical(self, point: Point) -> Point:
        return Point(x=point.x / self.scale_x, y=point.y / self.scale_y)

    def summary(self) -> CoordinateTransformSummary:
        transform_name = (
            "identity"
            if self.exif_orientation == 1
            else f"exif_orientation_{self.exif_orientation}"
        )
        return CoordinateTransformSummary(
            encoded_size=self.encoded_size,
            canonical_size=self.canonical_size,
            analysis_size=self.analysis_size,
            exif_orientation=self.exif_orientation,
            orientation_applied=self.orientation_applied,
            encoded_to_canonical=transform_name,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
        )


def calculate_analysis_size(
    canonical_size: tuple[int, int],
    max_long_side: int,
) -> tuple[int, int]:
    width, height = canonical_size
    longest = max(width, height)
    if longest <= max_long_side:
        return canonical_size
    scale = max_long_side / longest
    return (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )

