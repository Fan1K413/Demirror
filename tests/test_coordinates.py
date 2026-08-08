import random

from image_trust.schemas import Point
from image_trust.utils.coordinates import CoordinateTransform, calculate_analysis_size


def test_calculate_analysis_size_only_downscales() -> None:
    assert calculate_analysis_size((800, 600), 1024) == (800, 600)
    assert calculate_analysis_size((4000, 2000), 1000) == (1000, 500)


def test_coordinate_round_trip_is_precise() -> None:
    transform = CoordinateTransform(
        encoded_size=(4000, 3000),
        canonical_size=(3000, 4000),
        analysis_size=(960, 1280),
        exif_orientation=6,
        orientation_applied=True,
    )
    point = Point(x=1234.5, y=2345.25)
    restored = transform.analysis_to_canonical(transform.canonical_to_analysis(point))
    assert abs(restored.x - point.x) < 1e-9
    assert abs(restored.y - point.y) < 1e-9
    assert transform.summary().encoded_to_canonical == "exif_orientation_6"


def test_coordinate_round_trip_covers_corners_center_and_random_points() -> None:
    transform = CoordinateTransform(
        encoded_size=(4321, 2719),
        canonical_size=(2719, 4321),
        analysis_size=calculate_analysis_size((2719, 4321), 1280),
        exif_orientation=8,
        orientation_applied=True,
    )
    width, height = transform.canonical_size
    points = [
        Point(x=0.0, y=0.0),
        Point(x=float(width - 1), y=0.0),
        Point(x=0.0, y=float(height - 1)),
        Point(x=float(width - 1), y=float(height - 1)),
        Point(x=width / 2.0, y=height / 2.0),
    ]
    rng = random.Random(123)
    points.extend(Point(x=rng.random() * width, y=rng.random() * height) for _ in range(100))
    for point in points:
        restored = transform.analysis_to_canonical(transform.canonical_to_analysis(point))
        assert abs(restored.x - point.x) <= 0.5
        assert abs(restored.y - point.y) <= 0.5
