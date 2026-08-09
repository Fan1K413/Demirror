from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_trust.geometry_ai.features import feature_names, relationship_features
from image_trust.geometry_ai.inference import assess_geometry_ai


def test_relationship_features_are_permutation_invariant() -> None:
    lines = np.asarray(
        [
            [10, 10, 190, 20],
            [10, 40, 190, 50],
            [20, 10, 30, 190],
            [80, 10, 90, 190],
        ],
        dtype=float,
    )
    first = relationship_features(lines, (200, 200))
    second = relationship_features(lines[[2, 0, 3, 1]], (200, 200))
    assert list(first) == feature_names()
    assert list(first) == list(second)
    assert np.allclose(list(first.values()), list(second.values()))


def test_low_line_image_is_not_applicable_with_installed_model(tmp_path: Path) -> None:
    # The model may not exist in a fresh source checkout.  When installed, a
    # blank image must be gated rather than interpreted as evidence either way.
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (256, 256), "white").save(image_path)
    result = assess_geometry_ai(image_path)
    assert result.status in {"unavailable", "not_applicable"}
    if result.status == "not_applicable":
        assert result.probability is None
        assert result.risk_band == "unknown"


def test_repeated_parallel_lines_produce_relationship_support() -> None:
    lines = np.asarray([[10, y, 190, y + 3] for y in range(20, 181, 20)], dtype=float)
    features = relationship_features(lines, (200, 200))
    assert features["orientation_top1_weight"] > 0.9
    assert features["near_parallel_pair_ratio"] > 0.9
