from __future__ import annotations

from pathlib import Path

import pytest

from image_trust.ai_likelihood.nonescape import (
    NonescapeMiniUnavailableError,
    score_nonescape_mini_isolated,
)


def test_missing_checkpoint_is_reported_without_starting_a_worker(tmp_path: Path) -> None:
    input_path = tmp_path / "asset.jpg"
    input_path.write_bytes(b"not-an-image")

    with pytest.raises(NonescapeMiniUnavailableError, match="checkpoint_not_available"):
        score_nonescape_mini_isolated(input_path, checkpoint_path=tmp_path / "missing.safetensors")
