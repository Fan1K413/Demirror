from __future__ import annotations

from pathlib import Path

import pytest

from image_trust.ai_likelihood.community_forensics import (
    CommunityForensicsUnavailableError,
    score_community_forensics_isolated,
)


def test_missing_model_files_are_reported_without_starting_a_worker(tmp_path: Path) -> None:
    with pytest.raises(CommunityForensicsUnavailableError, match="config_not_available"):
        score_community_forensics_isolated(tmp_path / "asset.jpg", model_root=tmp_path)
