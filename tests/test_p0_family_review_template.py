from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "p0_review_queue", REPOSITORY_ROOT / "scripts" / "evaluate_p0_cross_generator_review_queue.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_family_review_template_is_source_blind_and_matches_overlay_order() -> None:
    features = {
        "families": [
            {"family_id": "vp001", "stable": True, "scope": "global_vp", "member_line_ids": ["l1", "l2"]},
            {"family_id": "vp002", "stable": False, "scope": "global_vp", "member_line_ids": ["l3"]},
        ],
        "local_families": [
            {"family_id": "localdirection001", "stable": True, "scope": "local_image_plane_parallel", "member_line_ids": ["l4"]}
        ],
        "parallel_families": [
            {"family_id": "parallel001", "stable": True, "member_line_ids": ["l5"]}
        ],
    }

    template = module._family_review_template("p0r0001", features)

    assert template["source_label_visibility"] == "forbidden"
    assert template["reviewer_id"] == "p0r0001"
    assert "sample_id" not in template
    assert [entry["family_id"] for entry in template["families"]] == ["vp001", "localdirection001"]
    assert template["families"][0]["overlay_color_rgb"] == [70, 150, 255]
    assert template["families"][1]["overlay_color_rgb"] == [255, 185, 70]
    assert "declared_source_slice" not in str(template)
