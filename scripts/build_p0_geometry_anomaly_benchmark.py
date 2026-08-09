"""Build a controlled P0 geometry-anomaly benchmark and review registry.

The synthetic fixtures have exact procedural labels.  The separate PixArt and
SDXL queue is deliberately *unlabelled for geometry*: its declared source is
kept only for slice reporting and must never be used as a geometry label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


CANVAS = (640, 480)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _base_image(seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", CANVAS, (242, 244, 247))
    pixels = image.load()
    for y in range(CANVAS[1]):
        for x in range(CANVAS[0]):
            shade = int(3 * ((x / CANVAS[0]) - (y / CANVAS[1])) + rng.uniform(-1.0, 1.0))
            pixels[x, y] = (242 + shade, 244 + shade, 247 + shade)
    return image


def _draw_vanishing_fixture(seed: int, anomaly: bool) -> tuple[Image.Image, dict[str, object] | None]:
    rng = random.Random(seed)
    image = _base_image(seed)
    draw = ImageDraw.Draw(image)
    vp = (320 + rng.randint(-34, 34), 92 + rng.randint(-16, 16))
    bases = [55, 150, 245, 395, 490, 585]
    target_index = 4
    target: dict[str, object] | None = None
    for index, base_x in enumerate(bases):
        end = vp
        if anomaly and index == target_index:
            end = (vp[0] + 100 + rng.randint(-10, 10), vp[1] + 70 + rng.randint(-8, 8))
            target = {
                "kind": "vanishing_point_direction_outlier",
                "segment": {"p1": [base_x, 448], "p2": list(end)},
                "tolerance_px": 28,
            }
        draw.line([(base_x, 448), end], fill=(28, 46, 66), width=5)
        draw.line([(base_x + 7, 448), (end[0] + 5, end[1])], fill=(118, 135, 151), width=2)
    draw.line([(30, 448), (610, 448)], fill=(62, 76, 89), width=4)
    return image, target


def _draw_parallel_fixture(seed: int, anomaly: bool) -> tuple[Image.Image, dict[str, object] | None]:
    rng = random.Random(seed)
    image = _base_image(seed)
    draw = ImageDraw.Draw(image)
    angle_delta = rng.uniform(-0.8, 0.8)
    y_values = [112, 158, 204, 250, 296, 342]
    target_index = 3
    target: dict[str, object] | None = None
    for index, y in enumerate(y_values):
        end_y = y + angle_delta
        if anomaly and index == target_index:
            end_y = y + 50 + rng.randint(-4, 4)
            target = {
                "kind": "parallel_family_direction_outlier",
                "segment": {"p1": [78, y], "p2": [564, end_y]},
                "tolerance_px": 28,
            }
        draw.line([(78, y), (564, end_y)], fill=(35, 53, 72), width=5)
        draw.line([(78, y + 8), (564, end_y + 8)], fill=(138, 151, 163), width=2)
    return image, target


def _fixture_record(
    *,
    sample_id: str,
    relative_path: str,
    anomaly: bool,
    target: dict[str, object] | None,
    family: str,
    split: str,
    image_path: Path,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "relative_path": relative_path,
        "sha256": _sha256(image_path),
        "resolution": list(CANVAS),
        "geometry_annotation": {
            "status": "reference_procedural",
            "anomaly_present": anomaly,
            "anomaly_types": [target["kind"]] if target else [],
            "target_segments": [target] if target else [],
            "basis": "deterministic_procedural_fixture",
        },
        "fixture_family": family,
    }


def _build_fixtures(root: Path) -> list[dict[str, object]]:
    image_root = root / "fixtures"
    image_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    index = 0
    for family, renderer in (
        ("vanishing_point", _draw_vanishing_fixture),
        ("parallel_family", _draw_parallel_fixture),
    ):
        for anomaly in (False, True):
            for variation in range(12):
                index += 1
                split = "development" if variation < 6 else "holdout"
                sample_id = f"fixture-{family}-{'anomaly' if anomaly else 'clean'}-{variation + 1:02d}"
                image, target = renderer(20260809 + index * 101, anomaly)
                relative_path = f"fixtures/{sample_id}.png"
                image_path = root / relative_path
                image.save(image_path)
                records.append(
                    _fixture_record(
                        sample_id=sample_id,
                        relative_path=relative_path,
                        anomaly=anomaly,
                        target=target,
                        family=family,
                        split=split,
                        image_path=image_path,
                    )
                )
    return records


def _build_cross_generator_queue(project_root: Path, p3_registry_path: Path) -> list[dict[str, object]]:
    registry = json.loads(p3_registry_path.read_text(encoding="utf-8"))
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("P3 registry entries must be a list")
    queue: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("split") not in {"calibration", "external_test"}:
            continue
        image_path = Path(str(entry["relative_path"])).resolve()
        relative_path = image_path.relative_to(project_root.resolve()).as_posix()
        queue.append(
            {
                "sample_id": f"review-{entry['sample_id']}",
                "split": "development" if entry["split"] == "calibration" else "holdout",
                "relative_path": relative_path,
                "sha256": str(entry["sha256"]),
                "declared_source_slice": {
                    "archive_name": entry["archive_name"],
                    "generator_family": entry["generator_family"],
                    "label_name": entry["label_name"],
                    "must_not_be_used_as_geometry_label": True,
                },
                "geometry_annotation": {
                    "status": "pending_blinded_human_review",
                    "anomaly_present": None,
                    "anomaly_types": [],
                    "target_segments": [],
                    "basis": "requires_two_independent_reviewers_blinded_to_declared_source_slice",
                },
            }
        )
    if len(queue) != 160:
        raise ValueError(f"Expected 160 PixArt/SDXL review entries, found {len(queue)}")
    return sorted(queue, key=lambda entry: str(entry["sample_id"]))


def _protocol() -> str:
    return """# P0 几何异常标注协议 v1

目的：标注可见的**画面几何自洽性问题**，不判断图片是否由 AI 生成。

每张待审图片由两位独立标注者完成；标注界面不得展示 `declared_source_slice`。

可选结论：`present`、`absent`、`unassessable`。只有前两者一致时进入评测；不一致或不可评估的图片不参与准确率统计。

若为 `present`，需记录至少一个线段或矩形区域，并选择：

- `vanishing_point_direction_outlier`：同一应共点线族中有一条明显偏离；
- `parallel_family_direction_outlier`：局部重复平行结构中有一条明显偏离；
- `cross_structure_inconsistency`：相交/连接结构在局部无法同时成立；
- `other`：附简短说明。

不要把弧形边缘、反射、阴影、遮挡、鱼眼/全景、强透视本身或低分辨率压缩伪影标为几何错误。无足够直线支持时选 `unassessable`。

程序生成的 `fixtures/` 是唯一可直接作为真值的部分；`cross_generator_review_registry.jsonl` 必须经过盲审后才可用于几何准确率或阈值选择。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--p3-registry", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    fixtures = _build_fixtures(args.output_root)
    review_queue = _build_cross_generator_queue(args.project_root, args.p3_registry)
    _write_jsonl(args.output_root / "fixture_registry.jsonl", fixtures)
    _write_jsonl(args.output_root / "cross_generator_review_registry.jsonl", review_queue)
    _write_json(
        args.output_root / "manifest.json",
        {
            "schema_version": "p0-geometry-anomaly-benchmark-v1",
            "fixture_count": len(fixtures),
            "fixture_truth": "procedural_exact",
            "cross_generator_review_count": len(review_queue),
            "cross_generator_review_truth": "pending_blinded_human_review",
            "p3_registry_sha256": _sha256(args.p3_registry),
        },
    )
    (args.output_root / "ANNOTATION_PROTOCOL.md").write_text(_protocol(), encoding="utf-8")
    print(f"fixtures={len(fixtures)} review_queue={len(review_queue)} output={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
