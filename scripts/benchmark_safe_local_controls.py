"""Audit a high-threshold SAFE signal on local camera and ChatGPT controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from benchmark_safe_candidate import _metrics, _scores


THRESHOLD = 0.90


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    safe_source = root / "data/vendor/AIGIBench/detector_codes/SAFE-main"
    sys.path.insert(0, str(safe_source.resolve()))
    from models.resnet import resnet50

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checkpoint_path = root / "weights/aigibench-safe/SAFE-main/checkpoint-best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = resnet50(num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ai_paths: list[Path] = []
    for job in (root / ".demirror_web_jobs").iterdir():
        c2pa_path = job / "c2pa/c2pa_result.json"
        upload = next(job.glob("upload.*"), None)
        if not c2pa_path.exists() or upload is None:
            continue
        record = json.loads(c2pa_path.read_text(encoding="utf-8"))
        source_types = record.get("declared_digital_source_types") or []
        if record.get("signature_validation_status") == "valid" and any(
            value.rstrip("/").rsplit("/", 1)[-1].lower() == "trainedalgorithmicmedia"
            for value in source_types
        ):
            ai_paths.append(upload)
    ai_paths.sort()
    real_paths = sorted((root / "data/p0_f6_real_v2/images").glob("*"))
    paths = [*real_paths, *ai_paths]
    labels = np.asarray([0] * len(real_paths) + [1] * len(ai_paths), dtype=np.int64)
    scores = _scores(model, paths, 16, None)
    jpeg75_scores = _scores(model, paths, 16, 75)
    report = {
        "threshold": THRESHOLD,
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest(),
        "original": _metrics(labels, scores, THRESHOLD),
        "jpeg75": _metrics(labels, jpeg75_scores, THRESHOLD),
        "rows": [
            {
                "label": int(label),
                "path": str(path.relative_to(root)),
                "score": float(score),
                "jpeg75_score": float(jpeg_score),
            }
            for path, label, score, jpeg_score in zip(paths, labels, scores, jpeg75_scores)
        ],
        "scope": "Local ChatGPT-generated controls verified by C2PA versus ten camera-photo controls.",
    }
    output = root / "outputs/safe_local_controls_v1/report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
