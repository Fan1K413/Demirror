"""Audit GeoCalib measurements from labeled local web-job controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


AI_SLUG = "trainedalgorithmicmedia"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def slug(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1].lower()


def feature(camera: dict[str, object]) -> dict[str, float]:
    full = camera.get("full_image") if isinstance(camera.get("full_image"), dict) else {}
    crops = camera.get("crops") if isinstance(camera.get("crops"), list) else []
    estimates = [
        crop["estimate"]
        for crop in crops
        if isinstance(crop, dict)
        and isinstance(crop.get("estimate"), dict)
        and crop["estimate"].get("status") == "ok"
    ]
    def values(key: str) -> np.ndarray:
        return np.asarray(
            [float(estimate[key]) for estimate in estimates if estimate.get(key) is not None],
            dtype=np.float64,
        )
    roll = values("roll")
    pitch = values("pitch")
    vfov = np.asarray(
        [
            float(estimate["vfov_or_focal"]["value"])
            for estimate in estimates
            if isinstance(estimate.get("vfov_or_focal"), dict)
            and estimate["vfov_or_focal"].get("kind") == "vfov_deg"
        ],
        dtype=np.float64,
    )
    e_cam = camera.get("e_cam") if isinstance(camera.get("e_cam"), dict) else {}
    uncertainty = full.get("uncertainty") if isinstance(full.get("uncertainty"), dict) else {}
    return {
        "e_cam": float(e_cam["value"]) if e_cam.get("value") is not None else float("nan"),
        "full_uncertainty": float(uncertainty["overall"]) if uncertainty.get("overall") is not None else 1.0,
        "qualified_crop_ratio": len(e_cam.get("qualified_crop_ids", [])) / max(len(crops), 1),
        "roll_std": float(roll.std()) if len(roll) else float("nan"),
        "pitch_std": float(pitch.std()) if len(pitch) else float("nan"),
        "vfov_cv": float(vfov.std() / max(vfov.mean(), 1e-6)) if len(vfov) else float("nan"),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    real_hashes = {digest(path) for path in (root / "data/p0_f6_real_v2/images").glob("*")}
    rows: list[dict[str, object]] = []
    for job in (root / ".demirror_web_jobs").iterdir():
        upload = next(job.glob("upload.*"), None)
        camera_path = job / "camera/camera_result.json"
        c2pa_path = job / "c2pa/c2pa_result.json"
        if upload is None or not camera_path.exists() or not c2pa_path.exists():
            continue
        c2pa = json.loads(c2pa_path.read_text(encoding="utf-8"))
        source_types = c2pa.get("declared_digital_source_types") or []
        if c2pa.get("signature_validation_status") == "valid" and any(
            slug(value) == AI_SLUG for value in source_types
        ):
            label = 1
        elif digest(upload) in real_hashes:
            label = 0
        else:
            continue
        rows.append(
            {
                "job": job.name,
                "label": label,
                **feature(json.loads(camera_path.read_text(encoding="utf-8"))),
            }
        )
    report: dict[str, object] = {"count": len(rows), "rows": rows, "features": {}}
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    for name in ("e_cam", "full_uncertainty", "qualified_crop_ratio", "roll_std", "pitch_std", "vfov_cv"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        valid = np.isfinite(values)
        if valid.sum() and len(np.unique(labels[valid])) == 2:
            report["features"][name] = {
                "available": int(valid.sum()),
                "real_mean": float(values[valid & (labels == 0)].mean()),
                "ai_mean": float(values[valid & (labels == 1)].mean()),
                "ai_direction_auc": float(roc_auc_score(labels[valid], values[valid])),
            }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
