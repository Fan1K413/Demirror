"""Run the production-isolated DDA path on local AI and camera controls."""

from __future__ import annotations

import json
from pathlib import Path

from image_trust.ai_likelihood.dda import DDA_HIGH_CONFIDENCE_THRESHOLD, score_dda_isolated


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    controls = [
        ("ai_chatgpt_1", 1, root / ".demirror_web_jobs/0cab2ef8692e493c8f41b776744de99c/upload.png"),
        ("ai_chatgpt_2", 1, root / ".demirror_web_jobs/11ff13cf688b4abdaa565fc6eb0666fe/upload.png"),
        ("ai_chatgpt_3", 1, root / ".demirror_web_jobs/73de54ecdef34ad7bcdcb71a5ef74e13/upload.png"),
        ("real_railway", 0, root / "data/p0_f6_real_v2/images/f6_01_railway_perspective.jpg"),
        ("real_tuileries", 0, root / "data/p0_f6_real_v2/images/f6_03_tuileries_rivoli.jpg"),
        ("real_bukchon", 0, root / "data/p0_f6_real_v2/images/f6_10_bukchon_street.jpg"),
    ]
    rows: list[dict[str, object]] = []
    for name, label, path in controls:
        scored = score_dda_isolated(path)
        row = {
            "name": name,
            "label": label,
            "path": str(path.relative_to(root)),
            "score": scored.score,
            "detected": scored.score >= DDA_HIGH_CONFIDENCE_THRESHOLD,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    generated = [row for row in rows if row["label"] == 1]
    real = [row for row in rows if row["label"] == 0]
    report = {
        "threshold": DDA_HIGH_CONFIDENCE_THRESHOLD,
        "generated_recall": sum(bool(row["detected"]) for row in generated) / len(generated),
        "real_false_positive_rate": sum(bool(row["detected"]) for row in real) / len(real),
        "rows": rows,
    }
    output = root / "outputs/dda_local_smoke_v1/report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
