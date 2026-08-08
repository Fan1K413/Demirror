import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_generated_fixture_manifest_and_evaluation_workflow(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    generated = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "generate_p0_fixtures.py"),
            "--output",
            str(fixture_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "manifest" in generated.stdout.lower()
    manifest_path = fixture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = [fixture["group"] for fixture in manifest["fixtures"]]
    assert len(manifest["fixtures"]) >= 36
    assert groups.count("F1_file_gate") >= 12
    assert groups.count("F2_single_vp") >= 6
    assert groups.count("F3_multi_vp") >= 6
    assert groups.count("F4_low_geometry") >= 8
    assert groups.count("F5_special_imaging") >= 4
    assert all(len(fixture["sha256"]) == 64 for fixture in manifest["fixtures"])

    evaluation_dir = tmp_path / "evaluation"
    evaluated = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate_p0.py"),
            str(fixture_dir),
            "--config",
            str(REPOSITORY_ROOT / "configs" / "p0.yaml"),
            "--output",
            str(evaluation_dir),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(evaluated.stdout)
    assert summary["expectation_mismatch_count"] == 0
    assert summary["image_count"] == len(manifest["fixtures"])
    assert (evaluation_dir / "evaluation.jsonl").is_file()
    assert (evaluation_dir / "summary.json").is_file()
