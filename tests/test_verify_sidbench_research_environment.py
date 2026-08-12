from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_sidbench_research_environment",
    ROOT / "scripts" / "verify_sidbench_research_environment.py",
)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)


def _record(binding_path: str, binding_hash: str) -> dict[str, object]:
    return {
        "python": {"implementation": "cpython", "version": "3.10.6"},
        "packages": {"torch": "2.13.0+cpu", "torchvision": "0.28.0+cpu"},
        "frozen_bindings": [{"path": binding_path, "sha256": binding_hash}],
    }


def test_registered_environment_matches_current_checkout() -> None:
    record_path = ROOT / verify.DEFAULT_RECORD
    record = json.loads(record_path.read_text(encoding="utf-8"))

    errors = verify.verify_environment(record, ROOT)

    assert errors == []


def test_verifier_reports_packages_and_frozen_file_mismatches(tmp_path: Path) -> None:
    frozen = tmp_path / "evaluator.py"
    frozen.write_text("print('changed')\n", encoding="utf-8")

    def distribution_version(name: str) -> str:
        if name == "torch":
            return "2.12.0+cpu"
        raise importlib.metadata.PackageNotFoundError(name)

    errors = verify.verify_environment(
        _record("evaluator.py", "0" * 64),
        tmp_path,
        python_version="3.11.0",
        implementation="pypy",
        distribution_version=distribution_version,
    )

    assert any("Python version mismatch" in error for error in errors)
    assert any("Python implementation mismatch" in error for error in errors)
    assert any("torch version mismatch" in error for error in errors)
    assert any("Missing distribution torchvision" in error for error in errors)
    assert any("SHA-256 mismatch" in error for error in errors)


def test_verifier_rejects_binding_outside_repository(tmp_path: Path) -> None:
    errors = verify.verify_environment(
        _record("../outside.py", "0" * 64),
        tmp_path,
        python_version="3.10.6",
        implementation="cpython",
        distribution_version=lambda name: {
            "torch": "2.13.0+cpu",
            "torchvision": "0.28.0+cpu",
        }[name],
    )

    assert len(errors) == 1
    assert errors[0].startswith("Frozen binding escapes repository root:")
    assert errors[0].endswith("outside.py")
