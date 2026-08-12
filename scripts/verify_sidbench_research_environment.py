"""Verify the pinned CPU environment and frozen files for the SIDBench replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Callable, Mapping


DEFAULT_RECORD = Path(
    "research/records/2026-08-12/pixel/"
    "sidbench_patchcraft_npr_environment_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_environment(
    record: Mapping[str, Any],
    repo_root: Path,
    *,
    python_version: str | None = None,
    implementation: str | None = None,
    distribution_version: Callable[[str], str] = importlib.metadata.version,
) -> list[str]:
    """Return every mismatch instead of failing after the first one."""

    errors: list[str] = []
    expected_python = record.get("python")
    if not isinstance(expected_python, Mapping):
        return ["environment record has no python object"]
    actual_python = python_version or platform.python_version()
    actual_implementation = implementation or platform.python_implementation().lower()
    if actual_python != str(expected_python.get("version")):
        errors.append(
            f"Python version mismatch: expected {expected_python.get('version')}, "
            f"got {actual_python}"
        )
    if actual_implementation != str(expected_python.get("implementation")):
        errors.append(
            "Python implementation mismatch: expected "
            f"{expected_python.get('implementation')}, got {actual_implementation}"
        )

    packages = record.get("packages")
    if not isinstance(packages, Mapping):
        errors.append("environment record has no packages object")
    else:
        for distribution, expected in packages.items():
            try:
                actual = distribution_version(str(distribution))
            except importlib.metadata.PackageNotFoundError:
                errors.append(
                    f"Missing distribution {distribution}; install "
                    "requirements-research-sidbench.lock"
                )
                continue
            if actual != str(expected):
                errors.append(
                    f"{distribution} version mismatch: expected {expected}, got {actual}"
                )

    bindings = record.get("frozen_bindings")
    if not isinstance(bindings, list) or not bindings:
        errors.append("environment record has no frozen_bindings")
    else:
        resolved_root = repo_root.resolve()
        for binding in bindings:
            if not isinstance(binding, Mapping):
                errors.append("invalid frozen binding entry")
                continue
            relative = Path(str(binding.get("path", "")))
            path = (resolved_root / relative).resolve()
            try:
                path.relative_to(resolved_root)
            except ValueError:
                errors.append(f"Frozen binding escapes repository root: {relative}")
                continue
            if not path.is_file():
                errors.append(f"Missing frozen file: {relative.as_posix()}")
                continue
            actual_hash = _sha256(path)
            expected_hash = str(binding.get("sha256", "")).lower()
            if actual_hash != expected_hash:
                errors.append(
                    f"SHA-256 mismatch for {relative.as_posix()}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    record_path = args.record
    if not record_path.is_absolute():
        record_path = repo_root / record_path
    record = _read_record(record_path.resolve())
    errors = verify_environment(record, repo_root)
    if errors:
        print("SIDBench research environment verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("SIDBench research environment verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
