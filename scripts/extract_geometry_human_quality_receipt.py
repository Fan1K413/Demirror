"""Extract a source-neutral quality receipt from the completed human pilot audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from image_trust.geometry_ai.surface_comparison import (
    HumanRelationQualityReceipt,
    extract_human_quality_receipt,
)


def extract_receipt(
    input_path: Path,
    output_path: Path,
) -> HumanRelationQualityReceipt:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("quality receipt must not overwrite its input audit")
    pilot_audit = json.loads(input_path.read_text(encoding="utf-8-sig"))
    receipt = extract_human_quality_receipt(
        pilot_audit,
        input_audit_sha256=_sha256(input_path),
    )
    _atomic_write_json(output_path, receipt.model_dump(mode="json"))
    return receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = extract_receipt(args.pilot_audit, args.output)
    print(json.dumps({"passed": receipt.passed}, sort_keys=True))
    return 0 if receipt.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
