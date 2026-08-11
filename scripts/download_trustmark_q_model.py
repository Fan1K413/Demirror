"""Download and verify the optional official TrustMark Q ONNX decoder.

This is an explicit setup command.  Demirror never calls it while analyzing an
image or starting the local server.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

from image_trust.watermark.trustmark import (
    TRUSTMARK_Q_MODEL_PATH,
    TRUSTMARK_Q_MODEL_SHA256,
)


MODEL_URL = "https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_Q.onnx"
MODEL_SIZE_BYTES = 47_401_222


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(destination: Path) -> str:
    """Download to a temporary file, enforce size/hash, then replace atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == TRUSTMARK_Q_MODEL_SHA256:
        return "already_verified"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="decoder_Q-",
            suffix=".onnx.part",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(MODEL_URL, headers={"User-Agent": "Demirror model setup"})
            with urlopen(request, timeout=30) as response:
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None and int(declared_length) != MODEL_SIZE_BYTES:
                    raise ValueError("TrustMark Q download size header mismatch")
                written = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > MODEL_SIZE_BYTES:
                        raise ValueError("TrustMark Q download exceeded pinned size")
                    temporary.write(block)
        if temporary_path.stat().st_size != MODEL_SIZE_BYTES:
            raise ValueError("TrustMark Q download size mismatch")
        if _sha256(temporary_path) != TRUSTMARK_Q_MODEL_SHA256:
            raise ValueError("TrustMark Q download SHA-256 mismatch")
        os.replace(temporary_path, destination)
        temporary_path = None
        return "downloaded_and_verified"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm the Adobe TrustMark MIT license recorded in THIRD_PARTY_NOTICES.md",
    )
    parser.add_argument("--output", type=Path, default=TRUSTMARK_Q_MODEL_PATH)
    args = parser.parse_args()
    if not args.accept_license:
        parser.error("review THIRD_PARTY_NOTICES.md, then pass --accept-license")
    status = download(args.output.resolve())
    print(f"TrustMark Q model: {status}: {args.output.resolve()}")
    print(f"SHA-256: {TRUSTMARK_Q_MODEL_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
