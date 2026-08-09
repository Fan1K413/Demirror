"""Resumably fetch and verify the official non-commercial B-Free checkpoint.

The publisher's server transfers long responses very slowly in this
environment.  This helper uses small HTTP range requests, verifies every
completed chunk length before retaining it, and verifies the publisher's MD5
before extracting anything.  It intentionally defaults to one small batch so
the caller can monitor network use and stop safely between batches.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


URL = "https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip"
TOTAL_BYTES = 321_653_488
EXPECTED_MD5 = "f3f53fa647848b16cf81c913f148a198"
DEFAULT_CHUNK_BYTES = 1_048_576


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    @property
    def filename(self) -> str:
        return f"{self.start:012d}-{self.end:012d}.part"


def iter_ranges(total_bytes: int, chunk_bytes: int) -> tuple[ByteRange, ...]:
    if total_bytes < 1 or chunk_bytes < 1:
        raise ValueError("total_bytes and chunk_bytes must both be positive")
    return tuple(
        ByteRange(start, min(total_bytes - 1, start + chunk_bytes - 1))
        for start in range(0, total_bytes, chunk_bytes)
    )


def _is_complete(path: Path, expected_size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == expected_size
    except OSError:
        return False


def _fetch_one(byte_range: ByteRange, directory: Path, timeout_seconds: int, retry_attempts: int) -> str:
    destination = directory / byte_range.filename
    if _is_complete(destination, byte_range.size):
        return "cached"
    temporary = destination.with_suffix(".downloading")
    last_error: Exception | None = None
    for attempt in range(1, retry_attempts + 1):
        request = urllib.request.Request(URL, headers={"Range": f"bytes={byte_range.start}-{byte_range.end}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status}")
                content_range = response.headers.get("Content-Range", "")
                expected_range = f"bytes {byte_range.start}-{byte_range.end}/{TOTAL_BYTES}"
                if content_range != expected_range:
                    raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=128 * 1024)
            if not _is_complete(temporary, byte_range.size):
                actual = temporary.stat().st_size if temporary.exists() else 0
                raise RuntimeError(f"incomplete range {byte_range.start}-{byte_range.end}: {actual}/{byte_range.size}")
            os.replace(temporary, destination)
            return "downloaded"
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            print(f"retry={attempt}/{retry_attempts}: {byte_range.start}-{byte_range.end}: {type(error).__name__}", flush=True)
    assert last_error is not None
    raise RuntimeError(f"range failed after {retry_attempts} attempts: {byte_range.start}-{byte_range.end}") from last_error


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        archive.extractall(destination)


def finalize(parts_dir: Path, destination: Path, ranges: tuple[ByteRange, ...]) -> Path:
    missing = [item.filename for item in ranges if not _is_complete(parts_dir / item.filename, item.size)]
    if missing:
        raise RuntimeError(f"cannot finalize; {len(missing)} verified chunks are missing")
    archive_path = destination / "BFREE_dino2reg4.zip"
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        for item in ranges:
            with (parts_dir / item.filename).open("rb") as source:
                shutil.copyfileobj(source, temporary, length=1024 * 1024)
    actual_md5 = _md5(temporary_path)
    if actual_md5 != EXPECTED_MD5:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"official MD5 mismatch: expected {EXPECTED_MD5}, got {actual_md5}")
    os.replace(temporary_path, archive_path)
    _safe_extract(archive_path, destination)
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("weights/b-free"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--finalize", action="store_true", help="Combine/extract only after every chunk is present.")
    args = parser.parse_args()
    if args.workers < 1 or args.max_batches < 0 or args.timeout_seconds < 1 or args.retry_attempts < 1:
        raise ValueError("workers, timeout, and retry-attempts must be positive; max-batches may be zero")

    destination = args.destination.resolve()
    parts_dir = destination / "bfree-download-parts-v1"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = iter_ranges(TOTAL_BYTES, DEFAULT_CHUNK_BYTES)
    incomplete = [item for item in ranges if not _is_complete(parts_dir / item.filename, item.size)]
    batches = [incomplete[index : index + args.workers] for index in range(0, len(incomplete), args.workers)]
    selected = batches[: args.max_batches]
    downloaded = cached = 0
    failures: list[str] = []
    for batch_index, batch in enumerate(selected, start=1):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_fetch_one, item, parts_dir, args.timeout_seconds, args.retry_attempts): item
                for item in batch
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    outcome = future.result()
                except RuntimeError:
                    failures.append(item.filename)
                    print(f"failed: {item.start}-{item.end}", flush=True)
                    continue
                downloaded += outcome == "downloaded"
                cached += outcome == "cached"
                print(f"{outcome}: {item.start}-{item.end}", flush=True)
        print(f"batch={batch_index}/{len(selected)} complete", flush=True)

    verified = sum(_is_complete(parts_dir / item.filename, item.size) for item in ranges)
    print(
        f"verified_chunks={verified}/{len(ranges)} downloaded={downloaded} cached={cached} failures={len(failures)}",
        flush=True,
    )
    if failures:
        print("failed_chunks=" + ",".join(failures), flush=True)
        return 2
    if args.finalize:
        archive = finalize(parts_dir, destination, ranges)
        print(f"verified_archive={archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
