"""Verified first-start installation of the runtime detector assets.

Large checkpoints deliberately stay outside the source distribution.  The
server calls this module before accepting uploads: it installs a fixed,
audited asset set in persistent storage, resumes incomplete downloads, and
checks every completed file's byte size and SHA-256 digest.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from image_trust.runtime_paths import runtime_cache_root, runtime_weights_root


class ModelBootstrapError(RuntimeError):
    """A declared runtime asset could not be installed and verified."""


@dataclass(frozen=True)
class DownloadAsset:
    """A byte-exact upstream asset that may be installed automatically."""

    relative_path: PurePosixPath
    url: str
    sha256: str
    size_bytes: int


ProgressReporter = Callable[[str], None]


# These revisions and hashes are the versions registered in the local detector
# audits. Never replace a URL with an unpinned ``main`` reference.
RUNTIME_ASSETS: tuple[DownloadAsset, ...] = (
    DownloadAsset(
        PurePosixPath("dda-v1/DDA_ckpt.pth"),
        "https://huggingface.co/Junwei-Xi/Dual-Data-Alignment/resolve/"
        "4390d9023899196b437480bb6a441915ef5d816c/DDA_ckpt.pth",
        "b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e",
        1_255_621_296,
    ),
    DownloadAsset(
        PurePosixPath("aigibench-safe/SAFE-main/checkpoint-best.pth"),
        "https://huggingface.co/HorizonTEL/AIGIBench/resolve/"
        "b74a6ab18d54bc029508ff88ca55f0beff8acc7a/SAFE-main/checkpoint-best.pth",
        "e168c6e6e4c3fa8f381fbebdb1a8299b3a3ad59ee7e8e697ba46a73752d4ffda",
        17_415_410,
    ),
    DownloadAsset(
        PurePosixPath("community-forensics-224/config.json"),
        "https://huggingface.co/OwensLab/commfor-model-224/resolve/"
        "26afc31e6b40c312c3fd42c05a758be62446215b/config.json",
        "9826208c73a17bb20c85d6624aa16b4a118508d89ff2c86d71e4b553474ff702",
        116,
    ),
    DownloadAsset(
        PurePosixPath("community-forensics-224/model.safetensors"),
        "https://huggingface.co/OwensLab/commfor-model-224/resolve/"
        "26afc31e6b40c312c3fd42c05a758be62446215b/model.safetensors",
        "a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0",
        86_678_644,
    ),
    DownloadAsset(
        PurePosixPath("wkaandemir-ai-detector/config.json"),
        "https://huggingface.co/wkaandemir/ai-image-detector/resolve/"
        "fefa013737a0c3477961d36ee8dbbdc751352366/config.json",
        "31d3678632a70c3b2ce8a62ad55dbdabf3390278d859e209aac9eefd2495ee5f",
        1_123,
    ),
    DownloadAsset(
        PurePosixPath("wkaandemir-ai-detector/model.safetensors"),
        "https://huggingface.co/wkaandemir/ai-image-detector/resolve/"
        "fefa013737a0c3477961d36ee8dbbdc751352366/model.safetensors",
        "41ce93c6c206a4f3929e19cf9b43b663c63a47422ab27a9bbb67757db5f42339",
        343_399_284,
    ),
    DownloadAsset(
        PurePosixPath("nonescape/nonescape-mini-v0.safetensors"),
        "https://huggingface.co/e3ntity/nonescape-v0/resolve/"
        "dd9d70c10cd0f6823e2af87d553217c25fe00b3d/nonescape-mini-v0.safetensors",
        "7a0d0740c813ce199bc32ed16a5f4f4915895c4c9fdee0a98bdbeedd4f3631fd",
        86_666_672,
    ),
    DownloadAsset(
        PurePosixPath("trustmark/Q/decoder_Q.onnx"),
        "https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_Q.onnx",
        "ee3268f057c9dabef680e169302f5973d0589feea86189ed229a896cc3aa88df",
        47_401_222,
    ),
    DownloadAsset(
        PurePosixPath("geocalib-pinhole.tar"),
        "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
        "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac",
        116_074_121,
    ),
)

DINO_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_SOURCE_ARCHIVE = DownloadAsset(
    PurePosixPath("torch/hub/.demirror-dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8.zip"),
    f"https://github.com/facebookresearch/dinov2/archive/{DINO_SOURCE_REVISION}.zip",
    "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48",
    3_001_681,
)
# This is a private Torch Hub-compatible directory. Its ``z`` prefix makes the
# existing DDA adapter's deterministic reverse-sort choose this checked source
# over an older ``facebookresearch_dinov2_main`` checkout without changing that
# audited adapter.
DINO_SOURCE_DIRECTORY = "facebookresearch_dinov2_zdemirror_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_SOURCE_MARKER = ".demirror-bootstrap.json"
INSTALL_MANIFEST_NAME = ".demirror-runtime-assets-v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _destination(root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ModelBootstrapError(f"unsafe_asset_path:{relative_path}")
    resolved_root = root.resolve()
    target = (resolved_root / Path(relative_path)).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as error:
        raise ModelBootstrapError(f"asset_path_escapes_root:{relative_path}") from error
    return target


def _is_verified(path: Path, asset: DownloadAsset) -> bool:
    return path.is_file() and path.stat().st_size == asset.size_bytes and _sha256(path) == asset.sha256


def _load_install_manifest(weights_root: Path) -> dict[str, dict[str, object]]:
    """Read the local installation record without trusting arbitrary fields."""

    manifest_path = weights_root / INSTALL_MANIFEST_NAME
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    assets = raw.get("assets") if isinstance(raw, dict) else None
    if not isinstance(assets, dict):
        return {}
    return {key: value for key, value in assets.items() if isinstance(key, str) and isinstance(value, dict)}


def _matches_install_manifest(destination: Path, asset: DownloadAsset, entry: dict[str, object] | None) -> bool:
    if entry is None or not destination.is_file():
        return False
    stat = destination.stat()
    return (
        entry.get("sha256") == asset.sha256
        and entry.get("size_bytes") == asset.size_bytes
        and entry.get("mtime_ns") == stat.st_mtime_ns
        and stat.st_size == asset.size_bytes
    )


def _write_install_manifest(weights_root: Path, entries: dict[str, dict[str, object]]) -> None:
    weights_root.mkdir(parents=True, exist_ok=True)
    target = weights_root / INSTALL_MANIFEST_NAME
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".demirror-runtime-assets-",
        suffix=".tmp",
        dir=weights_root,
        delete=False,
    ) as temporary:
        json.dump({"assets": entries, "version": 1}, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, target)


def _manifest_entry(destination: Path, asset: DownloadAsset) -> dict[str, object]:
    return {
        "mtime_ns": destination.stat().st_mtime_ns,
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
    }


def _report(reporter: ProgressReporter | None, message: str) -> None:
    if reporter is not None:
        reporter(message)


def _download_verified(
    asset: DownloadAsset,
    destination: Path,
    *,
    timeout_seconds: int,
    reporter: ProgressReporter | None = None,
) -> str:
    """Resume one download when possible, then atomically publish it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_verified(destination, asset):
        _report(reporter, f"model_bootstrap=cached:{asset.relative_path}")
        return "cached"

    partial = destination.with_name(f"{destination.name}.part")
    if partial.is_file() and partial.stat().st_size > asset.size_bytes:
        partial.unlink()
    if partial.is_file() and partial.stat().st_size == asset.size_bytes:
        if _is_verified(partial, asset):
            os.replace(partial, destination)
            _report(reporter, f"model_bootstrap=resumed:{asset.relative_path}")
            return "resumed"
        partial.unlink()

    existing_size = partial.stat().st_size if partial.is_file() else 0
    _report(
        reporter,
        f"model_bootstrap=downloading:{asset.relative_path}:resume_bytes={existing_size}:total_bytes={asset.size_bytes}",
    )
    headers = {"User-Agent": "Demirror-runtime-bootstrap/2"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
    request = Request(_resolved_url(asset.url), headers=headers)
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except (HTTPError, URLError, OSError) as error:
        raise ModelBootstrapError(f"download_request_failed:{asset.relative_path}:{type(error).__name__}") from error

    with response:
        status = response.getcode()
        append = existing_size > 0 and status == 206
        if status not in {200, 206}:
            raise ModelBootstrapError(f"download_http_status:{asset.relative_path}:{status}")
        if existing_size and not append:
            existing_size = 0
        mode = "ab" if append else "wb"
        last_percent = (existing_size * 100) // asset.size_bytes
        with partial.open(mode) as stream:
            total = existing_size
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > asset.size_bytes:
                    raise ModelBootstrapError(f"download_exceeds_expected_size:{asset.relative_path}")
                stream.write(block)
                percent = (total * 100) // asset.size_bytes
                if percent >= last_percent + 5:
                    last_percent = percent
                    _report(reporter, f"model_bootstrap=progress:{asset.relative_path}:{percent}%")

    if not _is_verified(partial, asset):
        actual_size = partial.stat().st_size if partial.is_file() else 0
        raise ModelBootstrapError(
            f"download_verification_failed:{asset.relative_path}:size={actual_size}:expected={asset.size_bytes}"
        )
    os.replace(partial, destination)
    outcome = "downloaded" if existing_size == 0 else "resumed"
    _report(reporter, f"model_bootstrap={outcome}:{asset.relative_path}")
    return outcome


def _resolved_url(url: str) -> str:
    """Optionally route Hugging Face assets through a user-selected endpoint."""

    endpoint = os.environ.get("DEMIRROR_HUGGINGFACE_ENDPOINT", "").strip().rstrip("/")
    prefix = "https://huggingface.co"
    if endpoint and url.startswith(prefix + "/"):
        return endpoint + url.removeprefix(prefix)
    return url


def is_verified_dinov2_source(target_directory: Path) -> bool:
    marker_path = target_directory / DINO_SOURCE_MARKER
    if not (target_directory / "hubconf.py").is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return marker == {"archive_sha256": DINO_SOURCE_ARCHIVE.sha256, "revision": DINO_SOURCE_REVISION}


def _safe_extract_dinov2(archive_path: Path, target_directory: Path) -> str:
    if is_verified_dinov2_source(target_directory):
        return "cached"
    if target_directory.exists():
        raise ModelBootstrapError(f"dinov2_target_incomplete:{target_directory}")
    target_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="demirror-dinov2-", dir=target_directory.parent) as temporary:
        temporary_root = Path(temporary).resolve()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    candidate = (temporary_root / member.filename).resolve()
                    try:
                        candidate.relative_to(temporary_root)
                    except ValueError as error:
                        raise ModelBootstrapError("dinov2_archive_path_traversal") from error
                archive.extractall(temporary_root)
        except (OSError, zipfile.BadZipFile) as error:
            raise ModelBootstrapError(f"dinov2_archive_extract_failed:{type(error).__name__}") from error
        roots = [path for path in temporary_root.iterdir() if path.is_dir()]
        if len(roots) != 1 or not (roots[0] / "hubconf.py").is_file():
            raise ModelBootstrapError("dinov2_archive_layout_invalid")
        (roots[0] / DINO_SOURCE_MARKER).write_text(
            json.dumps(
                {"archive_sha256": DINO_SOURCE_ARCHIVE.sha256, "revision": DINO_SOURCE_REVISION},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(roots[0], target_directory)
    return "downloaded"


def bootstrap_runtime_models(
    *,
    weights_root: Path,
    cache_root: Path,
    timeout_seconds: int = 60,
    reporter: ProgressReporter | None = None,
    verify_existing: bool = False,
) -> list[tuple[str, str]]:
    """Install the runtime asset set and return each asset's outcome."""

    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    weights_root = weights_root.resolve()
    cache_root = cache_root.resolve()
    # DDA resolves its verified local DINOv2 checkout through
    # ``torch.hub.get_dir()``. Keep that directory beside the fixed detector
    # assets instead of the container user's read-only home cache.
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    manifest_entries = {} if verify_existing else _load_install_manifest(weights_root)
    installed_entries: dict[str, dict[str, object]] = {}
    outcomes: list[tuple[str, str]] = []
    for asset in RUNTIME_ASSETS:
        destination = _destination(weights_root, asset.relative_path)
        asset_key = str(asset.relative_path)
        if not verify_existing and _matches_install_manifest(destination, asset, manifest_entries.get(asset_key)):
            outcome = "cached"
            _report(reporter, f"model_bootstrap=cached:{asset.relative_path}")
        else:
            outcome = _download_verified(
                asset,
                destination,
                timeout_seconds=timeout_seconds,
                reporter=reporter,
            )
        installed_entries[asset_key] = _manifest_entry(destination, asset)
        outcomes.append((str(asset.relative_path), outcome))
    _write_install_manifest(weights_root, installed_entries)

    source_directory = cache_root / "torch" / "hub" / DINO_SOURCE_DIRECTORY
    if is_verified_dinov2_source(source_directory):
        source_outcome = "cached"
    else:
        archive_path = _destination(cache_root, DINO_SOURCE_ARCHIVE.relative_path)
        _download_verified(
            DINO_SOURCE_ARCHIVE,
            archive_path,
            timeout_seconds=timeout_seconds,
            reporter=reporter,
        )
        source_outcome = _safe_extract_dinov2(archive_path, source_directory)
        archive_path.unlink(missing_ok=True)
    _report(reporter, f"model_bootstrap={source_outcome}:dinov2-source")
    outcomes.append(("dinov2-source", source_outcome))
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=runtime_weights_root(Path.cwd()))
    parser.add_argument("--cache-root", type=Path, default=runtime_cache_root(Path.cwd()))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Recalculate SHA-256 for all installed assets instead of trusting the local installation record.",
    )
    parser.add_argument(
        "--accept-trustmark-license",
        action="store_true",
        help="Confirm the TrustMark license recorded in THIRD_PARTY_NOTICES.md.",
    )
    args = parser.parse_args(argv)
    if not args.accept_trustmark_license:
        parser.error("review THIRD_PARTY_NOTICES.md, then pass --accept-trustmark-license")
    try:
        bootstrap_runtime_models(
            weights_root=args.weights_root,
            cache_root=args.cache_root,
            timeout_seconds=args.timeout_seconds,
            reporter=print,
            verify_existing=args.verify,
        )
    except (ModelBootstrapError, OSError, ValueError) as error:
        print(f"model_bootstrap_failed={error}", file=sys.stderr, flush=True)
        return 2
    print("model_bootstrap=complete", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - covered through the CLI entry point
    raise SystemExit(main())
