from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from image_trust import cli, model_bootstrap
from image_trust.camera.config import load_camera_config
from image_trust.runtime_paths import runtime_cache_root, runtime_weights_root


def _asset(path: str, payload: bytes) -> model_bootstrap.DownloadAsset:
    return model_bootstrap.DownloadAsset(
        relative_path=model_bootstrap.PurePosixPath(path),
        url="https://example.invalid/asset",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def test_destination_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(model_bootstrap.ModelBootstrapError, match="unsafe_asset_path"):
        model_bootstrap._destination(tmp_path, model_bootstrap.PurePosixPath("../outside"))


def test_download_uses_verified_existing_asset_without_network(tmp_path: Path, monkeypatch) -> None:
    payload = b"verified model bytes"
    asset = _asset("detector/model.bin", payload)
    destination = tmp_path / "detector" / "model.bin"
    destination.parent.mkdir()
    destination.write_bytes(payload)

    monkeypatch.setattr(model_bootstrap, "urlopen", lambda *_args, **_kwargs: pytest.fail("network used"))

    assert model_bootstrap._download_verified(asset, destination, timeout_seconds=1) == "cached"


def test_install_manifest_skips_rehash_when_size_and_mtime_are_unchanged(tmp_path: Path) -> None:
    payload = b"verified model bytes"
    asset = _asset("detector/model.bin", payload)
    destination = tmp_path / "detector" / "model.bin"
    destination.parent.mkdir()
    destination.write_bytes(payload)
    entry = model_bootstrap._manifest_entry(destination, asset)
    model_bootstrap._write_install_manifest(tmp_path, {str(asset.relative_path): entry})

    loaded = model_bootstrap._load_install_manifest(tmp_path)

    assert model_bootstrap._matches_install_manifest(destination, asset, loaded[str(asset.relative_path)])
    destination.touch()
    assert not model_bootstrap._matches_install_manifest(destination, asset, loaded[str(asset.relative_path)])


def test_extract_dinov2_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with model_bootstrap.zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "no")

    with pytest.raises(model_bootstrap.ModelBootstrapError, match="path_traversal"):
        model_bootstrap._safe_extract_dinov2(archive, tmp_path / "hub" / "demirror_dinov2")


def test_extract_dinov2_writes_and_requires_the_verification_marker(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with model_bootstrap.zipfile.ZipFile(archive, "w") as output:
        output.writestr("dinov2-source/hubconf.py", "# source")
    target = tmp_path / "hub" / model_bootstrap.DINO_SOURCE_DIRECTORY

    assert model_bootstrap._safe_extract_dinov2(archive, target) == "downloaded"
    assert json.loads((target / model_bootstrap.DINO_SOURCE_MARKER).read_text(encoding="utf-8")) == {
        "archive_sha256": model_bootstrap.DINO_SOURCE_ARCHIVE.sha256,
        "revision": model_bootstrap.DINO_SOURCE_REVISION,
    }
    assert model_bootstrap._safe_extract_dinov2(archive, target) == "cached"


def test_dda_prefers_the_managed_dinov2_source_over_a_legacy_checkout(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    legacy = hub / "facebookresearch_dinov2_main"
    legacy.mkdir(parents=True)
    (legacy / "hubconf.py").write_text("# legacy", encoding="utf-8")
    managed = hub / model_bootstrap.DINO_SOURCE_DIRECTORY
    managed.mkdir()
    (managed / "hubconf.py").write_text("# managed", encoding="utf-8")

    class _Hub:
        @staticmethod
        def get_dir() -> str:
            return str(hub)

    class _Torch:
        hub = _Hub()

    from image_trust.ai_likelihood import dda

    assert dda._local_dinov2_source(_Torch()) == managed


def test_resolved_url_only_redirects_hugging_face(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMIRROR_HUGGINGFACE_ENDPOINT", "https://hf-mirror.example/")
    assert model_bootstrap._resolved_url("https://huggingface.co/org/model/resolve/revision/file") == (
        "https://hf-mirror.example/org/model/resolve/revision/file"
    )
    assert model_bootstrap._resolved_url("https://github.com/org/project/releases/download/v1/file") == (
        "https://github.com/org/project/releases/download/v1/file"
    )


def test_runtime_weights_root_honors_the_container_override(tmp_path: Path, monkeypatch) -> None:
    configured = tmp_path / "persistent-models"
    monkeypatch.setenv("DEMIRROR_WEIGHTS_ROOT", str(configured))

    assert runtime_weights_root(tmp_path / "project") == configured.resolve()
    assert runtime_cache_root(tmp_path / "project") == configured.resolve() / ".cache"


def test_camera_config_resolves_weights_from_the_runtime_root(tmp_path: Path, monkeypatch) -> None:
    configured = tmp_path / "persistent-models"
    monkeypatch.setenv("DEMIRROR_WEIGHTS_ROOT", str(configured))
    config = load_camera_config(Path("configs/p1_geocalib.yaml"))

    assert config.camera_backend.weights_path == str(configured / "geocalib-pinhole.tar")


def test_serve_bootstraps_before_starting_the_web_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class _Store:
        def close(self) -> None:
            pass

    class _Server:
        relation_review_store = None
        job_store = _Store()

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            pass

    def _bootstrap(**kwargs) -> list[tuple[str, str]]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(cli, "bootstrap_runtime_models", _bootstrap)
    monkeypatch.setattr(cli, "serve_local_demo", lambda *_args, **_kwargs: _Server())

    assert cli.main(["serve", "--port", "8766"]) == 0
    assert len(calls) == 1
    assert calls[0]["weights_root"] == runtime_weights_root(Path.cwd())
    assert calls[0]["cache_root"] == runtime_cache_root(Path.cwd())


def test_bootstrap_routes_torch_hub_into_the_requested_runtime_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights_root = tmp_path / "weights"
    cache_root = weights_root / ".cache"
    monkeypatch.setattr(model_bootstrap, "RUNTIME_ASSETS", ())
    monkeypatch.setattr(model_bootstrap, "is_verified_dinov2_source", lambda _path: True)

    model_bootstrap.bootstrap_runtime_models(weights_root=weights_root, cache_root=cache_root)

    assert Path(os.environ["TORCH_HOME"]) == cache_root / "torch"


def test_serve_skip_model_bootstrap_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Store:
        def close(self) -> None:
            pass

    class _Server:
        relation_review_store = None
        job_store = _Store()

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            pass

    monkeypatch.setattr(cli, "bootstrap_runtime_models", lambda **_kwargs: pytest.fail("bootstrap used"))
    monkeypatch.setattr(cli, "serve_local_demo", lambda *_args, **_kwargs: _Server())

    assert cli.main(["serve", "--skip-model-bootstrap"]) == 0
