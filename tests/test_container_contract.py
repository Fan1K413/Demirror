from __future__ import annotations

from pathlib import Path

import yaml

from image_trust.cli import build_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_non_loopback_cli_bind_requires_an_explicit_opt_in() -> None:
    parser = build_parser()

    default_args = parser.parse_args(["serve", "--host", "0.0.0.0"])
    container_args = parser.parse_args(
        ["serve", "--host", "0.0.0.0", "--allow-non-loopback"]
    )

    assert default_args.allow_non_loopback is False
    assert container_args.allow_non_loopback is True


def test_compose_keeps_host_binding_local_and_container_restricted() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["demirror"]

    assert service["ports"] == [
        "${DEMIRROR_BIND_ADDRESS:-127.0.0.1}:${DEMIRROR_PORT:-8765}:8765"
    ]
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert "--allow-non-loopback" in service["command"]
    assert "0.0.0.0" in service["command"]
    assert "./weights:/app/weights:ro" in service["volumes"]
    assert "./data:/app/data:ro" in service["volumes"]


def test_dockerfile_uses_non_root_runtime_and_external_assets() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER demirror" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/api/health" in dockerfile
    assert 'ENTRYPOINT ["image-trust"]' in dockerfile
    assert "ARG TARGETARCH" in dockerfile
    assert 'if [ "${TARGETARCH}" = "arm64" ]' in dockerfile
    assert "build-essential" in dockerfile
    assert "apt-get purge --yes --auto-remove build-essential" in dockerfile
    assert "COPY weights" not in dockerfile
    assert "COPY data" not in dockerfile


def test_docker_workflow_parallelizes_architecture_tests_and_only_publishes_trusted_events() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/docker-image.yml").read_text(
        encoding="utf-8"
    )

    assert "pull_request:" in workflow
    assert "packages: write" in workflow
    assert "ghcr.io/${GITHUB_REPOSITORY,,}" in workflow
    assert "build-test:" in workflow
    assert "strategy:" in workflow
    assert "matrix:" in workflow
    assert "arch: amd64" in workflow
    assert "arch: arm64" in workflow
    assert "name: Build and test (${{ matrix.arch }})" in workflow
    assert "needs:\n      - prepare\n      - build-test" in workflow
    assert "docker/setup-qemu-action@v4" in workflow
    assert "if: matrix.arch == 'arm64'" in workflow
    assert "Build ${{ matrix.arch }} smoke-test image" in workflow
    assert "Smoke test ${{ matrix.arch }} health endpoint" in workflow
    assert "--platform '${{ matrix.platform }}'" in workflow
    assert "platforms: linux/amd64,linux/arm64" in workflow
    assert "scope=demirror-amd64" in workflow
    assert "scope=demirror-arm64" in workflow
    assert 'DOCKER_BUILD_RECORD_UPLOAD: "false"' in workflow
    assert (
        "publish:\n"
        "    needs:\n"
        "      - prepare\n"
        "      - build-test\n"
        "    if: github.event_name != 'pull_request'"
    ) in workflow
    assert "push: true" in workflow
    assert "provenance: mode=max" in workflow
    assert "sbom: true" in workflow


def test_dockerfile_keeps_readme_out_of_dependency_cache_layer() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    dependency_copy = dockerfile.split("RUN --mount=type=cache", maxsplit=1)[0]
    assert "README.md" not in dependency_copy
    assert "COPY README.md ./" in dockerfile
