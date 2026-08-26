"""Locations for large runtime assets outside the source tree."""

from __future__ import annotations

import os
from pathlib import Path


def runtime_weights_root(project_root: Path) -> Path:
    """Return the writable, persistent root containing detector weights.

    Source checkouts keep their existing ``weights/`` convention. A container
    may set ``DEMIRROR_WEIGHTS_ROOT`` to its writable bind-mounted ``weights``
    directory, so first-start setup never needs to write into the image
    filesystem.
    """

    configured = os.environ.get("DEMIRROR_WEIGHTS_ROOT", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()
    return (project_root / "weights").resolve()


def runtime_cache_root(project_root: Path) -> Path:
    """Return the writable cache root for runtime source files.

    It defaults below the model directory so container deployments need only
    one writable bind mount for weights, Torch Hub sources, and resumable
    downloads. ``DEMIRROR_RUNTIME_CACHE_ROOT`` remains available for an
    explicit alternative location.
    """

    configured = os.environ.get("DEMIRROR_RUNTIME_CACHE_ROOT", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()
    return runtime_weights_root(project_root) / ".cache"
