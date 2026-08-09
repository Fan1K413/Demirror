"""Convert one trusted official DeepLSD pickle checkpoint into safetensors.

The official project publishes ``deeplsd_md.tar`` as a PyTorch checkpoint that
contains OmegaConf metadata, so PyTorch's safe weights-only reader cannot load
it directly.  This one-time utility must be used only with a trusted official
download.  It extracts *only* the ``model`` tensor state dict, writes a safe
artifact, then prints hashes of both assets.  P0 runtime accepts only the
output safetensors file and never reads the pickle checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, output: Path) -> dict[str, str | int]:
    if not source.is_file():
        raise FileNotFoundError(f"Official DeepLSD checkpoint does not exist: {source}")
    if output.suffix.lower() != ".safetensors":
        raise ValueError("output must have the .safetensors suffix")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Trusted checkpoint has no non-empty model state dict")
    tensors = {
        str(name): value.detach().contiguous().cpu()
        for name, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    if len(tensors) != len(state_dict):
        raise ValueError("Trusted checkpoint model state contains a non-tensor entry")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"source_sha256": _sha256(source)})
    return {
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "tensor_count": len(tensors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("weights/deeplsd/deeplsd_md.tar"))
    parser.add_argument("--output", type=Path, default=Path("weights/deeplsd/deeplsd_md.safetensors"))
    args = parser.parse_args()
    print(convert(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
