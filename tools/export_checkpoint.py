#!/usr/bin/env python3
"""Export a portable inference checkpoint without development-only paths.

The output retains model tensors exactly, validates them with strict loading,
and stores only public vEMDestripe metadata. Optimizer/scheduler state remains
in the source checkpoint and is intentionally excluded from the release file.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from vemdestripe.model import VEMDestripe


def state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract and normalize a model state dictionary."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dictionary, got {type(checkpoint).__name__}")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a non-empty model state dictionary")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="training checkpoint to export")
    parser.add_argument("output", type=Path, help="portable checkpoint path")
    parser.add_argument("--num-reference", type=int, default=16)
    parser.add_argument("--reference-block-size", type=int, default=16)
    args = parser.parse_args()

    try:
        source = torch.load(args.source, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument.
        source = torch.load(args.source, map_location="cpu")
    state = state_dict_from_checkpoint(source)
    VEMDestripe().load_state_dict(state, strict=True)

    public = {
        "method_name": "vEMDestripe",
        "checkpoint_schema": 1,
        "model_state_dict": state,
        "reference_preprocess": "block_mean",
        "reference_block_size": int(args.reference_block_size),
        "num_reference": int(args.num_reference),
        "source_epoch": source.get("epoch") if isinstance(source, dict) else None,
        "source_best_val_loss": source.get("best_val_loss") if isinstance(source, dict) else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public, args.output)
    print(f"exported: {args.output}")
    print(f"parameters: {sum(t.numel() for t in state.values())}")
    print(f"sha256: {sha256(args.output)}")


if __name__ == "__main__":
    main()

