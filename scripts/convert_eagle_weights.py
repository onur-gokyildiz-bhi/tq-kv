#!/usr/bin/env python3
"""One-time conversion: EAGLE pytorch_model.bin → safetensors.

The EAGLE draft weights shipped by SafeAILab (e.g. yuhuili/EAGLE-Qwen2-7B-Instruct)
are only distributed in PyTorch pickle format (`pytorch_model.bin`). Our tq-engine
reads GGUF + safetensors directly in Rust, with no Python/torch runtime dependency.

To bridge the gap this script runs ONCE per EAGLE checkpoint to produce a
`model.safetensors` file in the same directory. Runtime inference stays pure
Rust+CUDA.

A future Rust-native pytorch pickle reader (Sprint 1.5 of the EAGLE plan)
would remove even this setup step. pytorch_model.bin is a ZIP containing a
`data.pkl` metadata file + raw tensor bytes under `data/{N}`; the pickle
format uses a small deterministic opcode subset for model weights, so a
200-ish-line Rust parser is feasible.

Usage:
  python scripts/convert_eagle_weights.py <pytorch_model.bin> [output.safetensors]

Dependencies (CPU-only ok):
  pip install torch safetensors
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="Path to pytorch_model.bin")
    parser.add_argument(
        "output", type=Path, nargs="?", default=None,
        help="Output model.safetensors (default: <input_dir>/model.safetensors)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After writing, load the safetensors back and compare tensor-by-tensor"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: {args.input} does not exist", file=sys.stderr)
        return 1

    if args.output is None:
        args.output = args.input.parent / "model.safetensors"

    try:
        import torch
    except ImportError:
        print("error: torch not installed. `pip install torch safetensors`", file=sys.stderr)
        return 1
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("error: safetensors not installed. `pip install safetensors`", file=sys.stderr)
        return 1

    print(f"loading {args.input} ({args.input.stat().st_size / 1e9:.2f} GB)")
    state = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        print(f"error: expected dict, got {type(state).__name__}", file=sys.stderr)
        return 1

    # safetensors requires all tensors be contiguous and of supported dtypes.
    # Normalise here to avoid surprises downstream.
    cleaned: dict[str, "torch.Tensor"] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"  skip non-tensor '{name}' ({type(tensor).__name__})")
            continue
        t = tensor.detach().contiguous()
        cleaned[name] = t
        print(f"  {name:<60} {str(tuple(t.shape)):<30} {t.dtype}")

    print(f"writing {args.output} ({len(cleaned)} tensors)")
    save_file(cleaned, str(args.output))
    print(f"  size: {args.output.stat().st_size / 1e9:.2f} GB")

    if args.verify:
        from safetensors.torch import load_file
        print("verifying...")
        loaded = load_file(str(args.output))
        for name, orig in cleaned.items():
            back = loaded[name]
            if orig.dtype != back.dtype:
                print(f"  MISMATCH dtype '{name}': {orig.dtype} vs {back.dtype}")
                return 1
            if orig.shape != back.shape:
                print(f"  MISMATCH shape '{name}': {orig.shape} vs {back.shape}")
                return 1
            if not torch.equal(orig.cpu(), back.cpu()):
                print(f"  MISMATCH values '{name}'")
                return 1
        print("  all tensors match")

    return 0


if __name__ == "__main__":
    sys.exit(main())
