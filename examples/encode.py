"""Minimal local-directory inference example."""

from __future__ import annotations

import argparse
from pathlib import Path

from opaque_encoder import StyleEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Downloaded model directory")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    encoder = StyleEncoder.from_pretrained(
        args.model,
        device=args.device,
        local_files_only=True,
    )
    result = encoder.encode(args.image)
    for name, tensor in result.items():
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
