"""Command-line download and encoding utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .inference import StyleEncoder, download_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opaque-encoder",
        description="Encode tightly cropped text regions into visual-style embeddings.",
        epilog="Examples: opaque-encoder download USER/MODEL --local-dir weights; "
        "opaque-encoder encode --model weights --input crops --output outputs/styles.npz",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download public model config and weights anonymously")
    download.add_argument("model_id", help="Public Hugging Face ID: USER/MODEL")
    download.add_argument("--local-dir", type=Path, required=True)
    download.add_argument("--revision", default="main", help="Model branch, tag, or commit")
    download.add_argument("--cache-dir", type=Path)
    download.add_argument("--local-files-only", action="store_true")
    encode = commands.add_parser("encode", help="Encode one image or a directory into NPZ + JSON")
    encode.add_argument("--model", required=True, help="Local model directory or public USER/MODEL ID")
    encode.add_argument("--input", type=Path, required=True, help="One image or a directory of text crops")
    encode.add_argument("--output", type=Path, required=True, help="Output .npz file; sibling .json stores metadata")
    encode.add_argument("--recursive", action="store_true", help="Search input subdirectories")
    encode.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0 (default: cpu)")
    encode.add_argument("--batch-size", type=int, default=32)
    encode.add_argument("--revision", default="main", help="Model branch, tag, or commit")
    encode.add_argument("--cache-dir", type=Path)
    encode.add_argument("--local-files-only", action="store_true", help="Disable model network access")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        model_dir = download_model(
            args.model_id, revision=args.revision, cache_dir=args.cache_dir,
            local_dir=args.local_dir, local_files_only=args.local_files_only,
        )
        print(f"Verified model downloaded to {model_dir}")
        return 0
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    metadata_path = args.output.with_suffix(".json")
    if args.output.exists() or metadata_path.exists():
        parser.error("Output NPZ or JSON already exists; choose another output path")
    if args.input.is_file():
        paths = [args.input]
        names = [args.input.name]
    elif args.input.is_dir():
        candidates = args.input.rglob("*") if args.recursive else args.input.iterdir()
        paths = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        names = [path.relative_to(args.input).as_posix() for path in paths]
    else:
        parser.error("--input must be an existing image or directory")
    if not paths:
        parser.error("No supported images found in the input directory")
    encoder = StyleEncoder.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.cache_dir, device=args.device,
        local_files_only=args.local_files_only,
    )
    encoded = encoder.encode(paths, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: value.numpy() for key, value in encoded.items()})
    metadata = {
        "format_version": 1,
        "count": len(paths),
        "files": names,
        "image_size": encoder.config["image_size"],
        "weights_sha256": encoder.config.get("weights_sha256"),
        "outputs": {key: {"shape": list(value.shape), "dtype": "float32"} for key, value in encoded.items()},
        "normalized_outputs": ["z_global", "z_font", "z_appearance"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Encoded {len(paths)} image(s): {args.output} and {metadata_path}")
    return 0
