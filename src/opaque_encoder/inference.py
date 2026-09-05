"""Load public or local safetensors weights and encode text-region crops."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from .model import EncoderModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
MODEL_FILES = ("config.json", "model.safetensors")
MODEL_KEYS = {
    "model_name", "num_queries", "embedding_dim", "attention_heads", "factor_embedding_dim",
}


def _load_config(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format_version") != 1:
        raise ValueError("Unsupported model format_version; expected 1")
    kwargs = config.get("model_kwargs")
    if not isinstance(kwargs, dict) or set(kwargs) != MODEL_KEYS:
        raise ValueError(f"model_kwargs must contain exactly {sorted(MODEL_KEYS)}")
    size = config.get("image_size")
    if not isinstance(size, list) or len(size) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in size
    ):
        raise ValueError("image_size must contain two positive integers [height, width]")
    weight_path = model_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing weight file: {weight_path}")
    expected = config.get("weights_sha256")
    if expected is not None:
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("weights_sha256 must be a 64-character SHA-256 digest")
        digest = hashlib.sha256()
        with weight_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected.lower():
            raise ValueError("model.safetensors SHA-256 does not match config.json")
    return config


def download_model(
    model_id: str,
    *,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Download only config and weights from a public Hugging Face model.

    Requests are anonymous, even when this machine has a saved access token.
    The returned snapshot or local directory is verified against the manifest.
    """
    from huggingface_hub import snapshot_download

    result = snapshot_download(
        repo_id=model_id, repo_type="model", revision=revision,
        allow_patterns=list(MODEL_FILES), token=False,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_dir=str(local_dir) if local_dir is not None else None,
        local_files_only=local_files_only,
    )
    path = Path(result)
    _load_config(path)
    return path


def preprocess_style_crop(
    image: str | Path | Image.Image,
    size: tuple[int, int] = (112, 448),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aspect-preserving LANCZOS resize, gray padding, ImageNet normalization."""
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = min(width / rgb.width, height / rgb.height)
    resized_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    resized = rgb.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    valid = torch.zeros((1, height, width), dtype=torch.float32)
    valid[:, top : top + resized.height, left : left + resized.width] = 1.0
    return tensor, valid


class StyleEncoder:
    """Inference wrapper returning normalized embeddings and shared style tokens."""

    def __init__(self, model: EncoderModel, config: dict[str, Any], device: str = "cpu"):
        self.device = torch.device(device)
        self.config = config
        self.image_size = tuple(config["image_size"])
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        local_path_or_hf_id: str | Path,
        revision: str = "main",
        cache_dir: str | Path | None = None,
        device: str = "cpu",
        *,
        local_files_only: bool = False,
    ) -> "StyleEncoder":
        """Load a model directory or an anonymous public Hugging Face model ID.

        Pass a local directory and local_files_only=True for fully offline use.
        CUDA users select their device explicitly, e.g. device="cuda:0".
        """
        model_dir = Path(local_path_or_hf_id).expanduser()
        if model_dir.is_dir():
            pass
        elif isinstance(local_path_or_hf_id, Path) or str(local_path_or_hf_id).startswith(("/", "./", "../", "~")):
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
        else:
            model_dir = download_model(
                str(local_path_or_hf_id), revision=revision, cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        config = _load_config(model_dir)
        model = EncoderModel(**config["model_kwargs"])
        if any(value % model.backbone.patch_size for value in config["image_size"]):
            raise ValueError("image_size must be divisible by the backbone patch size")
        model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device="cpu"), strict=True)
        return cls(model, config, device)

    def encode(
        self,
        images: Iterable[str | Path | Image.Image],
        batch_size: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Encode a nonempty iterable, preprocessing at most one batch at a time.

        Returns CPU float32 tensors z_global [N,256], z_font [N,128],
        z_appearance [N,128], and style_tokens [N,8,384] for the published model.
        All three z vectors are L2-normalized; style_tokens are not.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(images, (str, Path, Image.Image)):
            raise TypeError("images must be an iterable of images; wrap a single image in a list")
        iterator = iter(images)
        chunks: dict[str, list[torch.Tensor]] = {}
        with torch.inference_mode():
            while True:
                batch = list(islice(iterator, batch_size))
                if not batch:
                    break
                tensors, masks = zip(*(preprocess_style_crop(item, self.image_size) for item in batch))
                output = self.model(
                    torch.stack(tensors).to(self.device), torch.stack(masks).to(self.device),
                )
                for key, value in output.items():
                    result = value.float().cpu()
                    if not torch.isfinite(result).all():
                        raise RuntimeError(f"Non-finite output in {key}; check the input crop and weights")
                    chunks.setdefault(key, []).append(result)
        if not chunks:
            raise ValueError("images must contain at least one image")
        return {key: torch.cat(values, dim=0) for key, values in chunks.items()}
