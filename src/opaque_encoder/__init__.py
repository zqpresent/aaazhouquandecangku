"""Portable inference API for a scene-text visual style encoder."""

from .inference import StyleEncoder, download_model, preprocess_style_crop

__all__ = ["StyleEncoder", "download_model", "preprocess_style_crop"]
__version__ = "1.0.0"
