"""Inference architecture; no training heads or external backbone downloads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DINOBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        import timm

        self.model_name = model_name
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=0, dynamic_img_size=True,
        )
        self.dim = int(self.model.num_features)
        self.patch_size = int(self.model.patch_embed.patch_size[0])
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images)
        if isinstance(features, dict):
            patches = features.get("x_norm_patchtokens")
            if patches is None:
                raise RuntimeError("Unsupported backbone feature dictionary")
            return patches
        return features[:, self.num_prefix_tokens :]


class StyleQueryHead(nn.Module):
    def __init__(self, feature_dim: int, num_queries: int, output_dim: int, num_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, feature_dim) * 0.02)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(feature_dim, output_dim),
        )

    def forward(self, patch_tokens: torch.Tensor, key_padding_mask: torch.Tensor | None):
        queries = self.queries.expand(patch_tokens.shape[0], -1, -1)
        local, _ = self.attention(
            self.query_norm(queries), self.feature_norm(patch_tokens),
            self.feature_norm(patch_tokens), key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        local = self.output_norm(local + queries)
        embedding = F.normalize(self.projection(local.mean(dim=1)).float(), dim=-1)
        return embedding, local


class FactorProjectionHeads(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        hidden = max(feature_dim, output_dim)
        self.font = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )
        self.appearance = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, local_tokens: torch.Tensor):
        pooled = local_tokens.mean(dim=1)
        return (
            F.normalize(self.font(pooled).float(), dim=-1),
            F.normalize(self.appearance(pooled).float(), dim=-1),
        )


class EncoderModel(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        num_queries: int = 8,
        embedding_dim: int = 256,
        attention_heads: int = 8,
        factor_embedding_dim: int = 128,
    ):
        super().__init__()
        if factor_embedding_dim <= 0:
            raise ValueError("factor_embedding_dim must be positive")
        self.backbone = DINOBackbone(model_name)
        self.style_head = StyleQueryHead(
            self.backbone.dim, num_queries, embedding_dim, attention_heads,
        )
        self.factor_heads = FactorProjectionHeads(self.backbone.dim, factor_embedding_dim)

    def _padding_mask(self, valid_mask: torch.Tensor | None, patch_count: int):
        if valid_mask is None:
            return None
        pooled = F.avg_pool2d(
            valid_mask.float(), kernel_size=self.backbone.patch_size,
            stride=self.backbone.patch_size,
        )
        valid = pooled.flatten(1) > 0.01
        if valid.shape[1] != patch_count:
            return None
        return ~valid

    def forward(self, images: torch.Tensor, valid_mask: torch.Tensor | None = None):
        patches = self.backbone(images)
        embedding, tokens = self.style_head(
            patches, self._padding_mask(valid_mask, patches.shape[1]),
        )
        font, appearance = self.factor_heads(tokens)
        return {
            "z_global": embedding,
            "z_font": font,
            "z_appearance": appearance,
            "style_tokens": tokens,
        }
