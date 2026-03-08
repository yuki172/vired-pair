"""
Relation Decoder
================
A stack of transformer decoder layers that allow object tokens to attend to
image tokens, refining object representations with global visual context.

Inputs:
    object_tokens : (B, N, D)  – output of the Object Encoder
    image_tokens  : (B, T, D)  – output of the Vision Encoder

Output:
    updated_object_tokens : (B, N, D)

Each decoder layer contains:
    1. Self-attention over object tokens         (objects relate to each other)
    2. Cross-attention: objects → image tokens   (objects ground themselves in the image)
    3. Feed-forward network
    4. Residual connections + LayerNorm + Dropout after each sub-layer

Normalisation order is controlled by ``config.decoder_pre_norm``:
    True  → pre-norm:  LayerNorm(x) is passed INTO each sub-layer;
                       residual is added to the *un-normalised* stream.
                       Matches the ViT convention, trains more stably.
    False → post-norm: LayerNorm is applied AFTER the residual addition.
                       Original "Attention is All You Need" convention.

Number of layers is controlled by ``config.num_decoder_layers``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from vired_model.config import ViredConfig
from vired_model.utils.tensor_shapes import assert_shape


def _make_activation(name: str) -> nn.Module:
    """Return an activation module by name ('relu' or 'gelu')."""
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation '{name}'. Choose 'relu' or 'gelu'.")


class RelationDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Supports both pre-norm and post-norm via ``config.decoder_pre_norm``.

    Notation:
        Q = object tokens
        K = V = image tokens  (for cross-attention)
    """

    def __init__(self, config: ViredConfig) -> None:
        super().__init__()
        D = config.embedding_dim
        H = config.num_attention_heads
        ffn_dim = config.ffn_hidden_dim
        drop = config.dropout
        self.pre_norm = config.decoder_pre_norm

        # 1. Self-attention over object tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=H,
            dropout=drop,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(D)
        self.drop1 = nn.Dropout(drop)

        # 2. Cross-attention: objects attend to image tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=H,
            dropout=drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(D)
        self.drop2 = nn.Dropout(drop)

        # 3. Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(D, ffn_dim),
            _make_activation(config.ffn_activation),
            nn.Dropout(drop),
            nn.Linear(ffn_dim, D),
        )
        self.norm3 = nn.LayerNorm(D)
        self.drop3 = nn.Dropout(drop)

    # ------------------------------------------------------------------ #

    def _self_attn_block(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
        )
        return self.drop1(out)

    def _cross_attn_block(
        self,
        x: torch.Tensor,
        image_tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out, _ = self.cross_attn(
            query=x, key=image_tokens, value=image_tokens,
            key_padding_mask=key_padding_mask,
        )
        return self.drop2(out)

    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop3(self.ffn(x))

    # ------------------------------------------------------------------ #

    def forward(
        self,
        object_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        object_key_padding_mask: Optional[torch.Tensor] = None,
        image_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            object_tokens:            (B, N, D)
            image_tokens:             (B, T, D)
            object_key_padding_mask:  (B, N)  bool – True = ignore that token
            image_key_padding_mask:   (B, T)  bool – True = ignore that token

        Returns:
            updated object_tokens:    (B, N, D)
        """
        if self.pre_norm:
            # ── Pre-norm: normalise BEFORE each sub-layer ──────────────── #
            # Residual stream (object_tokens) is NOT normalised;
            # norm is applied only inside each sub-layer input.

            # Self-attention
            object_tokens = object_tokens + self._self_attn_block(
                self.norm1(object_tokens), object_key_padding_mask
            )

            # Cross-attention: objects query image
            object_tokens = object_tokens + self._cross_attn_block(
                self.norm2(object_tokens), image_tokens, image_key_padding_mask
            )

            # FFN
            object_tokens = object_tokens + self._ffn_block(
                self.norm3(object_tokens)
            )

        else:
            # ── Post-norm: normalise AFTER residual addition ────────────── #
            # Original "Attention is All You Need" convention.

            # Self-attention
            object_tokens = self.norm1(
                object_tokens + self._self_attn_block(object_tokens, object_key_padding_mask)
            )

            # Cross-attention: objects query image
            object_tokens = self.norm2(
                object_tokens + self._cross_attn_block(object_tokens, image_tokens, image_key_padding_mask)
            )

            # FFN
            object_tokens = self.norm3(
                object_tokens + self._ffn_block(object_tokens)
            )

        return object_tokens


# --------------------------------------------------------------------------- #


class RelationDecoder(nn.Module):
    """Stack of RelationDecoderLayers.

    Iteratively refines object token representations by attending to the full
    image token sequence.
    """

    def __init__(self, config: ViredConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [RelationDecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )

        # For pre-norm, a final LayerNorm is applied to the output stream.
        # This mirrors the ViT convention and ensures the output distribution
        # is normalised before it reaches the pair builder.
        self.final_norm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(config.embedding_dim) if config.decoder_pre_norm else None
        )

    def forward(
        self,
        object_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        object_key_padding_mask: Optional[torch.Tensor] = None,
        image_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            object_tokens : (B, N, D)
            image_tokens  : (B, T, D)

        Returns:
            object_tokens : (B, N, D)  – refined representations
        """
        B, N, D = object_tokens.shape

        for layer in self.layers:
            object_tokens = layer(
                object_tokens,
                image_tokens,
                object_key_padding_mask=object_key_padding_mask,
                image_key_padding_mask=image_key_padding_mask,
            )

        if self.final_norm is not None:
            object_tokens = self.final_norm(object_tokens)

        assert_shape(object_tokens, (B, N, D), "decoded_object_tokens")
        return object_tokens
