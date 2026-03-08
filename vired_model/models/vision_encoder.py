"""
Vision Encoder
==============
Wraps a timm backbone (default: ViT-Small/16) and projects its output tokens
to the shared embedding dimension D.

Primary output  : (B, T, D)                     – sequence of image tokens
Secondary output: (B, D_backbone, H_p, W_p)     – raw spatial feature map
                                                   used for ROIAlign in the
                                                   Object Encoder.

T is the number of patch tokens produced by the backbone.  For a ViT with
patch_size=16, T = (image_size / 16)²:

    image_size=224  →  T = 196   (14×14 grid)
    image_size=384  →  T = 576   (24×24 grid)
    image_size=448  →  T = 784   (28×28 grid)
    image_size=640  →  T = 1600  (40×40 grid)

The backbone is always created with img_size=config.image_size so that
ViredConfig.image_size is the single source of truth for operating resolution.
Changing image_size in the config is all that is needed to run at a different
resolution — no backbone renaming required.

The [CLS] token (if present) is kept inside ``forward()``'s return value since
the relation decoder can freely attend to all positions.  ``forward_with_feature_map``
strips it when building the 2D spatial grid.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from vired_model.config import ViredConfig
from vired_model.utils.tensor_shapes import assert_shape


class VisionEncoder(nn.Module):
    """Extract spatial image tokens from a raw image using a timm backbone.

    The backbone's native feature dimension is projected to ``config.embedding_dim``
    via a single linear layer so that all downstream modules operate in a
    uniform space regardless of which backbone is chosen.

    Attributes
    ----------
    backbone_dim : int
        Native per-token feature dimension of the backbone (before projection).
        Exposed so callers (e.g. ObjectEncoder) can build compatible modules.
    """

    def __init__(self, config: ViredConfig) -> None:
        super().__init__()
        self.config = config

        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for VisionEncoder.  "
                "Install it with: pip install timm"
            )

        # Load backbone, disable classification head, keep patch tokens.
        #
        # img_size is passed so that ViT-family backbones adjust their
        # positional embeddings to match config.image_size.  This makes the
        # config's image_size the single source of truth for the operating
        # resolution — you can use any square size (224, 384, 448, 640, …)
        # without renaming the backbone.
        #
        # When pretrained=True timm interpolates the 224-px position
        # embeddings to the target grid; when pretrained=False they are
        # initialised from scratch at the requested size.
        #
        # CNN-family backbones (ResNet, EfficientNet, ConvNext …) do not
        # accept img_size because their convolutions naturally operate at
        # any resolution; a TypeError fallback handles those gracefully.
        try:
            self.backbone = timm.create_model(
                config.vision_backbone,
                pretrained=config.pretrained,
                num_classes=0,
                global_pool="",
                img_size=config.image_size,
            )
        except TypeError:
            # Backbone does not accept img_size (e.g. ResNet, ConvNext).
            self.backbone = timm.create_model(
                config.vision_backbone,
                pretrained=config.pretrained,
                num_classes=0,
                global_pool="",
            )

        # Resolve the backbone's native token dimension and cache it.
        # Exposed as a public attribute so ObjectEncoder can read it.
        self.backbone_dim: int = self._get_backbone_dim()

        # Linear projection: backbone_dim → embedding_dim
        self.proj = nn.Linear(self.backbone_dim, config.embedding_dim)

    # ------------------------------------------------------------------ #

    def _get_backbone_dim(self) -> int:
        """Return the backbone's per-token feature dimension.

        Tries the cheap path first (timm's ``num_features`` attribute which
        most models expose without running a forward pass), then falls back
        to a single dummy forward pass for models that do not.
        """
        # Cheap path: most timm models expose this attribute.
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is not None and isinstance(feat_dim, int) and feat_dim > 0:
            return feat_dim

        # Fallback: infer from a single dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.config.image_channels,
                self.config.image_size,
                self.config.image_size,
            )
            out = self.backbone(dummy)
            # timm ViT with global_pool="" returns (1, T, D_backbone)
            if out.dim() == 3:
                return out.shape[-1]
            # Some backbones return (1, D_backbone) – treat whole output as one token
            if out.dim() == 2:
                return out.shape[-1]
            raise ValueError(
                f"Unexpected backbone output shape: {tuple(out.shape)}"
            )

    # ------------------------------------------------------------------ #

    def _raw_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Run the backbone and normalise the output to 3D (B, T, D_backbone)."""
        tokens = self.backbone(images)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)  # (B, 1, D_backbone)
        assert tokens.dim() == 3, (
            f"Backbone output must be 3D after reshape, got {tuple(tokens.shape)}"
        )
        return tokens  # (B, T, D_backbone)

    # ------------------------------------------------------------------ #

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Standard forward pass; returns projected image tokens only.

        Args:
            images: (B, C, H, W)

        Returns:
            image_tokens: (B, T, D)
        """
        B = images.shape[0]
        assert images.dim() == 4, f"Expected 4D input (B,C,H,W), got {images.dim()}D"

        tokens = self._raw_tokens(images)          # (B, T, D_backbone)
        image_tokens = self.proj(tokens)           # (B, T, D)

        assert_shape(image_tokens, (B, None, self.config.embedding_dim), "image_tokens")
        return image_tokens

    # ------------------------------------------------------------------ #

    def forward_with_feature_map(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extended forward pass; also returns a 2D spatial feature map.

        The feature map is derived from the raw backbone patch tokens
        (before the projection layer) and is suitable for ROIAlign.

        Args:
            images: (B, C, H, W)

        Returns:
            image_tokens : (B, T, D)                   – projected tokens for decoder
            feature_map  : (B, D_backbone, H_p, W_p)  – spatial features for ROI
        """
        B = images.shape[0]
        assert images.dim() == 4, f"Expected 4D input (B,C,H,W), got {images.dim()}D"

        raw = self._raw_tokens(images)              # (B, T, D_backbone)
        image_tokens = self.proj(raw)               # (B, T, D)

        # ── Build the 2D feature map ──────────────────────────────────── #
        # Some ViT backbones prepend a CLS token so T = 1 + H_p*W_p.
        # We try to find the largest trailing sequence that is a perfect square.
        T = raw.shape[1]
        grid_size: int = 0
        n_spatial: int = 0
        for candidate in (T - 1, T):
            if candidate <= 0:
                continue
            g = int(math.isqrt(candidate))
            if g * g == candidate:
                n_spatial = candidate
                grid_size = g
                break

        if grid_size == 0:
            raise ValueError(
                f"VisionEncoder: cannot reshape backbone output of {T} tokens into "
                f"a square spatial grid.  Expected T = n² or T = 1 + n² "
                f"(e.g. 196 or 197 for ViT-16 at 224×224)."
            )

        # Take the last n_spatial tokens (patch tokens); ignore CLS if present.
        patch_tokens = raw[:, -n_spatial:]          # (B, n_spatial, D_backbone)

        # Reshape to (B, D_backbone, H_p, W_p)
        feature_map = (
            patch_tokens
            .reshape(B, grid_size, grid_size, self.backbone_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        assert_shape(image_tokens, (B, None, self.config.embedding_dim), "image_tokens")
        assert feature_map.shape == (B, self.backbone_dim, grid_size, grid_size), (
            f"feature_map shape mismatch: {tuple(feature_map.shape)}"
        )
        return image_tokens, feature_map
