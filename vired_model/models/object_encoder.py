"""
Object Encoder
==============
Encodes per-object information into fixed-size embedding vectors.

Two feature sources are fused before the final projection:

    1. Mask features  – from a 3-layer CNN applied to the binary object mask.
    2. ROI features   – from ROIAlign on the vision backbone's spatial feature
                        map, projected to ``config.roi_feature_dim``.
                        Only active when ``config.use_roi_features=True`` and
                        the feature map + boxes are passed to ``forward()``.

Inputs:
    masks        : (B, N, H, W)  – binary float masks, one per object
    object_types : (B, N)        – integer type indices
    feature_map  : (B, D_bk, H_p, W_p) – optional spatial feature map from
                                          VisionEncoder.forward_with_feature_map
    boxes        : (B, N, 4)     – optional bounding boxes (x1,y1,x2,y2) in
                                   image pixel coordinates

Output:
    object_embeddings : (B, N, D)

Pipeline
--------
    masks                           feature_map + boxes
      │                                    │
      ▼                                    ▼
  MaskCNN (3-layer)              ROIAlign → AdaptivePool
      │                                    │
  (B*N, C3)                      RoiEncoder.proj
      │                               (B*N, roi_feature_dim)
      └──────────────┬────────────────────┘
                     ▼
              concat → (B*N, C3 [+ roi_feature_dim])
                     │
              Linear → LayerNorm → (B*N, D)
                     │
              + type_embed(object_types)
                     │
              (B, N, D)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision.ops import roi_align as _torchvision_roi_align
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

from vired_model.config import ViredConfig
from vired_model.utils.tensor_shapes import assert_shape


# ────────────────────────────────────────────────────────────────────────── #


class RoiEncoder(nn.Module):
    """Project a batch of ROI-aligned feature patches to a fixed-size vector.

    Input  : (K, D_backbone, roi_pool_size, roi_pool_size)
    Output : (K, roi_feature_dim)

    AdaptiveAvgPool collapses the spatial dimensions, then a linear layer
    projects to the target dimension followed by LayerNorm for stable scaling.
    """

    def __init__(
        self,
        backbone_dim: int,
        roi_pool_size: int,
        roi_feature_dim: int,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, roi_feature_dim),
            nn.LayerNorm(roi_feature_dim),
        )

    def forward(self, roi_aligned: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_aligned: (K, D_backbone, roi_pool_size, roi_pool_size)

        Returns:
            features: (K, roi_feature_dim)
        """
        x = self.pool(roi_aligned)   # (K, D_backbone, 1, 1)
        x = x.flatten(1)             # (K, D_backbone)
        return self.proj(x)          # (K, roi_feature_dim)


# ────────────────────────────────────────────────────────────────────────── #


class ObjectEncoder(nn.Module):
    """Encode per-object masks (and optionally region image features) to embeddings.

    Processing is vectorised: the (B, N) object dimension is folded into the
    batch dimension before the CNN and ROI operations, then unfolded afterwards,
    so all weights are shared across objects.

    Parameters
    ----------
    config : ViredConfig
    backbone_dim : int, optional
        Native feature dimension of the vision backbone.  Required when
        ``config.use_roi_features=True``.  Obtained from
        ``VisionEncoder.backbone_dim``.
    """

    def __init__(
        self,
        config: ViredConfig,
        backbone_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        C1, C2, C3 = config.object_encoder_channels

        if config.use_roi_features:
            if not _TORCHVISION_AVAILABLE:
                raise ImportError(
                    "torchvision is required for ROI features.  "
                    "Install it with: pip install torchvision"
                )
            if backbone_dim is None:
                raise ValueError(
                    "backbone_dim must be provided to ObjectEncoder when "
                    "config.use_roi_features=True.  Pass VisionEncoder.backbone_dim."
                )

        # ── Mask CNN ────────────────────────────────────────────────────── #
        # Three convolutional layers with ReLU activations.
        # Input has a single channel (the binary mask).
        self.cnn = nn.Sequential(
            nn.Conv2d(1, C1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(C1, C2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(C2, C3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # ── ROI Encoder ─────────────────────────────────────────────────── #
        if config.use_roi_features:
            self.roi_encoder: Optional[RoiEncoder] = RoiEncoder(
                backbone_dim=backbone_dim,       # type: ignore[arg-type]
                roi_pool_size=config.roi_pool_size,
                roi_feature_dim=config.roi_feature_dim,
            )
            proj_in_dim = C3 + config.roi_feature_dim
        else:
            self.roi_encoder = None
            proj_in_dim = C3

        # ── Final projection ─────────────────────────────────────────────── #
        # Linear(C3 [+ roi_feature_dim] → D) + LayerNorm
        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )

        # ── Optional type embeddings ─────────────────────────────────────── #
        if config.use_type_embeddings:
            self.type_embed = nn.Embedding(
                config.num_object_types, config.embedding_dim
            )
        else:
            self.type_embed = None

    # ------------------------------------------------------------------ #

    def forward(
        self,
        masks: torch.Tensor,
        object_types: Optional[torch.Tensor] = None,
        feature_map: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            masks:        (B, N, H, W) – binary float masks, one per object
            object_types: (B, N)       – integer type indices; required when
                                         config.use_type_embeddings=True
            feature_map:  (B, D_backbone, H_p, W_p) – spatial backbone features
                          from VisionEncoder.forward_with_feature_map; required
                          when config.use_roi_features=True
            boxes:        (B, N, 4)   – bounding boxes (x1,y1,x2,y2) in image
                          pixel coordinates; required when use_roi_features=True

        Returns:
            object_embeddings: (B, N, D)
        """
        B, N, H_img, W_img = masks.shape

        # ── 1. Mask CNN ────────────────────────────────────────────────── #
        x = masks.view(B * N, 1, H_img, W_img)     # (B*N, 1, H, W)
        x = self.cnn(x)                             # (B*N, C3, 1, 1)
        mask_feats = x.view(B * N, -1)              # (B*N, C3)

        # ── 2. ROI features (optional) ────────────────────────────────── #
        if self.roi_encoder is not None:
            if feature_map is None or boxes is None:
                raise ValueError(
                    "feature_map and boxes must be provided to ObjectEncoder "
                    "when config.use_roi_features=True"
                )
            assert_shape(boxes, (B, N, 4), "boxes")
            # feature_map: (B, D_backbone, H_p, W_p)
            H_p = feature_map.shape[2]
            spatial_scale = float(H_p) / float(H_img)

            # Optionally expand boxes to capture local context (e.g. leader
            # lines that start outside the tight bounding box).  Expansion is
            # applied only here; the original `boxes` tensor is never mutated
            # so geometry features computed elsewhere still use tight boxes.
            if self.config.roi_context_pad > 0:
                pad = float(self.config.roi_context_pad)
                roi_boxes = boxes.float().clone()
                roi_boxes[..., 0] = (roi_boxes[..., 0] - pad).clamp(min=0.0)
                roi_boxes[..., 1] = (roi_boxes[..., 1] - pad).clamp(min=0.0)
                roi_boxes[..., 2] = (roi_boxes[..., 2] + pad).clamp(max=float(W_img))
                roi_boxes[..., 3] = (roi_boxes[..., 3] + pad).clamp(max=float(H_img))
            else:
                roi_boxes = boxes.float()

            # Build (B*N, 5) ROI tensor with batch indices in the first column.
            batch_idx = (
                torch.arange(B, device=boxes.device)
                .unsqueeze(1)
                .expand(B, N)
                .reshape(-1, 1)
                .float()
            )                                           # (B*N, 1)
            rois = torch.cat(
                [batch_idx, roi_boxes.reshape(B * N, 4)], dim=1
            )                                           # (B*N, 5)

            roi_aligned = _torchvision_roi_align(
                feature_map.float(),
                rois,
                output_size=self.config.roi_pool_size,
                spatial_scale=spatial_scale,
                aligned=True,
            )                                           # (B*N, D_backbone, P, P)

            roi_feats = self.roi_encoder(roi_aligned)   # (B*N, roi_feature_dim)

            # Concatenate mask and ROI features before projection.
            combined = torch.cat([mask_feats, roi_feats], dim=-1)  # (B*N, C3+roi_dim)
        else:
            combined = mask_feats                       # (B*N, C3)

        # ── 3. Project to embedding dim ───────────────────────────────── #
        x = self.proj(combined)                         # (B*N, D)

        # ── 4. Restore object dimension ───────────────────────────────── #
        object_embeddings = x.view(B, N, self.config.embedding_dim)  # (B, N, D)

        # ── 5. Add type embeddings ────────────────────────────────────── #
        if self.type_embed is not None:
            if object_types is None:
                raise ValueError(
                    "object_types must be provided when use_type_embeddings=True"
                )
            assert_shape(object_types, (B, N), "object_types")
            type_emb = self.type_embed(object_types)    # (B, N, D)
            object_embeddings = object_embeddings + type_emb

        assert_shape(object_embeddings, (B, N, self.config.embedding_dim), "object_embeddings")
        return object_embeddings
