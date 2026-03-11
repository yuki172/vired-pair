"""
ViredRelationModel – Top-Level Model
=====================================
Assembles all components into a single nn.Module with a clean forward API.

Forward pass
------------
    image         : (B, C, H, W)
    object_masks  : (B, N, H, W)   – binary float mask per object
    object_boxes  : (B, N, 4)      – bounding boxes (x1,y1,x2,y2) in pixels
    object_types  : (B, N)         – integer type ids (e.g. 0=symbol, 1=text)

Returns (as a dataclass for explicit field access)
    pair_logits   : (B, P, C)      – raw class scores per candidate pair
    pair_indices  : (B, P, 2)      – (i, j) object indices for each pair

Full pipeline
-------------
    image
      └─ VisionEncoder ──────────────────────────────────► image_tokens (B, T, D)
                        ─────── (when use_roi_features) ─► feature_map (B, D_bk, H_p, W_p)
                                                               │
    object_masks                                               │
    object_boxes (→ ROIAlign on feature_map)                   │
      └─ ObjectEncoder ──► object_tokens (B, N, D)            │
                                    │                          │
                                    └──► RelationDecoder ◄─────┘
                                               │
                                    decoded_tokens (B, N, D)
                                               │
    object_boxes (→ geometry features)         │
      └─ PairBuilder ──► pair_embeddings (B, P, 2D [+G])
                         pair_indices    (B, P, 2)
                                               │
                                    RelationHead ──► logits (B, P, C)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from vired_model.config import ViredConfig
from vired_model.models.vision_encoder import VisionEncoder
from vired_model.models.object_encoder import ObjectEncoder
from vired_model.models.relation_decoder import RelationDecoder
from vired_model.models.pair_builder import PairBuilder
from vired_model.models.relation_head import RelationHead
from vired_model.utils.tensor_shapes import assert_shape


@dataclass
class ViredOutput:
    """Structured output of ViredRelationModel.forward()."""

    pair_logits: torch.Tensor
    """(B, P, C) – raw (un-normalised) logits for each candidate pair."""

    pair_indices: torch.Tensor
    """(B, P, 2) – (i, j) indices identifying each pair in the object list."""


class ViredRelationModel(nn.Module):
    """ViRED-style relation detection model with geometry and ROI enrichment.

    Detects which pairs of objects (e.g. symbol + text) are "natural pairs"
    using a vision backbone, per-object mask + ROI encoder, transformer
    decoder, geometry-enriched pair builder, and MLP classification head.

    Parameters
    ----------
    config : ViredConfig
        All architectural hyperparameters.
    """

    def __init__(self, config: ViredConfig) -> None:
        super().__init__()
        self.config = config

        # Vision encoder is constructed first so we can read backbone_dim
        # and pass it to the object encoder.
        self.vision_encoder = VisionEncoder(config)
        self.object_encoder = ObjectEncoder(
            config,
            backbone_dim=self.vision_encoder.backbone_dim,
        )
        self.relation_decoder = RelationDecoder(config)
        self.pair_builder = PairBuilder(config)
        self.relation_head = RelationHead(config)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        image: torch.Tensor,
        object_masks: torch.Tensor,
        object_valid: torch.Tensor,
        object_boxes: torch.Tensor,
        object_types: torch.Tensor,
        object_key_padding_mask: Optional[torch.Tensor],
        image_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> ViredOutput:
        """Run the full ViRED pipeline.

        Args:
            image:
                (B, C, H, W) – normalised image tensor.
            object_masks:
                (B, N, H, W) – binary float masks, one channel per object.
                Values should be 0.0 (background) or 1.0 (object region).
            object_valid:
                (B, N) boolean tensor, indicates if object is in the labels, not a placeholder.
            object_boxes:
                (B, N, 4) – bounding boxes in image pixel coordinates,
                format (x1, y1, x2, y2).
            object_types:
                (B, N) – integer type id for each object.
                E.g. 0 = symbol, 1 = text (defined by the dataset).
            object_key_padding_mask:
                Optional (B, N) bool tensor passed to the decoder self-attention.
                True = ignore that object (e.g. padding objects).
            image_key_padding_mask:
                Optional (B, T) bool tensor passed to the decoder cross-attention.
                True = ignore that image token (e.g. padding tokens).

        Returns:
            ViredOutput with fields:
                pair_logits  : (B, P, C)
                pair_indices : (B, P, 2)
        """
        B, _C, _H, _W = image.shape
        N = object_masks.shape[1]

        assert object_masks.shape[0] == B,  "Batch size mismatch: image vs masks"
        assert object_boxes.shape[0] == B,  "Batch size mismatch: image vs boxes"
        assert object_types.shape[0] == B,  "Batch size mismatch: image vs types"
        assert object_boxes.shape == (B, N, 4), (
            f"object_boxes must be (B, N, 4), got {tuple(object_boxes.shape)}"
        )
        assert object_types.shape == (B, N), (
            f"object_types must be (B, N), got {tuple(object_types.shape)}"
        )

        # ── 1. Vision Encoder ─────────────────────────────────────────── #
        if self.config.use_roi_features:
            # Need both image tokens (for decoder) and spatial feature map
            # (for ROIAlign in the object encoder).
            image_tokens, feature_map = self.vision_encoder.forward_with_feature_map(image)
            # image_tokens : (B, T, D)
            # feature_map  : (B, D_backbone, H_p, W_p)
        else:
            image_tokens = self.vision_encoder(image)   # (B, T, D)
            feature_map = None

        assert_shape(image_tokens, (B, None, self.config.embedding_dim), "image_tokens")

        # ── 2. Object Encoder ─────────────────────────────────────────── #
        # Encodes binary masks; optionally fuses ROI features from the
        # backbone spatial feature map.
        # object_tokens : (B, N, D)
        object_tokens = self.object_encoder(
            masks=object_masks,
            object_types=object_types,
            feature_map=feature_map,
            boxes=object_boxes if self.config.use_roi_features else None,
        )
        assert_shape(object_tokens, (B, N, self.config.embedding_dim), "object_tokens")

        # ── 3. Relation Decoder ───────────────────────────────────────── #
        # decoded_tokens : (B, N, D)
        decoded_tokens = self.relation_decoder(
            object_tokens=object_tokens,
            image_tokens=image_tokens,
            object_key_padding_mask=object_key_padding_mask,
            image_key_padding_mask=image_key_padding_mask,
        )
        assert_shape(decoded_tokens, (B, N, self.config.embedding_dim), "decoded_tokens")

        # ── 4. Pair Builder ───────────────────────────────────────────── #
        # Enumerates candidate pairs; appends per-pair geometry vectors
        # when config.use_geometry_features=True.
        # pair_embeddings : (B, P, 2D [+G])
        # pair_indices    : (B, P, 2)
        pair_embeddings, pair_indices = self.pair_builder(
            object_tokens=decoded_tokens,
            object_types=object_types,
            boxes=object_boxes if self.config.use_geometry_features else None,
        )

        P = pair_embeddings.shape[1]
        assert pair_indices.shape == (B, P, 2), (
            f"pair_indices shape mismatch: {tuple(pair_indices.shape)}"
        )

        # ── 5. Relation Head ──────────────────────────────────────────── #
        # pair_logits : (B, P, C)
        pair_logits = self.relation_head(pair_embeddings)
        assert_shape(pair_logits, (B, P, self.config.num_relation_classes), "pair_logits")

        return ViredOutput(
            pair_logits=pair_logits,
            pair_indices=pair_indices,
        )

    # ------------------------------------------------------------------ #

    def predict(
        self,
        image: torch.Tensor,
        object_masks: torch.Tensor,
        object_boxes: torch.Tensor,
        object_types: torch.Tensor,
    ) -> ViredOutput:
        """Convenience wrapper: run a deterministic inference forward pass.

        Differences from calling ``forward()`` directly:

        1. The model is switched to eval mode for the duration of the call,
           disabling dropout and using running statistics in any batch-norm
           layers.  The original training/eval state is restored afterwards.
        2. Gradient computation is disabled via ``torch.no_grad()``.

        Returns the same ``ViredOutput`` as ``forward()``.  To obtain class
        probabilities, apply ``torch.softmax(pair_logits, dim=-1)`` on the
        returned logits.  Do **not** apply softmax before passing logits to
        ``nn.CrossEntropyLoss`` during training — that would be double-softmax.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(image, object_masks, object_boxes, object_types)
        finally:
            if was_training:
                self.train()

    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
