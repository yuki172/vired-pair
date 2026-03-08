"""
Central configuration dataclass for the ViRED relation model.

All architectural hyperparameters live here so that every component can be
instantiated from a single config object without scattered magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ViredConfig:
    # ------------------------------------------------------------------ #
    # Shared embedding dimension used throughout the model.               #
    # Every encoder and the decoder operate in this same space.           #
    # ------------------------------------------------------------------ #
    embedding_dim: int = 256

    # ------------------------------------------------------------------ #
    # Relation Decoder                                                     #
    # ------------------------------------------------------------------ #
    num_decoder_layers: int = 4
    num_attention_heads: int = 8
    ffn_hidden_dim: int = 1024       # hidden size of the FFN inside each decoder layer
    dropout: float = 0.1

    # ------------------------------------------------------------------ #
    # Object Encoder (3-layer CNN)                                        #
    # ------------------------------------------------------------------ #
    # Number of output channels for each of the 3 CNN layers.
    object_encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128])

    # ------------------------------------------------------------------ #
    # Relation MLP head                                                   #
    # ------------------------------------------------------------------ #
    relation_mlp_hidden: int = 512   # hidden size in the pair-classification MLP

    # ------------------------------------------------------------------ #
    # Vision backbone (timm model name)                                   #
    # ------------------------------------------------------------------ #
    vision_backbone: str = "vit_small_patch16_224"
    # Load pretrained ImageNet weights for the backbone.  Set True when
    # fine-tuning from a pretrained checkpoint; False for random init or
    # when loading a full saved model state_dict.
    pretrained: bool = False

    # ------------------------------------------------------------------ #
    # Object type vocabulary                                              #
    # ------------------------------------------------------------------ #
    # Number of distinct object types.  Learned type embeddings are added
    # to object tokens before they enter the relation decoder.
    #
    # Default types (electrical-plan task):
    #   0 = TEXT          (YOLO class_ids: 0, 2, 5)
    #   1 = SYMBOL        (YOLO class_ids: 1, 3)
    #   2 = SYMBOL_TEXT   (YOLO class_id:  4)
    num_object_types: int = 3
    use_type_embeddings: bool = True

    # ------------------------------------------------------------------ #
    # Pair Builder                                                        #
    # ------------------------------------------------------------------ #
    # "cross_type"  – all pairs (i, j) where type[i] != type[j].
    #                  With 3 types this covers TEXT↔SYMBOL, TEXT↔SYMBOL_TEXT,
    #                  and SYMBOL↔SYMBOL_TEXT automatically.
    # "all"         – every (i, j) with i < j regardless of type
    pair_mode: str = "cross_type"

    # ------------------------------------------------------------------ #
    # Image input                                                         #
    # ------------------------------------------------------------------ #
    # Square resolution that images are resized to before the backbone.
    # This is the single dial that controls operating resolution — changing
    # it is all that is needed; no backbone renaming required.
    #
    # For ViT backbones (patch_size=16) the patch grid and attention cost scale as:
    #
    #   image_size=224  →  T = 196  (14×14)  downscale from 1280px: 5.7×  attn: 1×
    #   image_size=384  →  T = 576  (24×24)  downscale from 1280px: 3.3×  attn: ~9×
    #   image_size=448  →  T = 784  (28×28)  downscale from 1280px: 2.9×  attn: ~16×
    #   image_size=640  →  T = 1600 (40×40)  downscale from 1280px: 2.0×  attn: ~67×
    #
    # Recommended configs for 1280 px electrical-plan slices:
    #
    #   Good balance (pretrained weights in timm):
    #     vision_backbone="vit_small_patch16_384", image_size=384
    #
    #   Higher fidelity (position embeddings interpolated from 224-px pretrained):
    #     vision_backbone="vit_small_patch16_224", image_size=640
    #     Note: ViT self-attention becomes ~67× more expensive than at 224 px.
    #     Consider a CNN backbone (e.g. "convnext_small") for this resolution.
    image_size: int = 224
    image_channels: int = 3

    # ------------------------------------------------------------------ #
    # Classification head                                                 #
    # ------------------------------------------------------------------ #
    num_relation_classes: int = 2    # binary: is_pair / not_pair

    # Dropout applied between MLP layers in the relation head.
    # Independent from the decoder dropout so the head can be tuned
    # separately (e.g. higher dropout for a small dataset).
    relation_head_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Normalisation order in the Relation Decoder                        #
    # ------------------------------------------------------------------ #
    # True  → pre-norm  (LayerNorm before each sub-layer) – recommended;
    #         matches the ViT convention used by the vision backbone and
    #         trains more stably.
    # False → post-norm (LayerNorm after residual addition) – original
    #         "Attention is All You Need" convention.
    decoder_pre_norm: bool = True

    # ------------------------------------------------------------------ #
    # Activation function used in the decoder FFN and the relation MLP   #
    # ------------------------------------------------------------------ #
    # "gelu" – standard choice for transformer FFNs, matches ViT.
    # "relu" – alternative; used in the original transformer paper.
    ffn_activation: str = "gelu"

    # ------------------------------------------------------------------ #
    # ROI Visual Features (Modification 2)                               #
    # ------------------------------------------------------------------ #
    # When True, the Object Encoder extracts region features from the
    # vision backbone's spatial feature map using ROIAlign and fuses them
    # with the mask CNN features.  Requires torchvision.
    use_roi_features: bool = True

    # Spatial output size of ROIAlign (roi_pool_size × roi_pool_size).
    roi_pool_size: int = 7

    # Dimension of the projected ROI feature vector that is concatenated
    # with the mask CNN features before the final linear projection.
    roi_feature_dim: int = 64

    # Number of pixels to expand each bounding box (in all four directions)
    # before running ROIAlign.  This gives the ROI features access to the
    # local neighbourhood of the object (e.g. leader lines that start just
    # outside the tight bounding box).  Expansion is applied symmetrically;
    # the resulting box is clamped to the image boundary.
    #
    # Set to 0 to use the original tight bounding boxes (default, no change
    # in behaviour compared to previous versions).
    #
    # Practical guidance (in model-space pixels, after resize):
    #   image_size=224  →  try 16–32 px
    #   image_size=384  →  try 24–48 px  (~100 original-space px at 3.3× downscale)
    #   image_size=640  →  try 40–80 px  (~100 original-space px at 2.0× downscale)
    #
    # Note: this parameter only affects the boxes used for ROIAlign.  The
    # geometry features (dx, dy, dist, IoU …) always use the original tight
    # boxes passed to the model forward so that spatial relationships are
    # computed on the true object footprints.
    roi_context_pad: int = 0

    # ------------------------------------------------------------------ #
    # Geometry Features (Modification 1)                                 #
    # ------------------------------------------------------------------ #
    # When True, the Pair Builder computes a 6-dimensional geometry vector
    # for each candidate pair (dx, dy, dist, log_w_ratio, log_h_ratio, iou)
    # and concatenates it with the pair embedding before the relation head.
    use_geometry_features: bool = True

    # Must be 6 – matches the fixed set of geometry features produced by
    # compute_pair_geometry_features().  Kept in config so RelationHead
    # can compute its input dimension without importing pair_builder.
    geometry_feature_dim: int = 6

    def __post_init__(self) -> None:
        assert self.embedding_dim % self.num_attention_heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert self.pair_mode in ("cross_type", "all"), (
            f"pair_mode must be 'cross_type' or 'all', got '{self.pair_mode}'"
        )
        assert len(self.object_encoder_channels) == 3, (
            "object_encoder_channels must have exactly 3 entries (one per CNN layer)"
        )
        assert self.ffn_activation in ("relu", "gelu"), (
            f"ffn_activation must be 'relu' or 'gelu', got '{self.ffn_activation}'"
        )
        assert self.roi_pool_size >= 1, (
            f"roi_pool_size must be >= 1, got {self.roi_pool_size}"
        )
        assert self.roi_feature_dim >= 1, (
            f"roi_feature_dim must be >= 1, got {self.roi_feature_dim}"
        )
        if self.use_geometry_features:
            assert self.geometry_feature_dim == 6, (
                f"geometry_feature_dim must be 6 (dx, dy, dist, log_w_ratio, "
                f"log_h_ratio, iou), got {self.geometry_feature_dim}"
            )
