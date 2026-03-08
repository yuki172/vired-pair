"""
Relation Prediction Head
========================
A three-layer MLP that maps pair embeddings to relation logits.

Input  : (B, P, in_dim)  – pair embeddings from PairBuilder
Output : (B, P, C)       – raw logits per pair

Where C = config.num_relation_classes (default 2 for binary classification).

Input dimension
---------------
``in_dim`` is computed at construction time from config:

    in_dim = 2 * embedding_dim                           (always)
           + geometry_feature_dim  (if use_geometry_features=True)

This matches the output of PairBuilder.forward exactly.

Architecture
------------
    Linear(in_dim → H) → Activation → Dropout
    Linear(H      → H) → Activation → Dropout
    Linear(H      → C)

H          = config.relation_mlp_hidden
Activation = config.ffn_activation  ('gelu' or 'relu')
Dropout    = config.relation_head_dropout

No softmax or sigmoid is applied here; that is left to the loss function
(e.g. nn.CrossEntropyLoss or nn.BCEWithLogitsLoss) chosen by the caller.

Training vs inference consistency
----------------------------------
The head outputs raw logits in all cases.  When using nn.CrossEntropyLoss,
pass logits directly.  When computing probabilities at inference time, apply
torch.softmax(logits, dim=-1) externally; do NOT apply softmax before
nn.CrossEntropyLoss (that would be double-softmax).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vired_model.config import ViredConfig
from vired_model.models.relation_decoder import _make_activation
from vired_model.utils.tensor_shapes import assert_shape


class RelationHead(nn.Module):
    """MLP that predicts a relation class for each candidate pair."""

    def __init__(self, config: ViredConfig) -> None:
        super().__init__()
        # Input dimension: object embeddings (×2) plus optional geometry vector.
        in_dim = 2 * config.embedding_dim
        if config.use_geometry_features:
            in_dim += config.geometry_feature_dim
        self.in_dim = in_dim

        H = config.relation_mlp_hidden
        C = config.num_relation_classes
        drop = config.relation_head_dropout

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, H),
            _make_activation(config.ffn_activation),
            nn.Dropout(drop),
            nn.Linear(H, H),
            _make_activation(config.ffn_activation),
            nn.Dropout(drop),
            nn.Linear(H, C),
        )

    def forward(self, pair_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_embeddings: (B, P, in_dim)
                in_dim = 2*D [+ G]  as computed in __init__

        Returns:
            logits: (B, P, C)
        """
        assert pair_embeddings.dim() == 3, (
            f"RelationHead expects 3D input (B, P, in_dim), got {pair_embeddings.dim()}D"
        )
        assert pair_embeddings.shape[-1] == self.in_dim, (
            f"RelationHead: expected last dim {self.in_dim}, "
            f"got {pair_embeddings.shape[-1]}.  "
            f"Check that PairBuilder and RelationHead use matching configs."
        )

        logits = self.mlp(pair_embeddings)  # (B, P, C)

        B, P, _ = pair_embeddings.shape
        # mlp[-1] is the final Linear layer
        assert_shape(logits, (B, P, self.mlp[-1].out_features), "relation_logits")
        return logits
