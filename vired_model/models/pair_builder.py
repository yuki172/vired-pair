"""
Pair Representation Builder
============================
Constructs candidate (subject, object) pairs from decoded object embeddings,
then optionally enriches the pair embedding with geometric context derived
from the objects' bounding boxes.

Inputs:
    object_tokens : (B, N, D)
    object_types  : (B, N)    – integer type ids (0 = symbol, 1 = text, …)
    boxes         : (B, N, 4) – optional bounding boxes (x1,y1,x2,y2) in
                                 image pixel coordinates; required when
                                 config.use_geometry_features=True

Outputs:
    pair_embeddings : (B, P_max, 2D [+ G])  – concatenation of the two object
                                          embeddings, with geometry vector
                                          appended when use_geometry_features=True
    pair_indices    : (B, P_max, 2)         – (i, j) indices into the N objects

Pair construction modes
-----------------------
"cross_type" (default)
    Generate all pairs (a, b) where type[a] != type[b].
    The object with the **lower type-id** is always placed at position a,
    guaranteeing that the pair embedding is always:
        concat(lower-type-id embedding, higher-type-id embedding)
    regardless of the order in which objects appear in the input list.
    For the electrical-plan task (symbol=0, text=1) this means the embedding
    is always concat(symbol_emb, text_emb).

"all"
    Generate all unordered pairs (i, j) with i < j, regardless of type.

Geometry features (G = 6)
--------------------------
When ``config.use_geometry_features=True`` and ``boxes`` is provided, a 6-dim
vector is appended per pair:

    [dx, dy, dist, log_w_ratio, log_h_ratio, iou]

where:
    dx           = (cx_j - cx_i) / image_size   (normalised horizontal offset)
    dy           = (cy_j - cy_i) / image_size   (normalised vertical offset)
    dist         = sqrt(dx² + dy²)               (normalised centre distance)
    log_w_ratio  = log(w_j / w_i)               (log width ratio)
    log_h_ratio  = log(h_j / h_i)               (log height ratio)
    iou          = IoU(box_i, box_j)             (intersection over union)

Subscripts (i, j) follow the same canonical ordering as the embedding:
the lower-type-id object is always i (subject), the higher is j (object).

Note on padding
---------------
Different images may have different numbers of objects (N varies across the
batch).  The current implementation assumes a *fixed* N per batch (i.e. the
caller has already padded/truncated to the same N).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.pair_builder import is_feasible_pair
from vired_model.config import ViredConfig
from vired_model.utils.tensor_shapes import assert_shape


# ────────────────────────────────────────────────────────────────────────── #
# Geometry feature computation (no learnable parameters)                    #
# ────────────────────────────────────────────────────────────────────────── #

def compute_pair_geometry_features(
    boxes: torch.Tensor,
    pair_indices: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Compute a 6-dimensional geometry vector for each candidate pair.

    All computations are fully vectorised over (B, P_max).

    Args:
        boxes:        (B, N, 4)  – bounding boxes in image pixel coordinates,
                                   format (x1, y1, x2, y2).  Degenerate boxes
                                   (zero width/height) are clamped to 1 pixel.
        pair_indices: (B, P_max, 2)  – each row is (i, j) into the N objects.
        image_size:   int        – image side length used for normalisation.

    Returns:
        geom_features: (B, P_max, 6)
            dim 0 – dx           : normalised horizontal centre offset
            dim 1 – dy           : normalised vertical centre offset
            dim 2 – dist         : normalised Euclidean distance between centres
            dim 3 – log_w_ratio  : log(width_j / width_i)
            dim 4 – log_h_ratio  : log(height_j / height_i)
            dim 5 – iou          : intersection-over-union
    """
    B, N, _ = boxes.shape
    _, P, _ = pair_indices.shape

    idx_i = pair_indices[:, :, 0]   # (B, P)  – subject index
    idx_j = pair_indices[:, :, 1]   # (B, P)  – object index

    # Expand indices to gather all four box coordinates at once.
    idx_i4 = idx_i.unsqueeze(-1).expand(B, P, 4)   # (B, P, 4)
    idx_j4 = idx_j.unsqueeze(-1).expand(B, P, 4)   # (B, P, 4)

    boxes_i = boxes.gather(1, idx_i4)   # (B, P, 4)
    boxes_j = boxes.gather(1, idx_j4)   # (B, P, 4)

    # Box dimensions.  Clamp to ≥ 1 pixel to avoid log(0) / division by zero.
    w_i = (boxes_i[..., 2] - boxes_i[..., 0]).clamp(min=1.0)   # (B, P)
    h_i = (boxes_i[..., 3] - boxes_i[..., 1]).clamp(min=1.0)
    w_j = (boxes_j[..., 2] - boxes_j[..., 0]).clamp(min=1.0)
    h_j = (boxes_j[..., 3] - boxes_j[..., 1]).clamp(min=1.0)

    # Centre coordinates.
    cx_i = (boxes_i[..., 0] + boxes_i[..., 2]) * 0.5   # (B, P)
    cy_i = (boxes_i[..., 1] + boxes_i[..., 3]) * 0.5
    cx_j = (boxes_j[..., 0] + boxes_j[..., 2]) * 0.5
    cy_j = (boxes_j[..., 1] + boxes_j[..., 3]) * 0.5

    s = float(image_size)

    # 1–2. Normalised centre offset.
    dx = (cx_j - cx_i) / s                  # (B, P)
    dy = (cy_j - cy_i) / s

    # 3. Normalised Euclidean distance.
    dist = torch.sqrt(dx * dx + dy * dy)    # (B, P)

    # 4–5. Log size ratios.
    log_w_ratio = torch.log(w_j / w_i)     # (B, P)
    log_h_ratio = torch.log(h_j / h_i)

    # 6. Intersection over union.
    inter_x1 = torch.max(boxes_i[..., 0], boxes_j[..., 0])
    inter_y1 = torch.max(boxes_i[..., 1], boxes_j[..., 1])
    inter_x2 = torch.min(boxes_i[..., 2], boxes_j[..., 2])
    inter_y2 = torch.min(boxes_i[..., 3], boxes_j[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    intersection = inter_w * inter_h                        # (B, P)
    union = w_i * h_i + w_j * h_j - intersection
    iou = intersection / (union + 1e-6)                     # (B, P)

    geom = torch.stack(
        [dx, dy, dist, log_w_ratio, log_h_ratio, iou], dim=-1
    )   # (B, P, 6)

    assert_shape(geom, (B, P, 6), "pair_geometry_features")
    return geom


# ────────────────────────────────────────────────────────────────────────── #



class PairBuilder(nn.Module):
    """Enumerate candidate pairs and build their (optionally geometry-enriched)
    concatenated embeddings.

    This module has no learnable parameters; it is a nn.Module purely for
    consistency with the rest of the API (e.g. to allow .to(device) calls).
    """

    def __init__(self, config: ViredConfig) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------ #

    @staticmethod
    def _candidate_pair_indices(
        object_types: torch.Tensor,  # (N,)
        object_key_padding_mask: torch.Tensor, # (N,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (row_idx, col_idx) for all cross-type pairs.

        Each unordered pair is included exactly once.  Within each pair the
        object with the **lower type-id** is placed at row_idx so that the
        resulting pair embedding is always:
            concat(lower-type-id-emb, higher-type-id-emb)

        This guarantees a consistent embedding structure regardless of the
        order in which objects appear in the input tensor.  For example, with
        symbol=0 and text=1, the embedding is always
        concat(symbol_emb, text_emb) — never the reverse.
        """
        N = object_types.shape[0]
        rows, cols = [], []
        for i in range(N):
            for j in range(i + 1, N):
                ti = object_types[i].item()
                tj = object_types[j].item()
                if not object_key_padding_mask[i].item() and not object_key_padding_mask[j].item() and is_feasible_pair(ti, tj): # type: ignore
                    if ti <= tj:
                        rows.append(i)
                        cols.append(j)
                    else:
                        rows.append(j)
                        cols.append(i)
        device = object_types.device
        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(cols, dtype=torch.long, device=device),
        )

    @staticmethod
    def _all_pair_indices(object_key_mask: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (row_idx, col_idx) for all unordered pairs (i < j)."""
        valid_idx = torch.nonzero(~object_key_mask, as_tuple=False).squeeze(-1)
        if valid_idx.numel() < 2:
            row_idx = torch.zeros((0,), dtype=torch.long, device=device)
            col_idx = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            r, c = torch.triu_indices(valid_idx.numel(), valid_idx.numel(), offset=1, device=device)
            row_idx = valid_idx[r]
            col_idx = valid_idx[c]
        return row_idx, col_idx
        

    # ------------------------------------------------------------------ #

    def forward(
        self,
        object_tokens: torch.Tensor,
        object_types: torch.Tensor,
        object_key_padding_mask: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            object_tokens : (B, N, D)
            object_types  : (B, N)   – integer type ids
            object_key_padding_mask:   (B, N), bool
                                   True = ignore / padding
                                   False = valid object
            boxes         : (B, N, 4) – bounding boxes (x1,y1,x2,y2) in
                            image pixel coordinates; required when
                            config.use_geometry_features=True

        Returns:
            pair_embeddings : (B, P_max, 2*D [+ G])
                              G = config.geometry_feature_dim when
                              use_geometry_features=True and boxes is provided,
                              0 otherwise.
            pair_indices    : (B, P_max, 2)   – each row is (i, j)
            pair_padding_mask      : (B, P_max) False means pair is candidate
        """
        B, N, D = object_tokens.shape
        assert_shape(object_types, (B, N), "object_types")
        assert_shape(object_key_padding_mask, (B, N), "object_key_padding_mask")
        if object_key_padding_mask.dtype != torch.bool:
            raise TypeError("object_key_padding_mask must have dtype torch.bool")

        if self.config.use_geometry_features and boxes is None:
            raise ValueError(
                "boxes must be provided to PairBuilder when "
                "config.use_geometry_features=True"
            )

        all_pair_embs = []
        all_pair_idxs = []

        for b in range(B):
            types_b = object_types[b]      # (N,)
            tokens_b = object_tokens[b]    # (N, D)
            object_key_mask_b = object_key_padding_mask[b]

            if self.config.pair_mode == "all":
                row_idx, col_idx = self._all_pair_indices(object_key_mask=object_key_mask_b, device=tokens_b.device)
            else:  
                row_idx, col_idx = self._candidate_pair_indices(types_b, object_key_mask_b)
                

            if row_idx.numel() == 0:
                pair_embs = tokens_b.new_zeros((0, 2 * D))
                pair_idxs = tokens_b.new_zeros((0, 2), dtype=torch.long)
            else:
                emb_i = tokens_b[row_idx]                           # (P, D)
                emb_j = tokens_b[col_idx]                           # (P, D)
                pair_embs = torch.cat([emb_i, emb_j], dim=-1)      # (P, 2D)
                pair_idxs = torch.stack([row_idx, col_idx], dim=-1) # (P, 2)

            all_pair_embs.append(pair_embs)
            all_pair_idxs.append(pair_idxs)

        P_max = max(pair_embs.shape[0] for pair_embs in all_pair_embs)
        device = object_tokens.device
        pair_embeddings = torch.zeros((B, P_max, 2 * D), dtype=object_tokens.dtype, device=device)
        pair_indices = torch.zeros((B, P_max, 2), dtype=torch.long, device=device)
        pair_padding_mask = torch.ones((B, P_max), dtype=torch.bool, device=device)
        for b, (pair_embs_b, pair_idxs_b) in enumerate(zip(all_pair_embs, all_pair_idxs)):
            P_b = pair_embs_b.shape[0]
            pair_embeddings[b][: P_b] = pair_embs_b
            pair_indices[b][: P_b] = pair_idxs_b
            pair_padding_mask[b][: P_b] = False
            
        # ── Append geometry features ────────────────────────────────────── #
        if self.config.use_geometry_features and boxes is not None:
            assert_shape(boxes, (B, N, 4), "boxes")
            geom = compute_pair_geometry_features(
                boxes=boxes,
                pair_indices=pair_indices,
                image_size=self.config.image_size,
            )   # (B, P_max, 6)
            geom[pair_padding_mask] = 0
            pair_embeddings = torch.cat(
                [pair_embeddings, geom], dim=-1
            )   # (B, P_max, 2D+G)

        return pair_embeddings, pair_indices, pair_padding_mask
