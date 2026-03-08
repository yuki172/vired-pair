# ViRED Architecture Notes

## Summary

This is a clean PyTorch re-implementation of the core model architecture from:

> **ViRED: Prediction of Visual Relations in Engineering Drawings**

The implementation targets a *relation detection* task in electrical plans:
given an image containing **symbols** and **text** objects, predict which
(symbol, text) pairs are "natural pairs" — i.e. the text describes the symbol.

---

## High-Level Pipeline

```
image (B, C, H, W)
  └─ VisionEncoder ──────────────────────────────────► image_tokens  (B, T, D)
                                                              │
object_masks (B, N, H, W)                                    │
object_types (B, N)                                          │
  └─ ObjectEncoder ──► object_tokens (B, N, D)               │
                              │                               │
                              └──► RelationDecoder ◄──────────┘
                                         │
                              decoded_tokens (B, N, D)
                                         │
                              PairBuilder ──► pair_embeddings (B, P, 2D)
                                                    pair_indices    (B, P, 2)
                                         │
                              RelationHead ──► pair_logits (B, P, C)
```

---

## Tensor Shapes

| Tensor              | Shape         | Notes                                      |
|---------------------|---------------|--------------------------------------------|
| `image`             | (B, C, H, W)  | Input image, normalised float32            |
| `object_masks`      | (B, N, H, W)  | Binary float masks, one per object         |
| `object_types`      | (B, N)        | Integer type ids, e.g. 0=symbol, 1=text   |
| `image_tokens`      | (B, T, D)     | T = number of ViT patch tokens             |
| `object_tokens`     | (B, N, D)     | Raw CNN object embeddings                  |
| `decoded_tokens`    | (B, N, D)     | Object tokens after transformer fusion     |
| `pair_embeddings`   | (B, P, 2D)    | Concatenated embeddings for each pair      |
| `pair_indices`      | (B, P, 2)     | (i, j) index into the N-object list        |
| `pair_logits`       | (B, P, C)     | Raw logits; C=2 for binary classification  |

Where:
- `B` = batch size
- `C` = image channels (3)
- `H`, `W` = image height / width
- `N` = number of objects per image
- `T` = number of image tokens (e.g. 196 for ViT-S/16 at 224×224)
- `D` = embedding dimension (default 256)
- `P` = number of candidate pairs
- `C` = number of relation classes (default 2)

For `pair_mode="cross_type"` with `n_sym` symbols and `n_txt` texts:
```
P = n_sym × n_txt
```
For `pair_mode="all"`:
```
P = N × (N−1) / 2
```

---

## Component Descriptions

### 1. Vision Encoder (`models/vision_encoder.py`)

Wraps a **timm** backbone (default: `vit_tiny_patch16_224`).  Global pooling is
disabled so the full sequence of patch tokens is returned.  A linear projection
maps the backbone's native dimension to `embedding_dim`.

**Assumption:** The paper does not specify which exact vision backbone is used.
We expose `vision_backbone` in the config so any timm model can be substituted.

---

### 2. Object Encoder (`models/object_encoder.py`)

A **3-layer CNN** processes each binary object mask independently:

```
Conv(1→C1, 3×3) → ReLU → MaxPool(2)
Conv(C1→C2, 3×3) → ReLU → MaxPool(2)
Conv(C2→C3, 3×3) → ReLU → AdaptiveAvgPool(1×1)
→ Flatten → Linear(C3, D) → LayerNorm
```

All objects are processed in a single batched forward pass by folding the
`(B, N)` dimensions together before the CNN.

**Optional type embeddings:** Learned embeddings indexed by `object_types`
are added after the CNN projection.  This lets the model distinguish symbol
tokens from text tokens even after the decoder has mixed their representations.

---

### 3. Relation Decoder (`models/relation_decoder.py`)

A stack of standard transformer decoder layers.  Each layer contains:

1. **Self-attention** over object tokens — objects can attend to each other.
2. **Cross-attention** — object tokens attend to image tokens, grounding each
   object in its visual context.
3. **FFN** — position-wise feed-forward network.
4. **Residual + LayerNorm + Dropout** after each sub-layer.

**Assumption:** The paper describes a transformer-style decoder but does not
detail the exact normalisation order (pre-norm vs post-norm).  This
implementation uses **post-norm** (norm applied after residual addition), which
is the standard original transformer convention.

---

### 4. Pair Builder (`models/pair_builder.py`)

Enumerates candidate pairs and constructs their joint representation:

```
pair_embedding[k] = concat(decoded_tokens[i], decoded_tokens[j])
```

Two modes:

| Mode          | Pairs generated                              |
|---------------|----------------------------------------------|
| `cross_type`  | All (i, j) with `type[i] ≠ type[j]`, `i<j` |
| `all`         | All (i, j) with `i < j`                     |

**Assumption:** The paper constructs all cross-type pairs.  We also expose an
`"all"` mode for datasets without meaningful type distinctions.

**Batch constraint:** The implementation requires all samples in a batch to
produce the same number of pairs (i.e. same `N` and same type distribution).
For variable-length object sets, the caller should pad objects and pass an
`object_key_padding_mask`.

---

### 5. Relation Head (`models/relation_head.py`)

Three-layer MLP operating on the `2D`-dimensional pair embeddings:

```
Linear(2D → H) → ReLU → Linear(H → H) → ReLU → Linear(H → C)
```

Outputs raw logits.  No activation is applied at the output — the caller should
apply `softmax` (multi-class) or `sigmoid` (binary score on class 1) depending
on their loss function.

---

## Assumptions and Design Decisions

| Topic | Decision | Reason |
|-------|----------|--------|
| Backbone | timm ViT | Paper uses ViT; timm gives access to many variants |
| Patch token vs CLS | Keep all tokens (no CLS removal) | Decoder can freely attend to all positions |
| Norm order | Post-norm (original transformer) | Paper does not specify; post-norm is the common default |
| Type embeddings | Learned, added after CNN projection | Standard practice; absent in paper but clearly useful |
| Pair construction | Concatenation | Paper states concatenation; alternatives (sum, MLP-fused) not used |
| Pair ordering | `i < j` enforced | Prevents duplicate (i,j) and (j,i) pairs |
| Variable-N batching | Fixed N per batch required | Simplifies implementation; variable-N support delegated to caller |
| Mask resolution | Same as image resolution | Simplest choice; downsampling is a dataset preprocessing concern |
| Backbone pretrained weights | `pretrained=False` default | Architecture-only module; caller controls weight loading |

---

## SELF-AUDIT

This section documents every assumption in the codebase, risk level, and
the remediation applied.

---

### Finding 1 — Wrong comment: "pre-norm style" label on a post-norm block

| Field | Detail |
|---|---|
| File | `models/relation_decoder.py` → `RelationDecoderLayer.forward` |
| Class/function | `RelationDecoderLayer.forward` |
| Risk | **HIGH** |

**Description.**  The original code performed post-norm (residual first, then
LayerNorm) but its inline comment read `"Self-attention (pre-norm style)"`.
This is directly contradictory.  Post-norm is harder to train at depth and
inconsistent with the ViT backbone, which uses pre-norm.

**Alternative (applied).**  Switch to genuine **pre-norm**: LayerNorm is
applied to the residual stream *before* it is passed into each sub-layer, and
the clean (unnormalised) residual is added back.  This matches ViT convention
and is more stable to train.  The normalisation order is now configurable via
`config.decoder_pre_norm` (default `True`).  A final `LayerNorm` is applied
after the last decoder layer in pre-norm mode (also matching ViT).

---

### Finding 2 — Pair ordering bug: docstring said type-id order, code used index order

| Field | Detail |
|---|---|
| File | `models/pair_builder.py` → `PairBuilder._cross_type_indices` |
| Class/function | `_cross_type_indices` |
| Risk | **HIGH** |

**Description.**  The docstring stated "the lower type-id object is always
first", but the implementation enforced `i < j` (lower *index* first).  For a
batch where text objects (type 1) happen to appear before symbol objects
(type 0) in the object list, the pair embedding would be
`concat(text_emb, symbol_emb)` — the opposite of what the docstring promised.
The relation head thus sees inconsistent input structure during training,
which could prevent it from learning a reliable decision function.

**Alternative (applied).**  For each cross-type pair the lower-type-id object
is now always placed in position 0 of the pair embedding, regardless of
which object has the smaller list index.  The resulting `pair_indices[b, k]`
tensor always satisfies `types[b, pair_indices[b, k, 0]] ≤ types[b, pair_indices[b, k, 1]]`.

---

### Finding 3 — `predict()` left dropout active (stochastic inference)

| Field | Detail |
|---|---|
| File | `models/vired_model.py` → `ViredRelationModel.predict` |
| Class/function | `predict` |
| Risk | **HIGH** |

**Description.**  `predict()` wrapped the forward call in `torch.no_grad()`
but did not call `self.eval()`.  When the model was in training mode, all
`nn.Dropout` layers remained active, making predictions stochastic.  A caller
who forgot to call `model.eval()` before inference would silently receive
different logits on every call for the same input.

**Alternative (applied).**  `predict()` now saves the current training flag,
switches to eval mode, runs the forward pass, and restores the original mode
in a `finally` block.  This is the same pattern used by PyTorch's own
`torch.inference_mode` context.  Additionally, the unused `threshold` parameter
(which was accepted but never acted on) has been removed to eliminate a
misleading API surface.

---

### Finding 4 — Backbone feature dimension inferred via a dummy forward pass

| Field | Detail |
|---|---|
| File | `models/vision_encoder.py` → `VisionEncoder._get_backbone_dim` |
| Class/function | `_get_backbone_dim` |
| Risk | **LOW** |

**Description.**  `_get_backbone_dim` ran a full forward pass through the
backbone using a zero tensor just to read the output dimension.  This wastes
compute at construction time and also triggers any lazy parameter
initialisation inside timm models, potentially delaying CUDA memory
allocation.

**Alternative (applied).**  The method now reads `backbone.num_features` first
(a standard timm attribute that almost all models expose without a forward
pass).  The dummy forward pass is retained as a fallback for any unusual
backbone that does not expose this attribute.

---

### Finding 5 — Relation head had no dropout and a hardcoded activation

| Field | Detail |
|---|---|
| File | `models/relation_head.py` → `RelationHead.__init__` |
| Class/function | `RelationHead` |
| Risk | **MEDIUM** |

**Description.**  The MLP head used hardcoded `nn.ReLU` activations and had no
dropout between layers, while the rest of the model (decoder) used dropout and
a configurable activation.  This inconsistency could cause under-regularisation
of the head on small datasets, and using ReLU in the MLP while the decoder used
GELU would create a mismatch in gradient magnitude at the interface.

**Alternative (applied).**  The head now uses:
- `config.ffn_activation` for its activation (same as the decoder FFN).
- `config.relation_head_dropout` between each linear layer.  This is a
  separate knob from `config.dropout` so the head can be tuned independently.

---

### Finding 6 — `pretrained` flag hardcoded to `False` inside `VisionEncoder`

| Field | Detail |
|---|---|
| File | `models/vision_encoder.py` → `VisionEncoder.__init__` |
| Class/function | `VisionEncoder.__init__` |
| Risk | **LOW** |

**Description.**  `timm.create_model(..., pretrained=False)` was hardcoded.
A user wishing to fine-tune from ImageNet weights had no way to enable this
through the config.

**Alternative (applied).**  `config.pretrained` (default `False`) is now
forwarded to `timm.create_model`.

---

### Finding 7 — Decoder activation hardcoded to `ReLU`; inconsistent with ViT

| Field | Detail |
|---|---|
| File | `models/relation_decoder.py` → `RelationDecoderLayer.__init__` |
| Class/function | `RelationDecoderLayer` |
| Risk | **MEDIUM** |

**Description.**  The decoder FFN used `nn.ReLU`, while the ViT backbone it
works with internally uses `nn.GELU`.  Using ReLU in the decoder while feeding
it outputs from a GELU-based encoder can slow convergence.

**Alternative (applied).**  `config.ffn_activation` (default `"gelu"`)
controls the activation in both the decoder FFN and the relation head MLP.
`_make_activation()` in `relation_decoder.py` is the single factory for both.

---

### Training vs Inference Mathematical Consistency

| Check | Status |
|---|---|
| Logits are always raw (no softmax/sigmoid at output) | ✓ Consistent |
| `nn.CrossEntropyLoss` can be applied directly to `pair_logits` | ✓ Consistent |
| `nn.BCEWithLogitsLoss` on `pair_logits[:, :, 1]` is valid for binary tasks | ✓ Consistent |
| `predict()` uses eval mode → no stochastic dropout at inference | ✓ Fixed (was broken) |
| Pair embedding order is consistent between train and predict calls | ✓ Fixed (was broken) |
| Pre-norm final LayerNorm in decoder is applied consistently in both modes | ✓ Consistent |
| `relation_head_dropout=0.0` default → no random masking at inference in eval | ✓ Consistent |

**Double-softmax warning (user responsibility).**  If a user applies
`torch.softmax(logits, dim=-1)` before passing to `nn.CrossEntropyLoss`, the
result will be mathematically wrong (softmax of softmax).  This is documented
in the `RelationHead` and `predict()` docstrings.

---

### Config Coverage After Audit

The following were previously hardcoded and are now in `ViredConfig`:

| Parameter | Config field | Default |
|---|---|---|
| Backbone pretrained weights | `pretrained` | `False` |
| Decoder normalisation order | `decoder_pre_norm` | `True` |
| FFN / MLP activation function | `ffn_activation` | `"gelu"` |
| Relation head dropout | `relation_head_dropout` | `0.0` |

---

## What Is NOT Implemented

- Dataset loading or preprocessing
- Training loop
- Evaluation or metrics
- Loss functions
- Experiment management
- Inference scripts

These are intentionally excluded so the architecture can be integrated into any
pipeline.
