# vired-pair

PyTorch implementation of the relation prediction architecture from:

> **ViRED: Prediction of Visual Relations in Engineering Drawings**
> https://arxiv.org/abs/2303.10584

Adapted for detecting **natural pairs** between symbol and text objects in electrical plans.

---

## What this is

This repository contains only the **model architecture and data pipeline** — no training loop, no evaluation scripts, no experiment management. Those are added separately.

**A natural pair** is a (symbol, text) pairing where the text label describes the symbol. The goal is to detect all such pairs in an electrical plan image.

---

## Project structure

```
vired_model/
    config.py                  # ViredConfig dataclass — all hyperparameters
    models/
        vision_encoder.py      # timm backbone → image tokens + spatial feature map
        object_encoder.py      # mask CNN + ROIAlign → object embeddings
        relation_decoder.py    # transformer decoder (object tokens attend to image)
        pair_builder.py        # candidate pair construction + geometry features
        relation_head.py       # MLP classifier → relation logits
        vired_model.py         # ViredRelationModel top-level class
    utils/
        tensor_shapes.py       # assert_shape helper
    tests/
        test_forward_pass.py   # smoke tests for the full forward pass

datasets/
    plan_relation_dataset.py   # ViREDDataset (PyTorch Dataset)
    tests/
        test_pair_generation.py

utils/
    pair_builder.py            # object type encoding, pair generation (no PyTorch)

data.py                        # dataset slicing utility (slice_dataset)
ARCHITECTURE_NOTES.md          # architecture details, tensor shapes, self-audit
```

---

## Architecture overview

```
image  ──► VisionEncoder  ──► image tokens (B, T, D)
                          └──► feature map (B, D_bk, H_p, W_p)
                                          │
object masks ──► ObjectEncoder ◄──────────┘  (ROIAlign)
object boxes ──►               ──► object tokens (B, N, D)
object types ──►

object tokens + image tokens ──► RelationDecoder ──► decoded tokens (B, N, D)

decoded tokens ──► PairBuilder ──► pair embeddings (B, P, 2D+G)
object boxes   ──►               (geometry features appended)

pair embeddings ──► RelationHead ──► pair logits (B, P, 2)
                                     pair indices (B, P, 2)
```

---

## Object types

YOLO class IDs are mapped to three object types:

| Type        | Value | YOLO class IDs |
| ----------- | ----- | -------------- |
| TEXT        | 0     | 0, 2, 5        |
| SYMBOL      | 1     | 1, 3           |
| SYMBOL_TEXT | 2     | 4              |

Candidate pairs are generated for all cross-type combinations: SYMBOL ↔ TEXT, SYMBOL ↔ SYMBOL_TEXT, TEXT ↔ SYMBOL_TEXT.

---

## Dataset format

```
{split}/
    images/         # image files
    labels/         # YOLO format: class_id x_c y_c w h  (normalised)
    pair_labels/    # natural pairs: i j  per line  (0-based object indices)
```

After slicing (`data.py`), a fourth directory is added:

```
    object_maps/    # slice_idx  original_idx  per line
```

---

## Quick start

```python
from vired_model.config import ViredConfig
from vired_model.models.vired_model import ViredRelationModel

config = ViredConfig(
    vision_backbone="vit_small_patch16_384",
    image_size=384,
    num_object_types=3,
    roi_context_pad=32,
)
model = ViredRelationModel(config)

output = model(image, object_masks, object_boxes, object_types)
# output.pair_logits  : (B, P, 2)
# output.pair_indices : (B, P, 2)
```

Slice a dataset before training:

```python
from data import slice_dataset

slice_dataset("data/pair-1", slice_size=1280, overlap=0.2)
# writes → data/pair-1_sliced/
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Tests

```bash
pytest vired_model/tests/test_forward_pass.py -v
pytest datasets/tests/test_pair_generation.py -v
```
