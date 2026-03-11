"""
datasets/plan_relation_dataset.py
==================================
PyTorch Dataset for the electrical-plan relation detection task.

Expected directory layout
--------------------------
::

    root_dir/
        {split}/
            images/           ← image files (*.jpg, *.png, …)
            labels/           ← YOLO format object annotations
            pair_labels/      ← ground truth natural pairs

Object annotation format (labels/{name}.txt)
---------------------------------------------
One object per line::

    class_id  x_center  y_center  width  height

All coordinates are normalised to [0, 1].  ``class_id`` must be in {0..5}
and is mapped to an object type via ``YOLO_CLASS_TO_OBJECT_TYPE``.

Pair annotation format (pair_labels/{name}.txt)
------------------------------------------------
One ground-truth natural pair per line::

    i  j

where ``i`` and ``j`` are **0-based** indices into the object list for that
image (i.e. into the corresponding labels file).

Object type encoding
--------------------
See ``utils.pair_builder`` for the full mapping.  Summary:

    0 = TEXT          (YOLO class_ids 0, 2, 5)
    1 = SYMBOL        (YOLO class_ids 1, 3)
    2 = SYMBOL_TEXT   (YOLO class_id  4)

__getitem__ return value
------------------------
A dict with the following keys:

    image_name    str            stem of the image file
    image         FloatTensor    (C, H_model, W_model) – resized & normalised
    object_boxes  FloatTensor    (N, 4) – (x1, y1, x2, y2) in model-space pixels
    object_masks  FloatTensor    (N, H_model, W_model) – binary bbox masks
    object_types  LongTensor     (N,) – values in {0, 1, 2}
    pair_indices  LongTensor     (P, 2) – (subject_idx, object_idx)
    pair_labels   LongTensor     (P,)   – 1 = natural pair, 0 = not

Notes
-----
- N (objects per image) may vary.  Batching with ``torch.utils.data.DataLoader``
  requires a custom ``collate_fn`` that pads N to the batch maximum.
- The pair ordering is canonical (lower type-id subject first), matching
  ``PairBuilder._cross_type_indices`` in the architecture.  When the model
  processes the same sample its ``pair_indices`` will coincide with the
  dataset's ``pair_indices``, so ``pair_labels`` can be used directly as
  training targets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import cv2

from utils.pair_builder import (
    YOLO_CLASS_TO_OBJECT_TYPE,
    generate_candidate_pairs,
    label_candidate_pairs,
    make_gt_pair_set,
    validate_object_types,
    yolo_class_ids_to_object_types,
)

logger = logging.getLogger(__name__)

# Default normalisation statistics (ImageNet).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

class ViREDDataset(Dataset):
    """
    Args
    ----------
    root_dir : str or Path
        Path to the dataset root containing split sub-directories.
    split : str
        Sub-directory name, e.g. ``"train"``, ``"valid"``, ``"test"``.
    image_size : int
        Square size that images (and masks) are resized to before being fed
        to the model.  Must match ``ViredConfig.image_size``.
    transform : callable, optional
        Additional image transform applied *after* the default resize +
        normalise pipeline.  Receives a ``(C, H, W)`` FloatTensor.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        image_size: int = 384,
        transform: Optional[Callable] = None,
    ) -> None:
        self.split_dir    = Path(root_dir) / split
        self.images_dir   = self.split_dir / "images"
        self.labels_dir   = self.split_dir / "labels"
        self.pairs_dir    = self.split_dir / "pair_labels"
        self.image_size   = image_size
        self.transform    = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}"
            )
        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Object labels directory not found: {self.labels_dir}"
            )

        self.image_paths: List[Path] = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        if len(self.image_paths) == 0:
            logger.warning(
                f"No images found in {self.images_dir} "
                f"with extensions {_IMAGE_EXTENSIONS}"
            )

        # Pre-build the default image transform.
        self._img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        logger.info(
            f"ViREDDataset [{split}]: "
            f"{len(self.image_paths)} images, image_size={image_size}"
        )


    def __len__(self) -> int:
        return len(self.image_paths)


    def __getitem__(self, idx: int) -> Dict:
        image_path   = self.image_paths[idx]
        name       = image_path.stem
        label_path = self.labels_dir / f"{name}.txt"
        pair_path  = self.pairs_dir  / f"{name}.txt"

        # load image, convert to tensor, apply transform
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to read image {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H_orig, W_orig = image.shape[:2]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor  = self._img_transform(image_tensor) 

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        H_m = W_m = self.image_size 

        # load labels
        boxes_raw, class_ids = _parse_yolo_labels(label_path, W_orig, H_orig)
        # boxes_raw: (N, 4) as float  (x1, y1, x2, y2) in ORIGINAL pixel coords
        # class_ids: List[int]

        N = len(class_ids)

        if N == 0:
            logger.warning(f"No objects found for {name}; returning empty sample.")

        object_types_list = yolo_class_ids_to_object_types(class_ids)

        # ── scale boxes to model-space ─────────────────────────────── #
        scale_x = W_m / W_orig
        scale_y = H_m / H_orig

        if N > 0:
            boxes_model = boxes_raw.clone()
            boxes_model[:, 0] *= scale_x   # x1
            boxes_model[:, 2] *= scale_x   # x2
            boxes_model[:, 1] *= scale_y   # y1
            boxes_model[:, 3] *= scale_y   # y2
        else:
            boxes_model = torch.zeros((0, 4), dtype=torch.float32)

        # ── generate binary masks ───────────────────────────────────── #
        object_masks = _boxes_to_masks(boxes_model, H_m, W_m)   # (N, H_m, W_m)

        # ── load pair ground truth ─────────────────────────────────── #
        gt_pairs_raw  = _parse_pair_labels(pair_path, N)
        gt_pair_set   = make_gt_pair_set(gt_pairs_raw)

        # ── generate candidate pairs ───────────────────────────────── #
        candidate_pairs = generate_candidate_pairs(object_types_list)
        pair_labels_list = label_candidate_pairs(candidate_pairs, gt_pair_set)

        # ── build output tensors ────────────────────────────────────── #
        if len(candidate_pairs) > 0:
            pair_indices = torch.tensor(candidate_pairs, dtype=torch.long)  # (P, 2)
            pair_labels  = torch.tensor(pair_labels_list, dtype=torch.long) # (P,)
        else:
            pair_indices = torch.zeros((0, 2), dtype=torch.long)
            pair_labels  = torch.zeros((0,),   dtype=torch.long)

        object_types_tensor = torch.tensor(object_types_list, dtype=torch.long)  # (N,)


        return {
            "image_name":   name,
            "image":        image_tensor,   # (C, H_m, W_m)
            "object_boxes": boxes_model,    # (N, 4)
            "object_masks": object_masks,   # (N, H_m, W_m)
            "object_types": object_types_tensor,  # (N,)
            "pair_indices": pair_indices,   # (P, 2)
            "pair_labels":  pair_labels,    # (P,)
        }


def _parse_yolo_labels(
    label_path: Path,
    img_w: float,
    img_h: float,
) -> Tuple[torch.Tensor, List[int]]:
    """Parse a YOLO label file.

    Args:
        label_path: path to ``{name}.txt``.  If the file does not exist an
                    empty result is returned (images with no objects are valid).
        img_w: original image width  in pixels.
        img_h: original image height in pixels.

    Returns:
        boxes:     (N, 4) float tensor – (x1, y1, x2, y2) in pixel coordinates.
        class_ids: list of N integer class ids.
    """
    if not label_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32), []

    boxes: List[List[float]] = []
    class_ids: List[int] = []

    with open(label_path, "r") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) != 5:
                raise Exception(f"{label_path}:{lineno}: expected 5 fields, got {len(parts)}")

            cid   = int(parts[0])
            x_c   = float(parts[1])
            y_c   = float(parts[2])
            norm_w = float(parts[3])
            norm_h = float(parts[4])

            if cid not in YOLO_CLASS_TO_OBJECT_TYPE:
                raise Exception(f"{label_path}:{lineno}: unknown class_id {cid} – skipping.")

            # Convert YOLO normalised (x_c, y_c, w, h) → pixel (x1, y1, x2, y2)
            x1 = (x_c - norm_w / 2) * img_w
            y1 = (y_c - norm_h / 2) * img_h
            x2 = (x_c + norm_w / 2) * img_w
            y2 = (y_c + norm_h / 2) * img_h

            # Clamp to image bounds.
            x1 = max(0.0, min(x1, img_w))
            y1 = max(0.0, min(y1, img_h))
            x2 = max(0.0, min(x2, img_w))
            y2 = max(0.0, min(y2, img_h))

            if not (x1 < x2 and y1 < y2):
                raise Exception(f"invalid labels, f{label_path}")

            boxes.append([x1, y1, x2, y2])
            class_ids.append(cid)

    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), class_ids
    return torch.zeros((0, 4), dtype=torch.float32), []


def _parse_pair_labels(
    pair_path: Path,
    N: int
) -> List[Tuple[int, int]]:
    """Parse a pair label file.

    Each line contains two space-separated integers ``i j`` (0-based object
    indices).  If the file does not exist, an empty list is returned (valid for
    images that have no annotated natural pairs).

    N: number of objects
    Returns:
        List of (i, j) integer tuples.
    """
    if not pair_path.exists():
        return []

    pairs: List[Tuple[int, int]] = []
    with open(pair_path, "r") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()

            if len(parts) != 2:
                raise Exception(f"{pair_path}:{lineno}: expected 2 fields, got {len(parts)}")

            i, j = int(parts[0]), int(parts[1])
            if not (0 <= i < N and 0 <= j < N):
                raise Exception(f"{pair_path}: invalid pair, ({i}, {j}), total {N}")
            pairs.append((int(parts[0]), int(parts[1])))

    return pairs


def _boxes_to_masks(
    boxes: torch.Tensor,  # (N, 4) float – x1,y1,x2,y2 in model-space pixels
    H: int,
    W: int,
) -> torch.Tensor:
    """Create binary masks by filling each bounding box region with 1.

    Args:
        boxes: (N, 4) float tensor of bounding boxes in (x1, y1, x2, y2)
               model-space pixel coordinates.
        H: mask height in pixels.
        W: mask width  in pixels.

    Returns:
        masks: (N, H, W) float32 tensor with 1.0 inside each box, 0.0 outside.
    """
    N = boxes.shape[0]
    masks = torch.zeros((N, H, W), dtype=torch.float32)

    for i in range(N):
        x1, y1, x2, y2 = boxes[i].tolist()
        # Convert to integer pixel indices (inclusive).
        ix1 = max(0, int(torch.floor(boxes[i, 0]).item()))
        iy1 = max(0, int(torch.floor(boxes[i, 1]).item()))
        ix2 = min(W, int(torch.ceil(boxes[i, 2]).item()))
        iy2 = min(H, int(torch.ceil(boxes[i, 3]).item()))
        if ix2 > ix1 and iy2 > iy1:
            masks[i, iy1:iy2, ix1:ix2] = 1.0

    return masks


def vired_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    """
    Collate function for ViREDDataset.

    Input: list of samples, each containing:
        image_name:    str
        image:         (C, H, W)
        object_masks:  (N_i, H, W)
        object_boxes:  (N_i, 4)
        object_types:  (N_i,)
        pair_indices:  (P_i, 2)
        pair_labels:   (P_i,)

    Returns:
        image_names:               list of length B, each str
        images:                    (B, C, H, W)
        object_masks:              (B, N_max, H, W)
        object_boxes:              (B, N_max, 4)
        object_types:              (B, N_max)
        object_key_padding_mask:   (B, N_max), bool
                                   True = ignore / padding
                                   False = valid object
        pair_indices:              (B, P_max, 2)
        pair_labels:               (B, P_max)
        pair_labels_padding_mask:  (B, P_max) - true means ignore that pair
        num_objects:               (B,)
    """
    if len(batch) == 0:
        raise ValueError("Received empty batch.")
    
    image_names = [sample["image_name"] for sample in batch]

    images = torch.stack([sample["image"] for sample in batch], dim=0)
    B, C, H, W = images.shape

    num_objects_list = [sample["object_masks"].shape[0] for sample in batch]
    N_max = max(num_objects_list)

    num_pairs_list = [sample["pair_labels"].shape[0] for sample in batch]
    P_max = max(num_pairs_list)

    device = images.device
    dtype = images.dtype

    object_masks = torch.zeros((B, N_max, H, W), dtype=dtype, device=device)
    object_boxes = torch.zeros((B, N_max, 4), dtype=torch.float32, device=device)
    object_types = torch.zeros((B, N_max), dtype=torch.long, device=device)

    # True = ignore, so start fully padded
    object_key_padding_mask = torch.ones((B, N_max), dtype=torch.bool, device=device)

    pair_indices = torch.zeros((B, P_max, 2), dtype=torch.long, device=device)
    pair_labels = torch.zeros((B, P_max), dtype=torch.long, device=device)

    # True = ignore, so start fully padded
    pair_labels_padding_mask = torch.ones((B, P_max), dtype=torch.bool, device=device)

    for b, sample in enumerate(batch):
        N_b = sample["object_masks"].shape[0]

        object_masks[b, :N_b] = sample["object_masks"]
        object_boxes[b, :N_b] = sample["object_boxes"]
        object_types[b, :N_b] = sample["object_types"]

        # real objects => False
        object_key_padding_mask[b, :N_b] = False

        pair_indices_b = sample["pair_indices"]
        pair_labels_b = sample["pair_labels"]
        
        if pair_indices_b.ndim != 2 or pair_indices_b.shape[-1] != 2:
            raise ValueError(
                f"pair_indices for sample {b} must have shape (P, 2), got {tuple(pair_indices_b.shape)}"
            )

        P_b = pair_indices_b.shape[0]
        pair_indices[b, :P_b] = pair_indices_b
        pair_labels[b, :P_b] = pair_labels_b
        pair_labels_padding_mask[b, :P_b] = False
        


    num_objects = torch.tensor(num_objects_list, dtype=torch.long, device=device)

    return {
        "image_names": image_names,
        "images": images,
        "object_masks": object_masks,
        "object_boxes": object_boxes,
        "object_types": object_types,
        "object_key_padding_mask": object_key_padding_mask,
        "pair_indices": pair_indices,
        "pair_labels": pair_labels,
        "pair_labels_padding_mask": pair_labels_padding_mask,
        "num_objects": num_objects,
    }


if __name__ == "__main__":
    config = {
        "data_path": "data/train"
    }

    dataset = ViREDDataset(root_dir="data", split="train")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=vired_collate_fn,
    )

    for i, batch in enumerate(loader):
        images = batch["images"]                          # (B, C, H, W)
        object_masks = batch["object_masks"]              # (B, N_max, H, W)
        object_boxes = batch["object_boxes"]              # (B, N_max, 4)
        object_types = batch["object_types"]              # (B, N_max)
        object_key_padding_mask = batch["object_key_padding_mask"]  # (B, N_max)
        pair_labels = batch["pair_labels"]                # list of B tensors

        print(f" === batch {i} shapes")
        print("images", images.shape)
        print("object_boxes", object_boxes.shape)
        print("object_types", object_types.shape)
        print("object_key_padding_mask", object_key_padding_mask.shape)
        print("pair_labels")
        for pair_label in pair_labels:
            print(pair_label.shape)
        print("\n")

