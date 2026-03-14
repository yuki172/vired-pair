import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import warnings
from vired_model import ViredConfig
from vired_model.models.vired_model import ViREDModel
import torch.nn as nn
from utils.data import clip_box_to_slice, get_default_transform, boxes_to_masks

import logging

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

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_absolute_coords(
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
    img_w: int,
    img_h: int,
    clip_to_fit: bool = False
) -> Tuple[int, int, int, int]:
    """Convert normalized YOLO coordinates to absolute pixel coordinates."""
    x_center = x_center_norm * img_w
    y_center = y_center_norm * img_h
    width = width_norm * img_w
    height = height_norm * img_h
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    if clip_to_fit:
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))
    
    return x1, y1, x2, y2

def inference_batch(
    model: ViREDModel,
    batch: Dict,
    device: str,
    threshold: float
):
    image = batch['images'].to(device)
    object_masks = batch['object_masks'].to(device)
    object_boxes = batch['object_boxes'].to(device)
    object_types = batch['object_types'].to(device)
    object_key_padding_mask = batch['object_key_padding_mask'].to(device)
    output = model(
                image, 
                object_masks, 
                object_boxes, 
                object_types,
                object_key_padding_mask
            )
    pair_logits = output["pair_logits"] # (B, P_max, C), C = 2
    pair_indices_pred = output["pair_indices"] # (B, P_max, 2)
    pair_padding_mask_pred = output["pair_padding_mask"] # (B, P_max), True if is padding

    softmax = nn.Softmax(dim=-1)
    pair_probs = softmax(pair_logits)[..., -1]
    pairs_mask = (~pair_padding_mask_pred) & (pair_probs > threshold)

    B = pair_indices_pred.shape[0]

    pair_indices_batch = [pair_indices_pred[b][pairs_mask[b]].tolist() for b in range(B)]
    
    return pair_indices_batch

def process_batch(
    batch_slice_images: List[np.ndarray],
    batch_slice_meta: List[Dict],
    object_labels: List[Dict],
    slice_size: int,
    model_input_size: int
):
    batch_no_padding = []
    transform = get_default_transform(
        model_input_size=model_input_size
    )
    batch_slice_index_to_original = []
    for b, (slice_image, slice_meta) in enumerate(zip(batch_slice_images, batch_slice_meta)):
        object_types = []
        object_boxes = []
        object_masks = []
        slice_image = cv2.cvtColor(slice_image, cv2.COLOR_BGR2RGB)
        H_orig, W_orig = slice_image.shape[:2]

        slice_image_tensor = torch.from_numpy(slice_image).permute(2, 0, 1).float() / 255.0
        slice_image_tensor  = transform(slice_image_tensor) 

        H_m = W_m = model_input_size

        boxes_raw, class_ids = [], []
        slice_index_to_original = dict()
        for original_i, label in enumerate(object_labels):
            class_id, x1, y1, x2, y2 = label['class_id'], label['x1'], label['y1'], label['x2'], label['y2']
            x_off, y_off = slice_meta['x_off'], slice_meta['y_off']
            clipped = clip_box_to_slice(x1, y1, x2, y2, x_off, y_off, slice_size)
            if clipped is None:
                continue  # no intersection with this tile

            x1_clip, y1_clip, x2_clip, y2_clip = clipped
            slice_index_to_original[len(boxes_raw)] = original_i
            boxes_raw.append([x1_clip, y1_clip, x2_clip, y2_clip])
            class_ids.append(class_id)



        boxes_raw = torch.tensor(boxes_raw)
        object_types_list = yolo_class_ids_to_object_types(class_ids)
        object_types_tensor = torch.tensor(object_types_list, dtype=torch.long)  # (N,)

          # ── scale boxes to model-space ─────────────────────────────── #
        scale_x = W_m / W_orig
        scale_y = H_m / H_orig

        boxes_model = boxes_raw.clone()
        boxes_model[:, 0] *= scale_x   # x1
        boxes_model[:, 2] *= scale_x   # x2
        boxes_model[:, 1] *= scale_y   # y1
        boxes_model[:, 3] *= scale_y   # y2

        # ── generate binary masks ───────────────────────────────────── #
        object_masks = boxes_to_masks(boxes_model, H_m, W_m)   # (N, H_m, W_m)

        batch_no_padding.append({
            "image":        slice_image_tensor,   # (C, H_m, W_m)
            "object_boxes": boxes_model,    # (N, 4)
            "object_masks": object_masks,   # (N, H_m, W_m)
            "object_types": object_types_tensor,  # (N,)
        })
        batch_slice_index_to_original.append(slice_index_to_original)

    batch_with_padding = collate_batch(batch_no_padding)
    return batch_with_padding, batch_slice_index_to_original

def collate_batch(
    batch: List[Dict[str, torch.Tensor]]
):
    if len(batch) == 0:
        raise ValueError("Received empty batch.")
    

    images = torch.stack([sample["image"] for sample in batch], dim=0)
    B, C, H, W = images.shape

    num_objects_list = [sample["object_masks"].shape[0] for sample in batch]
    N_max = max(num_objects_list)

    device = images.device
    dtype = images.dtype

    object_masks = torch.zeros((B, N_max, H, W), dtype=dtype, device=device)
    object_boxes = torch.zeros((B, N_max, 4), dtype=torch.float32, device=device)
    object_types = torch.zeros((B, N_max), dtype=torch.long, device=device)

    # True = ignore, so start fully padded
    object_key_padding_mask = torch.ones((B, N_max), dtype=torch.bool, device=device)

    for b, sample in enumerate(batch):
        N_b = sample["object_masks"].shape[0]

        object_masks[b, :N_b] = sample["object_masks"]
        object_boxes[b, :N_b] = sample["object_boxes"]
        object_types[b, :N_b] = sample["object_types"]

        # real objects => False
        object_key_padding_mask[b, :N_b] = False

    return {
        "images": images,
        "object_masks": object_masks,
        "object_boxes": object_boxes,
        "object_types": object_types,
        "object_key_padding_mask": object_key_padding_mask,
    }

        
        



def inference_pairs(
    model: ViREDModel,
    image: np.ndarray,
    object_labels: List[Dict],
    model_input_size: int = 384,
    slice_size: int   = 640,
    overlap: float = 0.25,
    threshold: float = 0.25,
    batch_size: int = 16,
    device: str = 'cpu'
):

    batch_slice_images = []
    batch_slice_meta = []
    h, w = image.shape[:2]
    step = int(slice_size * (1 - overlap))

    y_positions = list(range(0, h, step))
    x_positions = list(range(0, w, step))

    if y_positions and (y_positions[-1] + slice_size < h):
        y_positions.append(h - slice_size)
    if x_positions and (x_positions[-1] + slice_size < w):
        x_positions.append(w - slice_size)

    all_pair_indices = set()

    def _infer_batch():
        batch_with_padding, batch_slice_index_to_original = process_batch(
            batch_slice_images,
            batch_slice_meta,
            object_labels,
            slice_size,
            model_input_size
        )
        pair_indices_batch = inference_batch(
            model=model,
            batch=batch_with_padding,
            device=device,
            threshold=threshold
        )
        pair_indices_original = [[[batch_slice_index_to_original[b][i] for i in pair] for pair in pairs] for b, pairs in enumerate(pair_indices_batch)]

        pair_indices = set()
        for pairs in pair_indices_original:
            pair_indices.update(set([frozenset(pair) for pair in pairs]))
        all_pair_indices.update(pair_indices)

    for y_pos in y_positions:
        for x_pos in x_positions:
            y_end = y_pos + slice_size
            x_end = x_pos + slice_size

            if y_end > h:
                y_end = h
            if x_end > w:
                x_end = w

            actual_h = y_end - y_pos
            actual_w = x_end - x_pos

            slice_img = image[y_pos:y_end, x_pos:x_end]

            if actual_h < slice_size or actual_w < slice_size:
                padded = np.zeros((slice_size, slice_size, 3), dtype=slice_img.dtype)
                padded[:actual_h, :actual_w] = slice_img
                slice_img = padded

            batch_slice_images.append(slice_img)
            batch_slice_meta.append({"x_off": x_pos, "y_off": y_pos})

            if len(batch_slice_images) >= batch_size:
                _infer_batch()

    _infer_batch()

    logger.info(f"Found {len(all_pair_indices)} pairs")

    return all_pair_indices

