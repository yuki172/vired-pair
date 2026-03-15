from typing import Optional, Tuple


import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import cv2

# Default normalisation statistics (ImageNet).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def clip_box_to_slice(
    x1: float, y1: float, x2: float, y2: float,
    x_off: int, y_off: int, slice_size: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Clip a bounding box (full-image pixel coords) to a tile window.

    Returns the clipped box in full-image pixel coordinates, or ``None``
    if the box does not intersect the tile at all.
    """
    cx1 = max(x1, x_off)
    cy1 = max(y1, y_off)
    cx2 = min(x2, x_off + slice_size)
    cy2 = min(y2, y_off + slice_size)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def get_default_transform(
    model_input_size: int = 384
):
    return transforms.Compose([
            transforms.Resize((model_input_size, model_input_size)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

def boxes_to_masks(
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