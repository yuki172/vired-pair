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
from utils.data import clip_box_to_slice, get_default_transform, boxes_to_masks, IMAGE_EXTENSIONS
from inference import inference_pairs
from utils.pair_builder import OBJECT_TYPE_NAMES, TEXT_OBJECT_CLASS, SYMBOL_TEXT_OBJECT_CLASS, SYMBOL_OBJECT_CLASS, YOLO_CLASS_TO_OBJECT_TYPE, yolo_class_id_to_object_type
import torch 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLORS = {
    TEXT_OBJECT_CLASS: (0, 0, 255),
    SYMBOL_TEXT_OBJECT_CLASS: (0, 165, 255),
    SYMBOL_OBJECT_CLASS: (255, 100, 0)
}

def visualize_predicted_pairs(
    model: ViREDModel,
    image: np.ndarray,
    object_labels: List[Dict],
    output_dir: str | Path,
    model_input_size: int = 384,
    slice_size: int   = 640,
    overlap: float = 0.25,
    threshold: float = 0.25,
    batch_size: int = 16,
    device: str = 'cpu'
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pair_indices = inference_pairs(
        model=model,
        image=image,
        object_labels=object_labels,
        model_input_size=model_input_size,
        slice_size=slice_size,
        overlap=overlap,
        threshold=threshold,
        batch_size=batch_size,
        device=device
    )
    image_background = image.copy()
    def _center(label: dict) -> Tuple[int, int]:
        return (label['x1'] + label['x2']) // 2, (label['y1'] + label['y2']) // 2

    for label_idx, label in enumerate(object_labels):
        class_id, x1, y1, x2, y2 = label['class_id'], label['x1'], label['y1'], label['x2'], label['y2']
        object_type = yolo_class_id_to_object_type(class_id)
        object_type_name = OBJECT_TYPE_NAMES[object_type]
        color = COLORS[object_type]
        cv2.rectangle(image_background, (x1, y1), (x2, y2), color, 2)

        text = f"({label_idx}){object_type_name}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y1 - 5, text_size[1])
        
        cv2.rectangle(
            image_background,
            (x1, text_y - text_size[1] - 5),
            (x1 + text_size[0], text_y + 5),
            color,
            -1
        )
        cv2.putText(
            image_background,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    for label_idx1, label_idx2 in pair_indices:
        label1 = object_labels[label_idx1]
        label2 = object_labels[label_idx2]
        center1 = _center(label1)
        center2 = _center(label2)
        cv2.line(image_background, center1, center2, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(image_background, center1, 5, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(image_background, center2, 5, (0, 165, 255), -1, cv2.LINE_AA)

    cv2.imwrite(Path(output_dir) / f"prediction_pairs.jpg", image_background) 
    logger.info(f"image saved to {Path(output_dir) / f'prediction_pairs.jpg'}")


def _parse_labels(
    label_path: Path,
    img_w: float,
    img_h: float,
) -> List[Dict]:

    if not label_path.exists():
        raise Exception(f"label_path does not exist {label_path}")

    labels = []

    with open(label_path, "r") as fh:
        for line_idx, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) != 5:
                raise Exception(f"{label_path}:{line_idx}: expected 5 fields, got {len(parts)}")

            class_id   = int(parts[0])
            x_c   = float(parts[1])
            y_c   = float(parts[2])
            norm_w = float(parts[3])
            norm_h = float(parts[4])

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


            labels.append({
                "class_id": class_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    return labels

if __name__ == "__main__":
    visualization_config = {
        "images_dir": "data/sample-1/images",
        "labels_dir": "data/sample-1/labels",
        "output_dir": "data/sample-1/output_visualizations",
        "device": "cpu", 
        "model_path": "checkpoints/best_val_0.1_train_0.1.pth"
    }
    model_config = ViredConfig(
        vision_backbone="vit_small_patch16_384",
        image_size=384,
        num_object_types=3,
        roi_context_pad=32,
    )
    device = visualization_config["device"]
    model = ViREDModel(model_config).to(device).eval()
    model.load_state_dict(torch.load(visualization_config["model_path"], map_location=device), strict=True)
    images_dir = Path(visualization_config['images_dir'])
    labels_dir = Path(visualization_config['labels_dir'])
    output_dir = Path(visualization_config['output_dir'])

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_files:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to read image {image_path}")


        H_orig, W_orig = image.shape[:2]
        name = image_path.stem
        image_output_dir = output_dir / name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / f"{name}.txt"
        labels = _parse_labels(label_path, img_w=W_orig, img_h=H_orig)

        visualize_predicted_pairs(
            model=model,
            image=image,
            object_labels=labels,
            output_dir=image_output_dir,
        )




