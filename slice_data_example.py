import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
import logging
import shutil
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bbox_intersects_slice(bbox_norm: List[float],  slice_x: int, slice_y: int, slice_size: int, img_w: int, img_h: int) -> bool:
    """
    Check if a normalized bounding box intersects with a slice.
    bbox_norm format: [x_center, y_center, width, height] (normalized 0-1)
    """
    x_center, y_center, norm_w, norm_h = bbox_norm
    
    x_abs = x_center * img_w
    y_abs = y_center * img_h
    w_abs = norm_w * img_w
    h_abs = norm_h * img_h
    
    bbox_x1 = x_abs - w_abs / 2
    bbox_y1 = y_abs - h_abs / 2
    bbox_x2 = x_abs + w_abs / 2
    bbox_y2 = y_abs + h_abs / 2
    
    slice_x1, slice_y1 = slice_x, slice_y
    slice_x2, slice_y2 = slice_x + slice_size, slice_y + slice_size
    
    return not (bbox_x2 < slice_x1 or bbox_x1 > slice_x2 or bbox_y2 < slice_y1 or bbox_y1 > slice_y2)

# the check with respect to ratio is from https://github.com/bobyard-real/sahi/blob/out-of-box2/sahi/slicing.py, process_coco_annotations 
# currently I want to go with a simpler approach that checks the area only. 
def adjust_cut_bbox_in_slice(coords_cut, coords_original, slice_size, fit_box_to_slice: bool):

    if fit_box_to_slice:
        return coords_cut

    x1_cut, y1_cut, x2_cut, y2_cut = coords_cut
    x1_original, y1_original, x2_original, y2_original = coords_original

    w_cut, h_cut = x2_cut - x1_cut, y2_cut - y1_cut
    w_original, h_original = x2_original - x1_original, y2_original - y1_original

    area_cut = w_cut * h_cut
    area_original = w_original * h_original

    if area_cut / area_original  > 0.5:
        return coords_original

    # ratio = max(w_original, h_original) / min(w_original, h_original)

    # if area_cut / area_original == 1.:
    #     return coords_original
    # # Assumption 1: Model cannot ignore something bigger than 10% of regular symbol (10% of circle)
    # elif 1 <= ratio < 1.05:
    #     # i -- ignore, s -- slice, f -- full, () -- dominates
    #     # assume 0 overlap 10 percent should be enough to produce two full boxes
    #     # 0.3x0.3=10% i | 0.7x0.3=20% f
    #     # -------------------------------
    #     # 0.7*0.3=20% f | 0.7*0.7=50% (f)
    #     if area_cut / area_original > 0.1:  # (A1)
    #         return coords_original
    #     # else ignore
    # elif 1.05 <= ratio < 2.3:
    #     # 0.4x0.2=10% i | 1x0.2=20% s
    #     # -----------------------------
    #     # 0.4*0.5=20% s | 1*0.5=50% (f)
    #     #
    #     # 0.6x0.3=15% s | 0.9*0.3=25% s
    #     # -------------------------------
    #     # 0.6x0.4=25% s | 0.9*0.4=35% (f)
    #     #
    #     # ------------------
    #     # |      |         |
    #     # |      |         |
    #     # |      |         |
    #     # ------------------
    #     # |      |         |
    #     # |      |    *    |
    #     # |      |    *    |
    #     # |      |         |
    #     # ------------------
    #     if area_cut / area_original > 0.35:
    #         return coords_original
    #     elif area_cut / area_original > 0.1:   # (A1)
    #         return coords_cut
    #     # else ignore
    # else:
    #     # We cannot guess full annotation in any way, but we can filter nonsense sliced.
    #     # To not lay fully at least in one slice, it should have d > overlap.
    #     # Intersections with a slice end:
    #     # 11112
    #     # +-+-+
    #     #   |||
    #     # -+-++
    #     # It prevents boxes which lay fully in one slice and partially in second.
    #     overlap = 0.2

    #     # at least some part is unique to this slice
    #     if w_cut / slice_size > overlap or h_cut / slice_size > overlap:
    #         return coords_cut


def adjust_bbox_for_slice(bbox_norm: List[float], slice_x: int, slice_y: int, slice_size: int, img_w: int, img_h: int, fit_box_to_slice: bool) -> List[float] | None:
    """
    Adjust normalized bbox coordinates relative to slice.
    Adjust keypoints relative to a slice, if keypoints exist. Accept exactly two keypoints. 
    Returns (adjusted_bbox, is_valid) where adjusted_bbox is [x_center, y_center, width, height] normalized to slice.
    """
    x_center, y_center, norm_w, norm_h = bbox_norm
    
    x_abs = x_center * img_w
    y_abs = y_center * img_h
    w_abs = norm_w * img_w
    h_abs = norm_h * img_h

    
    bbox_x1 = x_abs - w_abs / 2
    bbox_y1 = y_abs - h_abs / 2
    bbox_x2 = x_abs + w_abs / 2
    bbox_y2 = y_abs + h_abs / 2

    x1_original = bbox_x1 - slice_x
    y1_original = bbox_y1 - slice_y
    x2_original = bbox_x2 - slice_x
    y2_original = bbox_y2 - slice_y

    x1_cut = max(0, x1_original)
    y1_cut = max(0, y1_original)
    x2_cut = min(slice_size, x2_original)
    y2_cut = min(slice_size, y2_original)


    if x2_cut <= x1_cut or y2_cut <= y1_cut:
        return None

    coords_rel = adjust_cut_bbox_in_slice((x1_cut, y1_cut, x2_cut, y2_cut), (x1_original, y1_original, x2_original, y2_original), slice_size, fit_box_to_slice=fit_box_to_slice)
    if not coords_rel:
        return None
    x1_rel, y1_rel, x2_rel, y2_rel = coords_rel
    
    new_w = x2_rel - x1_rel
    new_h = y2_rel - y1_rel
    new_x_center = (x1_rel + x2_rel) / 2
    new_y_center = (y1_rel + y2_rel) / 2
    
    new_x_center_norm = new_x_center / slice_size
    new_y_center_norm = new_y_center / slice_size
    new_w_norm = new_w / slice_size
    new_h_norm = new_h / slice_size
    
    
    if new_w_norm > 0 and new_h_norm > 0:
        new_label = [new_x_center_norm, new_y_center_norm, new_w_norm, new_h_norm]
        return new_label
    
    return None


def slice_image_and_labels(
    image_path: Path,
    label_path: Path,
    slice_size: int,
    output_dir: Path,
    split_name: str,
    overlap: float,
    fit_box_to_slice: bool
) -> int:
    """
    Slice an image and adjust YOLO format labels. 

    Returns number of slices created.
    """
    assert 0 <= overlap < 1
        
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not load image {image_path}")
        return 0
    
    h, w = img.shape[:2]
    image_name = image_path.stem
    
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) != 5:
                        print("invalid labels", image_path.stem)
                        continue
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])

    step = int(slice_size * (1 - overlap))
    y_positions = list(range(0, h, step))
    x_positions = list(range(0, w, step))
    
    if y_positions[-1] + slice_size < h:
        y_positions.append(h - slice_size)
    if x_positions[-1] + slice_size < w:
        x_positions.append(w - slice_size)
    
    slice_count = 0
    
    for y_pos in y_positions:
        for x_pos in x_positions:
            y_end = min(y_pos + slice_size, h)
            x_end = min(x_pos + slice_size, w)
            
            actual_h = y_end - y_pos
            actual_w = x_end - x_pos
            
            slice_img = img[y_pos:y_end, x_pos:x_end]
            
            if actual_h < slice_size or actual_w < slice_size:
                padded_slice = np.zeros((slice_size, slice_size, 3), dtype=slice_img.dtype)
                padded_slice[:actual_h, :actual_w] = slice_img
                slice_img = padded_slice
            
            slice_filename = f"{image_name}_slice_{x_pos}_{y_pos}.jpg"
            slice_path = Path(output_dir) / split_name / "images" / slice_filename
            cv2.imwrite(str(slice_path), slice_img)
            
            label_filename = f"{image_name}_slice_{x_pos}_{y_pos}.txt"
            label_path_out = Path(output_dir) / split_name / "labels" / label_filename
            
            with open(label_path_out, 'w') as f:
                for i, label in enumerate(labels):
                    class_id, x_center, y_center, width, height = label
                    keypoints = None
                    bbox_norm = [x_center, y_center, width, height]
                    if bbox_intersects_slice(bbox_norm, x_pos, y_pos, slice_size, w, h):
                        adjusted_bbox = adjust_bbox_for_slice(bbox_norm, x_pos, y_pos, slice_size, w, h, fit_box_to_slice=fit_box_to_slice)
                        if adjusted_bbox:
                            f.write(f"{class_id} {adjusted_bbox[0]:.6f} {adjusted_bbox[1]:.6f} {adjusted_bbox[2]:.6f} {adjusted_bbox[3]:.6f}\n") # type: ignore
            
            slice_count += 1
    
    return slice_count



def process_split(
    split_dir: Path,
    split_name: str,
    slice_size: int,
    output_dir: Path,
    overlap: float = 0,
    fit_box_to_slice: bool = False
) -> None:
    """
    Process a single split (train/valid/test)
    
    fit_box_to_slice: if true, when a slice contains part of a bounding box, the coordinates will be adjusted to fit into the slice. 
                      If False, we may add the full box. The decision is made in adjust_cut_bbox_in_slice. 
    
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}, skipping {split_name}")
        return
    
    output_images_dir = Path(output_dir) / split_name / "images"
    output_labels_dir = Path(output_dir) / split_name / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(images_dir.glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    total_slices = 0
    for image_path in image_files:
        label_path = labels_dir / (image_path.stem + ".txt")
        
        logger.info(f"Processing {image_path.name}...")
        slices = slice_image_and_labels(
            image_path,
            label_path,
            slice_size,
            output_dir,
            split_name,
            overlap=overlap,
            fit_box_to_slice=fit_box_to_slice
        )
        total_slices += slices
    
    logger.info(f"{split_name.upper()} Split:")
    logger.info(f"  Original images: {len(image_files)}")
    logger.info(f"  Sliced images: {total_slices}")
    logger.info(f"  Saved to: {Path(output_dir) / split_name}")


def slice_data(detection_data_config):
    
    
    if not Path(detection_data_config["dataset_dir"]).exists():
        raise ValueError(f"Dataset directory not found: {detection_data_config['dataset_dir']}")
    
    logger.info("=" * 60)
    logger.info("YOLO Dataset Slicing")
    logger.info("=" * 60)
    logger.info(f"Input dataset: {detection_data_config['dataset_dir']}")
    logger.info(f"Output dataset: {detection_data_config['output_dir']}")
    logger.info(f"Slice size: {detection_data_config['image_size']}x{detection_data_config['image_size']}")
    logger.info("=" * 60)
    
    for split_name in ["train", "valid", "test"]:
        split_dir = Path(detection_data_config["dataset_dir"]) / split_name
        if split_dir.exists():
            process_split(split_dir, split_name, detection_data_config["image_size"], detection_data_config["output_dir"], overlap=detection_data_config["overlap"])
    
    # Copy data.yaml from input to output directory
    input_yaml = Path(detection_data_config["dataset_dir"]) / 'data.yaml'
    output_yaml = Path(detection_data_config["output_dir"]) / 'data.yaml'
    if input_yaml.exists():
        shutil.copy(str(input_yaml), str(output_yaml))
        print(f'Copied data.yaml from {input_yaml} to {output_yaml}')


    logger.info("=" * 60)
    logger.info("Dataset slicing complete!")
    logger.info(f"Output saved to: {detection_data_config['output_dir']}")
    logger.info("=" * 60)



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    detection_data_config = {
        "dataset_dir": BASE_DIR / "data/detection/test_images-2",
        "output_dir": BASE_DIR / "data/detection/test_images-2_sliced",
        "image_size": 1280,
        "overlap": 0.25,
    }

    slice_data(detection_data_config)

    
