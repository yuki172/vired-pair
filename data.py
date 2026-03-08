"""
data.py
=======
Dataset slicing utility for the electrical-plan relation detection task.

──────────────────────────────────────────────────────────────────────────────
Directory layout
──────────────────────────────────────────────────────────────────────────────

Input  (e.g. data/pair-1):
    {split}/
        images/           ← original image files
        labels/           ← YOLO format: "class_id x_c y_c w h" per object
        pair_labels/      ← natural pairs:  "i j" per line  (0-based obj indices)

Output (e.g. data/pair-1_sliced):
    {split}/
        images/           ← one tile per slice
        labels/           ← adjusted YOLO labels (clipped to slice, re-normalised)
        pair_labels/      ← pair labels remapped to slice-local object indices
        object_maps/      ← correspondence:  "slice_idx  original_idx"  per line

──────────────────────────────────────────────────────────────────────────────
Object map files
──────────────────────────────────────────────────────────────────────────────
Each slice writes one object_maps/{slice_name}.txt with one line per kept object:

    slice_idx  original_idx

slice_idx     – 0-based index into *this slice's* label file
original_idx  – 0-based index into the *original full-image* label file

This mapping is needed at inference time to aggregate pair predictions from
overlapping tiles back into full-image object-index space.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def slice_dataset(
    input_dir:  str | Path,
    slice_size: int   = 1280,
    overlap:    float = 0.2,
    splits:     Optional[List[str]] = None,
    labels_subdir:      str = "labels",
    pair_labels_subdir: str = "pair_labels",
) -> Path:
    """Slice every image in a relation-detection dataset into fixed-size tiles.

    The output directory is derived automatically:  if ``input_dir`` is
    ``data/pair-1`` the output is written to ``data/pair-1_sliced``.

    Args:
        input_dir:          Root of the input dataset (contains split sub-dirs).
        slice_size:         Side length of each square tile in pixels.
        overlap:            Fractional overlap between adjacent tiles (0–1).
        splits:             List of split names to process.  Defaults to
                            ``["train", "valid", "test"]``.
        labels_subdir:      Sub-directory that holds YOLO object label files.
        pair_labels_subdir: Sub-directory that holds pair label files.

    Returns:
        Path to the output (sliced) dataset root.
    """
    assert 0.0 <= overlap < 1.0, "overlap must be in [0, 1)"
    input_dir  = Path(input_dir)
    output_dir = input_dir.parent / (input_dir.name + "_sliced")
    splits     = splits or ["train", "valid", "test"]

    logger.info("=" * 60)
    logger.info("Dataset slicing")
    logger.info(f"  Input  : {input_dir}")
    logger.info(f"  Output : {output_dir}")
    logger.info(f"  Tile   : {slice_size}×{slice_size}  overlap={overlap}")
    logger.info("=" * 60)

    for split in splits:
        split_dir = input_dir / split
        if not split_dir.exists():
            logger.info(f"  [{split}] not found – skipping")
            continue
        _process_split(
            split_dir=split_dir,
            split_name=split,
            output_root=output_dir,
            slice_size=slice_size,
            overlap=overlap,
            labels_subdir=labels_subdir,
            pair_labels_subdir=pair_labels_subdir,
        )

    # Copy data.yaml if present.
    for yaml_name in ("data.yaml", "dataset.yaml"):
        src = input_dir / yaml_name
        if src.exists():
            shutil.copy(src, output_dir / yaml_name)
            logger.info(f"  Copied {yaml_name}")

    logger.info("Done.  Output: %s", output_dir)
    return output_dir


def read_object_map(path: Path) -> Dict[int, int]:
    """Read an object map file produced by ``slice_dataset``.

    Returns:
        Mapping  slice-local object index → original full-image object index.
    """
    mapping: Dict[int, int] = {}
    if not path.exists():
        return mapping
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                mapping[int(parts[0])] = int(parts[1])
    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _process_split(
    split_dir:          Path,
    split_name:         str,
    output_root:        Path,
    slice_size:         int,
    overlap:            float,
    labels_subdir:      str,
    pair_labels_subdir: str,
) -> None:
    images_dir      = split_dir / "images"
    labels_dir      = split_dir / labels_subdir
    pair_labels_dir = split_dir / pair_labels_subdir

    out_images_dir = output_root / split_name / "images"
    out_labels_dir = output_root / split_name / labels_subdir
    out_pairs_dir  = output_root / split_name / pair_labels_subdir
    out_maps_dir   = output_root / split_name / "object_maps"

    for d in (out_images_dir, out_labels_dir, out_pairs_dir, out_maps_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    total_slices = 0
    for img_path in image_files:
        name = img_path.stem
        n = _slice_image(
            image_path      = img_path,
            label_path      = labels_dir      / f"{name}.txt",
            pair_label_path = pair_labels_dir / f"{name}.txt",
            out_images_dir  = out_images_dir,
            out_labels_dir  = out_labels_dir,
            out_pairs_dir   = out_pairs_dir,
            out_maps_dir    = out_maps_dir,
            slice_size      = slice_size,
            overlap         = overlap,
        )
        total_slices += n
        logger.debug(f"  {name}: {n} slices")

    logger.info(f"  [{split_name}] {len(image_files)} images → {total_slices} slices")


def _slice_image(
    image_path:      Path,
    label_path:      Path,
    pair_label_path: Path,
    out_images_dir:  Path,
    out_labels_dir:  Path,
    out_pairs_dir:   Path,
    out_maps_dir:    Path,
    slice_size:      int,
    overlap:         float,
) -> int:
    """Slice one image and write all four output files per tile.

    Returns the number of tiles written.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not load image: {image_path}")
        return 0

    H, W = img.shape[:2]
    name = image_path.stem
    ext  = image_path.suffix.lower() or ".jpg"

    objects  = _read_yolo_labels(label_path)       # list[(class_id, xc, yc, w, h)] normalised
    gt_pairs = _read_pair_labels(pair_label_path)  # list[(i, j)]

    # Convert YOLO normalised → full-image pixel coords for clipping arithmetic.
    objects_px: List[Tuple[int, float, float, float, float]] = []
    for class_id, xc, yc, bw, bh in objects:
        x1 = (xc - bw / 2) * W
        y1 = (yc - bh / 2) * H
        x2 = (xc + bw / 2) * W
        y2 = (yc + bh / 2) * H
        objects_px.append((class_id, x1, y1, x2, y2))

    step      = max(1, int(slice_size * (1 - overlap)))
    positions = _compute_slice_positions(W, H, slice_size, step)
    count     = 0

    for x_off, y_off in positions:
        slice_name = f"{name}_slice_{x_off}_{y_off}"

        # ── Determine which objects fall into this tile ──────────────── #
        slice_labels: List[Tuple[int, float, float, float, float]] = []
        orig_to_slice: Dict[int, int] = {}   # original index → slice-local index

        for orig_idx, (class_id, x1, y1, x2, y2) in enumerate(objects_px):
            clipped = _clip_box_to_slice(x1, y1, x2, y2, x_off, y_off, slice_size)
            if clipped is None:
                continue  # no intersection with this tile

            cx1, cy1, cx2, cy2 = clipped
            # Re-normalise clipped box to slice-local coordinates.
            nxc = ((cx1 + cx2) / 2 - x_off) / slice_size
            nyc = ((cy1 + cy2) / 2 - y_off) / slice_size
            nw  = (cx2 - cx1) / slice_size
            nh  = (cy2 - cy1) / slice_size

            slice_idx = len(slice_labels)
            orig_to_slice[orig_idx] = slice_idx
            slice_labels.append((class_id, nxc, nyc, nw, nh))

        # ── Remap pair labels ─────────────────────────────────────────── #
        # A pair is kept only when BOTH objects survived into this tile.
        slice_pairs: List[Tuple[int, int]] = [
            (orig_to_slice[i], orig_to_slice[j])
            for i, j in gt_pairs
            if i in orig_to_slice and j in orig_to_slice
        ]

        # ── Write outputs ─────────────────────────────────────────────── #
        tile_img = _extract_slice(img, x_off, y_off, slice_size, W, H)
        cv2.imwrite(str(out_images_dir / f"{slice_name}{ext}"), tile_img)
        _write_yolo_labels(out_labels_dir / f"{slice_name}.txt", slice_labels)
        _write_pair_labels(out_pairs_dir  / f"{slice_name}.txt", slice_pairs)
        _write_object_map (out_maps_dir   / f"{slice_name}.txt", orig_to_slice)

        count += 1

    return count


def _compute_slice_positions(W: int, H: int, slice_size: int, step: int
                             ) -> List[Tuple[int, int]]:
    """Return (x_off, y_off) top-left corners for all tiles.

    The last tile in each row/column is shifted left/up to ensure full coverage
    even when the image dimensions are not divisible by the tile size.
    """
    xs = list(range(0, W, step))
    ys = list(range(0, H, step))

    if xs and xs[-1] + slice_size > W:
        last_x = max(0, W - slice_size)
        if last_x not in xs:
            xs.append(last_x)
    if ys and ys[-1] + slice_size > H:
        last_y = max(0, H - slice_size)
        if last_y not in ys:
            ys.append(last_y)

    return [(x, y) for y in ys for x in xs]


def _extract_slice(img: np.ndarray, x: int, y: int,
                   slice_size: int, W: int, H: int) -> np.ndarray:
    """Extract a tile from ``img``, zero-padding if the tile extends past the edge."""
    y_end = min(y + slice_size, H)
    x_end = min(x + slice_size, W)
    crop  = img[y:y_end, x:x_end]

    actual_h = y_end - y
    actual_w = x_end - x
    if actual_h < slice_size or actual_w < slice_size:
        padded = np.zeros((slice_size, slice_size, img.shape[2]), dtype=img.dtype)
        padded[:actual_h, :actual_w] = crop
        return padded
    return crop


def _clip_box_to_slice(
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


# ── File I/O ──────────────────────────────────────────────────────────────── #

def _read_yolo_labels(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read a YOLO label file.  Returns [] if the file does not exist."""
    if not path.exists():
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 5:
                rows.append((int(parts[0]),
                             float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4])))
    return rows


def _read_pair_labels(path: Path) -> List[Tuple[int, int]]:
    """Read a pair label file.  Returns [] if the file does not exist."""
    if not path.exists():
        return []
    pairs = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def _write_yolo_labels(
    path: Path,
    labels: List[Tuple[int, float, float, float, float]],
) -> None:
    with open(path, "w") as fh:
        for class_id, xc, yc, w, h in labels:
            fh.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def _write_pair_labels(path: Path, pairs: List[Tuple[int, int]]) -> None:
    with open(path, "w") as fh:
        for i, j in pairs:
            fh.write(f"{i} {j}\n")


def _write_object_map(path: Path, orig_to_slice: Dict[int, int]) -> None:
    """Write an object map file (slice_idx  original_idx, one line per kept object)."""
    with open(path, "w") as fh:
        for orig_idx, slice_idx in sorted(orig_to_slice.items(), key=lambda kv: kv[1]):
            fh.write(f"{slice_idx} {orig_idx}\n")
