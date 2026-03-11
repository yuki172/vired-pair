import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

def boxes_to_masks(object_boxes: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    B, C, H, W = image.shape
    _, N, _ = object_boxes.shape
    device = image.device

    x1, y1, x2, y2 = object_boxes.unbind(-1)

    ys = torch.arange(H, device=device).view(1, 1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, 1, W)

    x1 = x1.view(B, N, 1, 1)
    y1 = y1.view(B, N, 1, 1)
    x2 = x2.view(B, N, 1, 1)
    y2 = y2.view(B, N, 1, 1)

    mask_x = (xs >= x1) & (xs < x2)
    mask_y = (ys >= y1) & (ys < y2)

    masks = mask_x & mask_y
    return masks.float()


CLASS_MAP = {
        1: 0,
        3: 0,
        0: 1,
        2: 1,
        5: 1,
        4: 2,
}

class ViREDDataset(Dataset):


    def __init__(self, data_path: str | Path, class_map: Dict[int, int] = CLASS_MAP):
        self.data_path = Path(data_path)
        self.class_map = class_map

        self.images_dir = self.data_path / "images"
        self.object_labels_dir = self.data_path / "object_labels"
        self.pair_labels_dir = self.data_path / "pair_labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(self.images_dir)
        if not self.object_labels_dir.exists():
            raise FileNotFoundError(self.object_labels_dir)
        if not self.pair_labels_dir.exists():
            raise FileNotFoundError(self.pair_labels_dir)

        image_names = {p.stem for p in self.images_dir.glob("*.png")}
        object_names = {p.stem for p in self.object_labels_dir.glob("*.txt")}
        pair_names = {p.stem for p in self.pair_labels_dir.glob("*.txt")}

        if image_names != object_names or image_names != pair_names:
            raise ValueError("Mismatch between image, object_labels and pair_labels files.")

        self.image_names = sorted(image_names)

        # Validate dataset format
        for name in self.image_names:

            img_path = self.images_dir / f"{name}.png"
            obj_path = self.object_labels_dir / f"{name}.txt"
            pair_path = self.pair_labels_dir / f"{name}.txt"

            width, height = self._get_image_size(img_path)

            objects = self._parse_object_labels(obj_path, name, width, height)
            self._parse_pair_labels(pair_path, name, len(objects))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        name = self.image_names[idx]

        img_path = self.images_dir / f"{name}.png"
        obj_path = self.object_labels_dir / f"{name}.txt"
        pair_path = self.pair_labels_dir / f"{name}.txt"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to read image {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        objects = self._parse_object_labels(obj_path, name, width, height)
        pairs = self._parse_pair_labels(pair_path, name, len(objects))

        boxes = []
        types = []

        for class_id, nx, ny, nw, nh in objects:

            x1 = (nx - nw / 2) * width
            y1 = (ny - nh / 2) * height
            x2 = (nx + nw / 2) * width
            y2 = (ny + nh / 2) * height

            boxes.append([x1, y1, x2, y2])
            types.append(self.class_map[class_id])

        object_boxes = torch.tensor(boxes, dtype=torch.float32)
        object_types = torch.tensor(types, dtype=torch.long)

        object_masks = boxes_to_masks(
            object_boxes.unsqueeze(0),
            image.unsqueeze(0)
        ).squeeze(0)

        pair_labels = torch.tensor(pairs, dtype=torch.long)

        return {
            "image": image,
            "object_masks": object_masks,
            "object_boxes": object_boxes,
            "object_types": object_types,
            "pair_labels": pair_labels,
        }

    def _get_image_size(self, path: Path) -> Tuple[int, int]:

        img = cv2.imread(str(path))

        if img is None:
            raise ValueError(f"Cannot read image {path}")

        h, w = img.shape[:2]

        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image size {path}")

        return w, h

    def _parse_object_labels(
        self,
        path: Path,
        image_name: str,
        width: int,
        height: int,
    ) -> List[Tuple[int, float, float, float, float]]:

        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]

        if len(lines) == 0:
            raise ValueError(f"{path} empty")

        parsed = []

        for i, line in enumerate(lines):

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(f"{path} line {i+1} invalid")

            class_id = int(parts[0])

            if class_id not in self.class_map:
                raise ValueError(f"{path} invalid class id {class_id}")

            nx, ny, nw, nh = map(float, parts[1:])

            if not (0 <= nx <= 1 and 0 <= ny <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
                raise ValueError(f"{path} line {i+1} invalid normalized bbox")

            x1 = (nx - nw / 2) * width
            x2 = (nx + nw / 2) * width
            y1 = (ny - nh / 2) * height
            y2 = (ny + nh / 2) * height

            if not (0 <= x1 < x2 <= width):
                raise ValueError(f"{path} line {i+1} bbox outside image")

            if not (0 <= y1 < y2 <= height):
                raise ValueError(f"{path} line {i+1} bbox outside image")

            parsed.append((class_id, nx, ny, nw, nh))

        return parsed

    def _parse_pair_labels(
        self,
        path: Path,
        image_name: str,
        num_objects: int
    ) -> List[Tuple[int, int]]:

        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]

        pairs = []
        seen = set()

        for i, line in enumerate(lines):

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(f"{path} line {i+1} invalid")

            a = int(parts[0])
            b = int(parts[1])

            if not (0 <= a < num_objects):
                raise ValueError(f"{path} invalid index {a}")

            if not (0 <= b < num_objects):
                raise ValueError(f"{path} invalid index {b}")

            if a == b:
                raise ValueError(f"{path} self pair")

            key = tuple(sorted((a, b)))

            if key in seen:
                raise ValueError(f"{path} duplicate pair")

            seen.add(key)
            pairs.append((a, b))

        return pairs

import torch
from typing import Dict, List


def vired_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    """
    Collate function for ViREDDataset.

    Input: list of samples, each containing:
        image:         (C, H, W)
        object_masks:  (N_i, H, W)
        object_boxes:  (N_i, 4)
        object_types:  (N_i,)
        pair_labels:   (P_i, 2)

    Returns:
        images:                    (B, C, H, W)
        object_masks:              (B, N_max, H, W)
        object_boxes:              (B, N_max, 4)
        object_types:              (B, N_max)
        object_key_padding_mask:   (B, N_max), bool
                                   True = ignore / padding
                                   False = valid object
        pair_labels:               list of length B, each tensor (P_i, 2)
        pair_labels_mask:          list of length B, each tensor (P_i,)
                                   optional convenience mask, all True
        num_objects:               (B,)
    """
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

    pair_labels = []
    pair_labels_mask = []

    for i, sample in enumerate(batch):
        n_i = sample["object_masks"].shape[0]

        object_masks[i, :n_i] = sample["object_masks"]
        object_boxes[i, :n_i] = sample["object_boxes"]
        object_types[i, :n_i] = sample["object_types"]

        # real objects => False
        object_key_padding_mask[i, :n_i] = False

        pair_i = sample["pair_labels"]
        if pair_i.ndim != 2 or pair_i.shape[-1] != 2:
            raise ValueError(
                f"pair_labels for sample {i} must have shape (P, 2), got {tuple(pair_i.shape)}"
            )

        pair_labels.append(pair_i)
        pair_labels_mask.append(torch.ones(pair_i.shape[0], dtype=torch.bool, device=pair_i.device))

    num_objects = torch.tensor(num_objects_list, dtype=torch.long, device=device)

    return {
        "images": images,
        "object_masks": object_masks,
        "object_boxes": object_boxes,
        "object_types": object_types,
        "object_key_padding_mask": object_key_padding_mask,
        "pair_labels": pair_labels,
        "pair_labels_mask": pair_labels_mask,
        "num_objects": num_objects,
    }


if __name__ == "__main__":
    config = {
        "data_path": "data/train"
    }

    dataset = ViREDDataset(data_path="path/to/data")

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

