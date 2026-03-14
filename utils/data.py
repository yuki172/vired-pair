from typing import Optional, Tuple

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