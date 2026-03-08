"""
Lightweight shape-assertion helpers.

These are used throughout the codebase to catch shape mismatches early with
descriptive error messages.  They have zero computational cost when assertions
are disabled (python -O), but that is not the primary intended use; rather they
serve as living documentation of expected tensor layouts.
"""

from __future__ import annotations

import torch
from typing import Optional, Sequence, Tuple


def assert_shape(
    tensor: torch.Tensor,
    expected: Sequence[Optional[int]],
    name: str = "tensor",
) -> None:
    """Assert that *tensor* has the given shape.

    Use ``None`` as a wildcard for a dimension that can be any size.

    Example::

        assert_shape(x, (B, None, 256), "image_tokens")
    """
    if len(tensor.shape) != len(expected):
        raise AssertionError(
            f"{name}: expected {len(expected)}D tensor, got shape {tuple(tensor.shape)}"
        )
    for i, (actual, exp) in enumerate(zip(tensor.shape, expected)):
        if exp is not None and actual != exp:
            raise AssertionError(
                f"{name}: dim {i} expected {exp}, got {actual}  "
                f"(full shape: {tuple(tensor.shape)}, expected: {tuple(expected)})"
            )


def assert_dtype(
    tensor: torch.Tensor,
    dtype: torch.dtype,
    name: str = "tensor",
) -> None:
    if tensor.dtype != dtype:
        raise AssertionError(
            f"{name}: expected dtype {dtype}, got {tensor.dtype}"
        )


def shape_str(tensor: torch.Tensor) -> str:
    """Return a concise shape string, e.g. '(2, 10, 256)'."""
    return str(tuple(tensor.shape))
