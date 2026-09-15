"""Reference construction shared by training and inference.

The supplied paper-checkpoint profile selects 16 screened sections, averages
them in floating point, and replaces every non-overlapping 16 x 16 block by
its mean. Screening is performed before running this module so restored
outputs and labels cannot influence reference selection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def block_average(image: np.ndarray, block_size: int) -> np.ndarray:
    """Replace each non-overlapping block by its float mean."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got {source.shape}")
    height, width = source.shape
    result = np.empty_like(source)
    for y0 in range(0, height, block_size):
        y1 = min(y0 + block_size, height)
        for x0 in range(0, width, block_size):
            x1 = min(x0 + block_size, width)
            result[y0:y1, x0:x1] = source[y0:y1, x0:x1].mean(dtype=np.float32)
    return result


def build_reference(
    images: Sequence[np.ndarray],
    block_size: int = 16,
    quantize: bool = True,
) -> np.ndarray:
    """Float-mean screened sections, then apply non-overlapping block means."""
    if not images:
        raise ValueError("at least one screened reference section is required")
    arrays = [np.asarray(image, dtype=np.float32) for image in images]
    shape = arrays[0].shape
    if any(image.shape != shape for image in arrays):
        raise ValueError("all reference sections must have the same shape")
    fused = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float32)
    fused = block_average(fused, block_size)
    if quantize:
        return np.clip(fused, 0.0, 255.0).astype(np.uint8)
    return fused
