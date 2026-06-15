"""Random ground-truth masks and top-k binarization of heatmaps."""

import math
import random

import numpy as np
from PIL import Image, ImageDraw


def generate_mask(
    width: int,
    height: int,
    min_area: float = 0.1,
    max_area: float = 0.3,
) -> Image.Image:
    """Draw a random filled ellipse covering between ``min_area`` and ``max_area``
    of the image, as a binary PIL image."""
    image_area = width * height
    min_radius = math.sqrt(image_area * min_area / math.pi)
    max_radius = math.sqrt(image_area * max_area / math.pi)

    if 2 * max_radius > min(width, height):
        raise ValueError(f"max_area={max_area} does not fit a circle into {width}x{height}")

    radius = random.randint(int(min_radius), int(max_radius))
    center_x = random.randint(radius, width - radius)
    center_y = random.randint(radius, height - radius)

    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=1,
    )

    return mask


def select_k_largest(matrix: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
    """Return the 2D indices of the k largest elements."""
    index = np.argsort(matrix.flatten())[-k:]
    return np.unravel_index(index, matrix.shape)


def mask_matrix(matrix: np.ndarray, k: int) -> np.ndarray:
    """Binarize a heatmap: the k largest entries become 255, the rest 0."""
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
    if not 0 < k <= matrix.size:
        raise ValueError(f"k={k} must be in range [1, {matrix.size}]")

    index = select_k_largest(matrix, k)
    mask = np.zeros(matrix.shape, dtype=np.uint8)
    mask[index] = 255

    return mask
