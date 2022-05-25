import random
from operator import mul

import numpy as np
from PIL import Image, ImageDraw


def generate_mask(width, height, min_area=0.1, max_area=0.3):
    image_area = width * height
    min_size = int(image_area * min_area)
    max_size = int(image_area * max_area)

    pi = 3.14159265358979323846

    min_radius = (min_size / pi) ** 0.5
    max_radius = (max_size / pi) ** 0.5

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


def select_k_largest(matrix: np.ndarray, k: int):
    # Select the indices of the k largest elements (flattened)
    index = np.argsort(matrix.flatten())[-k:]
    # index = np.argpartition(matrix, kth=k, axis=None)[::-1][:k]

    # Turn flattened indices into 2D indices.
    unraveled_index = np.unravel_index(index, matrix.shape)

    return unraveled_index


def mask_matrix(matrix: np.ndarray, k: int):
    assert len(matrix.shape) == 2, f"Matrix must be 2D, got {matrix.shape}"
    assert (
        0 < k <= mul(*matrix.shape)
    ), f"k={k} must be in range [1, {mul(*matrix.shape)}] based on matrix shape {matrix.shape}"
    assert len(np.unique(matrix)) > 1, f"matrix must have at least 2 unique values"

    index = select_k_largest(matrix, k)
    mask = np.zeros_like(matrix)
    mask[index] = 255

    return mask
