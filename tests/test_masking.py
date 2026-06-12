import numpy as np
import pytest

from goldx.masking import generate_mask, mask_matrix, select_k_largest


def test_generate_mask_size_and_mode():
    mask = generate_mask(64, 48)
    assert mask.size == (64, 48)
    assert mask.mode == "1"


def test_generate_mask_area_within_bounds():
    min_area, max_area = 0.1, 0.3
    for _ in range(20):
        mask = generate_mask(100, 100, min_area=min_area, max_area=max_area)
        area = np.count_nonzero(np.array(mask)) / (100 * 100)
        # Discretization makes the ellipse slightly smaller than the ideal circle.
        assert 0.5 * min_area <= area <= 1.1 * max_area


def test_generate_mask_rejects_oversized_area():
    with pytest.raises(ValueError):
        generate_mask(10, 10, min_area=0.9, max_area=0.9)


def test_select_k_largest():
    matrix = np.array([[1, 5], [3, 4]])
    rows, cols = select_k_largest(matrix, 2)
    values = sorted(matrix[r, c] for r, c in zip(rows, cols, strict=True))
    assert values == [4, 5]


def test_mask_matrix_top_k():
    matrix = np.array([[0.1, 0.9], [0.5, 0.3]])
    mask = mask_matrix(matrix, k=2)
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) == 2
    assert mask[0, 1] == 255
    assert mask[1, 0] == 255


def test_mask_matrix_validates_input():
    with pytest.raises(ValueError):
        mask_matrix(np.zeros((2, 2, 2)), k=1)
    with pytest.raises(ValueError):
        mask_matrix(np.zeros((2, 2)), k=0)
    with pytest.raises(ValueError):
        mask_matrix(np.zeros((2, 2)), k=5)
