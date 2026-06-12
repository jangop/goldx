import numpy as np
import pytest

from goldx.metrics import intersection_over_union


def test_identical_masks():
    mask = np.array([[255, 0], [0, 255]])
    assert intersection_over_union(mask, mask) == 1.0


def test_disjoint_masks():
    a = np.array([[255, 0], [0, 0]])
    b = np.array([[0, 0], [0, 255]])
    assert intersection_over_union(a, b) == 0.0


def test_partial_overlap():
    a = np.array([[255, 255], [0, 0]])
    b = np.array([[255, 0], [0, 0]])
    assert intersection_over_union(a, b) == 0.5


def test_empty_masks():
    empty = np.zeros((2, 2))
    assert intersection_over_union(empty, empty) == 0.0


def test_shape_mismatch():
    with pytest.raises(ValueError):
        intersection_over_union(np.zeros((2, 2)), np.zeros((3, 3)))
