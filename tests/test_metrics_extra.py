import numpy as np
import pytest

from goldx.metrics import pixel_auc, relevance_mass


def test_relevance_mass_all_inside():
    heatmap = np.array([[1.0, 0.0], [0.0, 0.0]])
    mask = np.array([[255, 0], [0, 0]])
    assert relevance_mass(heatmap, mask) == 1.0


def test_relevance_mass_half():
    heatmap = np.array([[1.0, 1.0], [0.0, 0.0]])
    mask = np.array([[255, 0], [0, 0]])
    assert relevance_mass(heatmap, mask) == 0.5


def test_relevance_mass_empty_heatmap():
    assert relevance_mass(np.zeros((2, 2)), np.full((2, 2), 255)) == 0.0


def test_pixel_auc_perfect():
    heatmap = np.array([[0.9, 0.1], [0.1, 0.8]])
    mask = np.array([[255, 0], [0, 255]])
    assert pixel_auc(heatmap, mask) == 1.0


def test_pixel_auc_inverted():
    heatmap = np.array([[0.1, 0.9], [0.9, 0.1]])
    mask = np.array([[255, 0], [0, 255]])
    assert pixel_auc(heatmap, mask) == 0.0


def test_pixel_auc_constant_heatmap_is_chance():
    heatmap = np.ones((4, 4))
    mask = np.zeros((4, 4))
    mask[:2] = 255
    assert pixel_auc(heatmap, mask) == 0.5


def test_pixel_auc_degenerate_mask():
    assert pixel_auc(np.ones((2, 2)), np.zeros((2, 2))) == 0.5


def test_pixel_auc_random_near_chance():
    rng = np.random.default_rng(0)
    heatmap = rng.random((50, 50))
    mask = np.zeros((50, 50))
    mask[:, :25] = 255
    assert pixel_auc(heatmap, mask) == pytest.approx(0.5, abs=0.05)
