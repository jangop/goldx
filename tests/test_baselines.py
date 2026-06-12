import numpy as np
import torch

from goldx.baselines import diff_oracle_heatmap, highpass_heatmap, random_heatmap
from goldx.metrics import pixel_auc


def test_random_heatmap_shape_and_range():
    heatmap = random_heatmap(8, 12, np.random.default_rng(0))
    assert heatmap.shape == (8, 12)
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


def test_highpass_flat_image_is_zero():
    flat = torch.full((3, 16, 16), 0.5)
    heatmap = highpass_heatmap(flat)
    # Padding introduces edge response; the interior must be zero.
    assert np.allclose(heatmap[1:-1, 1:-1], 0)


def test_highpass_localizes_noise():
    torch.manual_seed(0)
    image = torch.full((3, 32, 32), 0.5)
    image[:, 8:16, 8:16] += (torch.rand(3, 8, 8) - 0.5) * 0.1
    heatmap = highpass_heatmap(image)
    mask = np.zeros((32, 32))
    mask[8:16, 8:16] = 255
    assert pixel_auc(heatmap, mask) > 0.8


def test_diff_oracle_localizes_change():
    original = torch.zeros(3, 8, 8)
    attacked = original.clone()
    attacked[:, 2:4, 2:4] = 0.5
    heatmap = diff_oracle_heatmap(attacked, original)
    assert heatmap[2:4, 2:4].min() == 1.0
    assert heatmap[0, 0] == 0.0


def test_diff_oracle_identical_images():
    image = torch.rand(3, 8, 8)
    heatmap = diff_oracle_heatmap(image, image)
    assert np.allclose(heatmap, 0)
