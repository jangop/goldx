"""Model-blind baseline "explainers" that calibrate the benchmark.

A real explanation method must beat all of these:

- ``random_heatmap`` — the floor; IoU scale is uninterpretable without it.
- ``highpass_heatmap`` — detects high-frequency perturbation noise with zero
  model insight. If a method doesn't beat this, the benchmark only measured
  noise detection.
- ``diff_oracle_heatmap`` — the ceiling; reads the clean reference image,
  which no honest method has access to.
"""

import numpy as np
import torch
from torch.nn import functional

LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
).reshape(1, 1, 3, 3)


def _normalize(heatmap: np.ndarray) -> np.ndarray:
    maximum = heatmap.max()
    if maximum <= 0:
        return np.zeros_like(heatmap)
    return heatmap / maximum


def random_heatmap(
    height: int, width: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Uniform random heatmap — the chance-level floor."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.random((height, width))


def highpass_heatmap(image: torch.Tensor) -> np.ndarray:
    """Laplacian magnitude of the (C, H, W) image, summed over channels.

    Model-blind noise detector: localizes high-frequency content only.
    """
    channels = image.shape[0]
    kernel = LAPLACIAN_KERNEL.expand(channels, 1, 3, 3).to(image.dtype)
    response = functional.conv2d(image.unsqueeze(0), kernel, padding=1, groups=channels)
    magnitude = response.squeeze(0).abs().sum(dim=0).cpu().numpy()
    return _normalize(magnitude)


def diff_oracle_heatmap(attacked: torch.Tensor, original: torch.Tensor) -> np.ndarray:
    """Absolute difference between attacked and clean image, summed over
    channels — privileged information, upper bound."""
    difference = (attacked - original).abs().sum(dim=0).cpu().numpy()
    return _normalize(difference)
