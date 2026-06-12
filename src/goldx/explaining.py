"""Compute saliency-style explanations with Captum."""

from typing import Any

import numpy as np
import torch


def normalize_attributions(
    attributions: np.ndarray, *, percentile: float = 98.0
) -> np.ndarray:
    """Reduce channel-wise attributions to a single [0, 1] heatmap.

    Keeps positive evidence only, sums over channels, and scales by the given
    percentile so a few outlier pixels don't wash out the rest.
    """
    if attributions.ndim != 3:
        raise ValueError(f"expected (C, H, W) attributions, got {attributions.shape}")
    positive = np.clip(attributions, 0.0, None).sum(axis=0)
    scale = np.percentile(positive, percentile)
    if scale <= 0:
        return np.zeros_like(positive)
    return np.clip(positive / scale, 0.0, 1.0)


def explain(
    *,
    method: type,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: int,
    args: dict[str, Any],
) -> np.ndarray:
    """Attribute the prediction for ``targets`` to input pixels.

    Returns a (H, W) heatmap in [0, 1].
    """
    explainer = method(model)
    attributions = (
        explainer.attribute(inputs, target=targets, **args)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    return normalize_attributions(attributions)
