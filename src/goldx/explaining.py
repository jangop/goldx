"""Compute saliency-style explanations with Captum."""

from typing import Any

import numpy as np
import torch


class ContrastiveLogit(torch.nn.Module):
    """Score "why ``target`` rather than ``reference``" as a logit difference.

    GoldX's ground truth is causal for the prediction *flip*, not for the
    prediction itself — context outside the mask legitimately supports the
    target class. Attributing the (target - reference) logit asks the
    question the intervention actually answers. Attribute with ``target=0``.
    """

    def __init__(self, model: torch.nn.Module, target: int, reference: int) -> None:
        super().__init__()
        self.model = model
        self.target = target
        self.reference = reference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return (logits[:, self.target] - logits[:, self.reference]).unsqueeze(1)


def normalize_attributions(attributions: np.ndarray, *, percentile: float = 98.0) -> np.ndarray:
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
        explainer.attribute(inputs, target=targets, **args).squeeze(0).detach().cpu().numpy()
    )
    return normalize_attributions(attributions)
