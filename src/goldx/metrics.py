"""Scoring explanations against ground-truth masks."""

import numpy as np


def intersection_over_union(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks (any nonzero value counts as set)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def relevance_mass(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of total heatmap mass that falls inside the mask.

    Threshold-free, so it doesn't depend on a method's spatial resolution
    the way top-k binarization does.
    """
    if heatmap.shape != mask.shape:
        raise ValueError(f"shape mismatch: {heatmap.shape} vs {mask.shape}")
    total = heatmap.sum()
    if total <= 0:
        return 0.0
    return float(heatmap[mask != 0].sum() / total)


def pixel_auc(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """AUC-ROC of the heatmap as a per-pixel classifier of mask membership.

    0.5 is chance; threshold-free like ``relevance_mass``. Ties are handled
    by midranking.
    """
    if heatmap.shape != mask.shape:
        raise ValueError(f"shape mismatch: {heatmap.shape} vs {mask.shape}")
    positive = (mask != 0).flatten()
    n_positive = int(positive.sum())
    n_negative = positive.size - n_positive
    if n_positive == 0 or n_negative == 0:
        return 0.5

    order = np.argsort(heatmap.flatten(), kind="stable")
    values = heatmap.flatten()[order]
    ranks = np.empty(positive.size, dtype=np.float64)
    # Midrank tied values so identical scores don't depend on sort order.
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[j + 1] == values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1

    rank_sum = ranks[positive].sum()
    auc = (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return float(auc)
