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
