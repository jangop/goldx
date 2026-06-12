"""GoldX: ground-truth explanations for visual classifiers."""

from . import (
    attacking,
    baselines,
    explaining,
    imagenet,
    masking,
    metrics,
    pipeline,
    reporting,
)
from .attacking import NormalizedModel, attack_image_with_mask
from .explaining import explain
from .masking import generate_mask, mask_matrix
from .metrics import intersection_over_union, pixel_auc, relevance_mass

__all__ = [
    "NormalizedModel",
    "attack_image_with_mask",
    "attacking",
    "baselines",
    "explain",
    "explaining",
    "generate_mask",
    "imagenet",
    "intersection_over_union",
    "mask_matrix",
    "masking",
    "metrics",
    "pipeline",
    "pixel_auc",
    "relevance_mass",
    "reporting",
]
