"""GoldX: ground-truth explanations for visual classifiers."""

from . import attacking, explaining, imagenet, masking, metrics, pipeline
from .attacking import NormalizedModel, attack_image_with_mask
from .explaining import explain
from .masking import generate_mask, mask_matrix
from .metrics import intersection_over_union

__all__ = [
    "NormalizedModel",
    "attack_image_with_mask",
    "attacking",
    "explain",
    "explaining",
    "generate_mask",
    "imagenet",
    "intersection_over_union",
    "mask_matrix",
    "masking",
    "metrics",
    "pipeline",
]
