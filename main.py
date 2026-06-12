"""Run the full GoldX pipeline: attack, explain, score, report."""

import argparse
import logging
from pathlib import Path

import torch
import torchvision

import goldx

MODELS = {
    "resnet18": (
        torchvision.models.resnet18,
        torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
    ),
    "resnet50": (
        torchvision.models.resnet50,
        torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
    ),
    "vgg16": (
        torchvision.models.vgg16,
        torchvision.models.VGG16_Weights.IMAGENET1K_V1,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="resnet18")
    parser.add_argument("--source", type=Path, default=Path("data/source"))
    parser.add_argument("--gold", type=Path, default=Path("data/gold"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--min-mask-area", type=float, default=0.1)
    parser.add_argument("--max-mask-area", type=float, default=0.1)
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constructor, weights = MODELS[arguments.model]
    model = goldx.NormalizedModel(constructor(weights=weights)).to(device).eval()

    goldx.pipeline.prepare_ground_truths(
        source_directory=arguments.source,
        target_directory=arguments.gold,
        image_size=arguments.image_size,
        model=model,
        min_mask_area=arguments.min_mask_area,
        max_mask_area=arguments.max_mask_area,
    )

    records = goldx.pipeline.compute_explanations(
        model=model, gold_directory=arguments.gold
    )

    goldx.reporting.render_report(arguments.gold, records)


if __name__ == "__main__":
    main()
