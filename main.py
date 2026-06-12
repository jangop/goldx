"""Run the full GoldX pipeline on images in data/source."""

import logging
from pathlib import Path

import torch
import torchvision

import goldx

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = torchvision.models.vgg16(
    weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
)
model = goldx.NormalizedModel(backbone).to(device).eval()

source_directory = Path("data/source")
gold_directory = Path("data/gold")

goldx.pipeline.prepare_ground_truths(
    source_directory=source_directory,
    target_directory=gold_directory,
    image_size=224,
    model=model,
)

goldx.pipeline.compute_explanations(
    model=model,
    gold_directory=gold_directory,
)

goldx.pipeline.compare_explanations(directory=gold_directory)
