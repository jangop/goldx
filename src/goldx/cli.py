"""GoldX command line: run the pipeline, re-render reports, compare models."""

import argparse
import logging
from pathlib import Path

import torch
import torchvision

from . import pipeline, reporting
from .attacking import NormalizedModel

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
    "vit_b_16": (
        torchvision.models.vit_b_16,
        torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,
    ),
}


def _load_model(name: str) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constructor, weights = MODELS[name]
    return NormalizedModel(constructor(weights=weights)).to(device).eval()


def _run(arguments: argparse.Namespace) -> None:
    model = _load_model(arguments.model)

    pipeline.prepare_ground_truths(
        source_directory=arguments.source,
        target_directory=arguments.gold,
        image_size=arguments.image_size,
        model=model,
        min_mask_area=arguments.min_mask_area,
        max_mask_area=arguments.max_mask_area,
        attacks_per_image=arguments.attacks_per_image,
    )

    records = pipeline.compute_explanations(model=model, gold_directory=arguments.gold)

    reporting.render_report(arguments.gold, records)


def _report(arguments: argparse.Namespace) -> None:
    reporting.render_report(arguments.gold)


def _compare(arguments: argparse.Namespace) -> None:
    named_directories = {}
    for entry in arguments.runs:
        label, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"expected LABEL=GOLD_DIR, got {entry!r}")
        named_directories[label] = Path(path)
    reporting.render_comparison(named_directories, arguments.output)


def main() -> None:
    parser = argparse.ArgumentParser(prog="goldx", description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    run = subparsers.add_parser("run", help="attack, explain, score, report")
    run.add_argument("--model", choices=MODELS, default="resnet18")
    run.add_argument("--source", type=Path, default=Path("data/source"))
    run.add_argument("--gold", type=Path, default=Path("data/gold"))
    run.add_argument("--image-size", type=int, default=224)
    run.add_argument("--min-mask-area", type=float, default=0.1)
    run.add_argument("--max-mask-area", type=float, default=0.1)
    run.add_argument(
        "--attacks-per-image",
        type=int,
        default=1,
        help="independent (target, mask) attacks per source image",
    )
    run.set_defaults(handler=_run)

    report = subparsers.add_parser(
        "report", help="re-render figures from an existing results.csv"
    )
    report.add_argument("gold", type=Path)
    report.set_defaults(handler=_report)

    compare = subparsers.add_parser(
        "compare", help="cross-model comparison from finished runs"
    )
    compare.add_argument(
        "runs", nargs="+", metavar="LABEL=GOLD_DIR", help="e.g. resnet18=data/gold-r18"
    )
    compare.add_argument("--output", type=Path, default=Path("data"))
    compare.set_defaults(handler=_compare)

    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
