import os
import random
from pathlib import Path

import foolbox.models
import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion, Saliency
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import functional

from .attacking import attack_image_with_mask
from .explaining import explain
from .imagenet import CLASS_NAMES
from .masking import generate_mask, mask_matrix


def prepare_ground_truths(
    *,
    source_directory: Path,
    target_directory: Path,
    image_size: int,
    fmodel: foolbox.models.Model,
):
    for filename in os.listdir(source_directory):
        if any([filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]]):
            file_path = source_directory / filename
            image = Image.open(file_path)
            # Resize such that the shorter side is image_size.
            width, height = image.size
            if width < height:
                image = image.resize(
                    (image_size, int(height * image_size / width)), Image.ANTIALIAS
                )
            else:
                image = image.resize(
                    (int(width * image_size / height), image_size), Image.ANTIALIAS
                )

            # Crop center.
            width, height = image.size
            left = (width - image_size) // 2
            top = (height - image_size) // 2
            right = left + image_size
            bottom = top + image_size
            image = image.crop((left, top, right, bottom))

            # Predict.
            tensor = functional.to_tensor(image).to(fmodel.device)
            batch = tensor.unsqueeze(0)
            prediction = fmodel(batch)
            label = prediction.argmax(dim=1).item()

            # Save.
            target_path = target_directory / filename / "original.png"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(target_path)

            # Save prediction.
            target_path = target_directory / filename / "original-label.txt"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w") as f:
                f.write(CLASS_NAMES[label])

            # Select random target label.
            target_label = label
            while target_label == label:
                target_label = random.randint(0, 1000)

            # Generate mask.
            mask = generate_mask(image_size, image_size, min_area=0.1, max_area=0.1)

            # Attack original.
            adv1, adv2, suc = attack_image_with_mask(
                fmodel,
                batch,
                torch.tensor([target_label], dtype=torch.long, device=fmodel.device),
                mask,
            )

            # Save adversarial.
            attacked_image = functional.to_pil_image(adv1.squeeze_(0))
            target_path = target_directory / filename / f"{target_label}-attacked.png"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            attacked_image.save(target_path)

            # Save mask.
            target_path = target_directory / filename / f"{target_label}-mask.png"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(target_path)

            # Save label name.
            target_path = target_directory / filename / f"{target_label}-label.txt"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w") as f:
                f.write(CLASS_NAMES[target_label])


def compute_explanations(
    *,
    fmodel: foolbox.models.Model,
    model: torch.nn.Module,
    gold_directory: Path,
    evaluation_directory: Path,
):
    methods = [
        (Saliency, {}),
        (IntegratedGradients, {}),
        (
            Occlusion,
            {
                "sliding_window_shapes": (3, 32, 32),
                "strides": (3, 32, 32),
            },
        ),
    ]

    for name in os.listdir(gold_directory):
        path = gold_directory / name
        for filename in os.listdir(path):
            if filename.endswith("-attacked.png"):
                label = int(filename.split("-")[0])
                image_path = path / filename
                mask_path = path / f"{label}-mask.png"

                # Load image.
                attacked_image = Image.open(image_path)
                attacked_tensor = functional.to_tensor(attacked_image).to(fmodel.device)
                attacked_singular_batch = attacked_tensor.unsqueeze(0)
                preprocessed_singular_batch = fmodel._preprocess(
                    attacked_singular_batch
                ).raw

                # Explain.
                for method, args in methods:
                    method_name = method.__name__
                    attributions = explain(
                        model=model,
                        inputs=preprocessed_singular_batch,
                        targets=label,
                        Method=method,
                        args=args,
                    )
                    explanation = functional.to_pil_image(
                        torch.tensor(attributions), mode="L"
                    )
                    target_path = path / f"{label}-{method_name}.png"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    explanation.save(target_path)

                    # Load mask.
                    mask = Image.open(mask_path)
                    mask_array = np.array(mask)
                    n_pixels_in_mask = np.count_nonzero(mask_array)

                    # Set brightest k pixels in explanation to white, and black everywhere else.
                    explanation_mask_array = mask_matrix(
                        np.array(explanation), k=n_pixels_in_mask
                    )

                    # Save explanation mask.
                    target_path = path / f"{label}-{method_name}-mask.png"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    explanation_mask_array = Image.fromarray(
                        explanation_mask_array, mode="L"
                    )
                    explanation_mask_array.save(target_path)

                    # Compute intersection over union of mask and explanation_mask_array.
                    intersection = np.sum(
                        np.logical_and(np.array(explanation_mask_array), np.array(mask))
                    )
                    union = np.sum(
                        np.logical_or(np.array(explanation_mask_array), np.array(mask))
                    )
                    iou = intersection / union

                    # Save iou.
                    target_path = path / f"{label}-{method_name}-iou.txt"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, "w") as f:
                        f.write(str(iou))


def compare_explanations_for_one_example(*, directory: Path, target: int):
    original_path = directory / f"original.png"
    original_image = Image.open(original_path)

    attacked_path = directory / f"{target}-attacked.png"
    attacked_image = Image.open(attacked_path)

    gold_path = directory / f"{target}-mask.png"
    gold_image = Image.open(gold_path)

    explanations = dict()
    for filename in os.listdir(directory):
        if filename.endswith("-mask.png"):
            splits = filename.split("-")
            if len(splits) != 3:
                continue
            label = int(splits[0])

            if label != target:
                continue

            method = splits[1]

            explanation_path = directory / f"{target}-{method}.png"
            explanation_image = Image.open(explanation_path)

            explanation_mask_path = directory / f"{target}-{method}-mask.png"
            explanation_mask_image = Image.open(explanation_mask_path)

            iou_path = directory / f"{target}-{method}-iou.txt"
            with open(iou_path, "r") as f:
                iou = float(f.read())

            explanations[method] = (
                explanation_image,
                explanation_mask_image,
                iou,
            )

    n_methods = len(explanations)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_methods + 1,
        figsize=(n_methods * 3, 6),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    axes[0, 0].imshow(attacked_image)
    axes[0, 0].set_title(f"Attacked Image\n{target}: {CLASS_NAMES[target]}")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(gold_image)
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis("off")

    for i, (method, (explanation, explanation_mask, iou)) in enumerate(
        explanations.items()
    ):
        axes[0, i + 1].imshow(explanation, cmap="gray")
        axes[0, i + 1].set_title(method)
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(explanation_mask, cmap="gray")
        axes[1, i + 1].set_title(f"IoU w/ Ground Truth\n{iou:.2f}")
        axes[1, i + 1].axis("off")

    fig.tight_layout()
    fig.savefig(directory / f"{target}-plot.png")


def compare_explanations(*, directory: Path):
    for image_name in os.listdir(directory):
        image_dir = directory / image_name

        # Skip files.
        if not image_dir.is_dir():
            continue

        # Determine prepared target labels.
        targets = set()
        for filename in os.listdir(image_dir):
            if filename.endswith("-attacked.png"):
                target = int(filename.split("-")[0])
                targets.add(target)

        # Compare explanations for each target.
        for target in sorted(list(targets)):
            compare_explanations_for_one_example(directory=image_dir, target=target)
