import os
import random
from pathlib import Path

import foolbox.models
import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion, Saliency
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
        if filename.endswith(".jpg"):
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
                    target_path = path / f"{label}-explanation-iou.txt"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, "w") as f:
                        f.write(str(iou))
