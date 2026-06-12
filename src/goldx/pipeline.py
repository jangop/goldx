"""End-to-end pipeline: attack images, explain, score against ground truth."""

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion, Saliency
from matplotlib import pyplot as plt
from PIL import Image

from .attacking import attack_image_with_mask
from .explaining import explain
from .imagenet import CLASS_NAMES
from .masking import generate_mask, mask_matrix
from .metrics import intersection_over_union

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

EXPLANATION_METHODS: list[tuple[type, dict[str, Any]]] = [
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


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB image -> float tensor (C, H, W) in [0, 1]."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _to_image(tensor: torch.Tensor) -> Image.Image:
    """Float tensor (C, H, W) in [0, 1] -> PIL RGB image."""
    array = tensor.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def _heatmap_to_image(heatmap: np.ndarray) -> Image.Image:
    """Float array (H, W) in [0, 1] -> grayscale PIL image."""
    return Image.fromarray((heatmap * 255).round().astype(np.uint8), mode="L")


def _resize_and_crop(image: Image.Image, size: int) -> Image.Image:
    """Resize so the shorter side is ``size``, then crop the center square."""
    width, height = image.size
    if width < height:
        image = image.resize(
            (size, round(height * size / width)), Image.Resampling.LANCZOS
        )
    else:
        image = image.resize(
            (round(width * size / height), size), Image.Resampling.LANCZOS
        )

    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def prepare_ground_truths(
    *,
    source_directory: Path,
    target_directory: Path,
    image_size: int,
    model: torch.nn.Module,
    min_mask_area: float = 0.1,
    max_mask_area: float = 0.1,
    max_attempts: int = 5,
) -> None:
    """Attack each source image inside a random mask and save the results.

    ``model`` must accept images in [0, 1] (see ``NormalizedModel``). Each
    attempt draws a fresh random target class and mask; images where no
    attempt produces a successful attack are skipped.
    """
    device = next(model.parameters()).device

    for file_path in sorted(source_directory.iterdir()):
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = _resize_and_crop(Image.open(file_path).convert("RGB"), image_size)
        batch = _to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            label = int(model(batch).argmax(dim=1).item())

        for _ in range(max_attempts):
            target_label = label
            while target_label == label:
                target_label = random.randrange(len(CLASS_NAMES))

            mask = generate_mask(
                image_size, image_size, min_area=min_mask_area, max_area=max_mask_area
            )
            mask_tensor = torch.from_numpy(np.array(mask))

            adversarial, success = attack_image_with_mask(
                model,
                batch,
                torch.tensor([target_label], dtype=torch.long, device=device),
                mask_tensor,
            )
            if success:
                break
        else:
            logger.warning(
                "no successful attack for %s after %d attempts, skipping",
                file_path.name,
                max_attempts,
            )
            continue

        case_directory = target_directory / file_path.name
        case_directory.mkdir(parents=True, exist_ok=True)

        image.save(case_directory / "original.png")
        (case_directory / "original-label.txt").write_text(CLASS_NAMES[label])
        _to_image(adversarial.squeeze(0)).save(
            case_directory / f"{target_label}-attacked.png"
        )
        mask.save(case_directory / f"{target_label}-mask.png")
        (case_directory / f"{target_label}-label.txt").write_text(
            CLASS_NAMES[target_label]
        )


def compute_explanations(*, model: torch.nn.Module, gold_directory: Path) -> None:
    """Explain each attacked image and score the explanation against its mask.

    ``model`` must be the same [0, 1]-space model used for the attack.
    """
    device = next(model.parameters()).device

    for case_directory in sorted(gold_directory.iterdir()):
        if not case_directory.is_dir():
            continue
        for image_path in sorted(case_directory.glob("*-attacked.png")):
            label = int(image_path.name.split("-")[0])

            attacked_image = Image.open(image_path).convert("RGB")
            batch = _to_tensor(attacked_image).unsqueeze(0).to(device)

            mask_array = np.array(Image.open(case_directory / f"{label}-mask.png"))
            n_pixels_in_mask = int(np.count_nonzero(mask_array))

            for method, args in EXPLANATION_METHODS:
                method_name = method.__name__
                heatmap = explain(
                    method=method,
                    model=model,
                    inputs=batch,
                    targets=label,
                    args=args,
                )
                _heatmap_to_image(heatmap).save(
                    case_directory / f"{label}-{method_name}.png"
                )

                # Binarize: brightest k pixels, k = ground-truth mask size.
                explanation_mask = mask_matrix(heatmap, k=n_pixels_in_mask)
                Image.fromarray(explanation_mask, mode="L").save(
                    case_directory / f"{label}-{method_name}-mask.png"
                )

                iou = intersection_over_union(explanation_mask, mask_array)
                (case_directory / f"{label}-{method_name}-iou.txt").write_text(str(iou))


def compare_explanations_for_one_example(*, directory: Path, target: int) -> None:
    attacked_image = Image.open(directory / f"{target}-attacked.png")
    gold_image = Image.open(directory / f"{target}-mask.png")

    explanations = {}
    for mask_path in sorted(directory.glob(f"{target}-*-mask.png")):
        splits = mask_path.name.split("-")
        if len(splits) != 3:
            continue
        method = splits[1]

        explanation_image = Image.open(directory / f"{target}-{method}.png")
        explanation_mask_image = Image.open(mask_path)
        iou = float((directory / f"{target}-{method}-iou.txt").read_text())

        explanations[method] = (explanation_image, explanation_mask_image, iou)

    n_methods = len(explanations)
    if n_methods == 0:
        logger.warning("no explanations found in %s for target %d", directory, target)
        return

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
    plt.close(fig)


def compare_explanations(*, directory: Path) -> None:
    for case_directory in sorted(directory.iterdir()):
        if not case_directory.is_dir():
            continue

        targets = {
            int(path.name.split("-")[0])
            for path in case_directory.glob("*-attacked.png")
        }
        for target in sorted(targets):
            compare_explanations_for_one_example(
                directory=case_directory, target=target
            )
