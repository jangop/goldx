"""End-to-end pipeline: attack images, explain, score against ground truth."""

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion, Saliency
from PIL import Image

from .attacking import attack_image_with_mask
from .baselines import diff_oracle_heatmap, highpass_heatmap, random_heatmap
from .explaining import explain
from .imagenet import CLASS_NAMES
from .masking import generate_mask, mask_matrix
from .metrics import intersection_over_union, pixel_auc, relevance_mass
from .reporting import write_records

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
    attempt draws a fresh random target class and mask. Success is verified
    on the uint8-quantized image that will actually be saved — quantization
    can undo a marginal attack. Images with no successful attempt are skipped.
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
            if not success:
                continue

            # Verify survival of the uint8 quantization applied on save.
            attacked_image = _to_image(adversarial.squeeze(0))
            quantized = _to_tensor(attacked_image).unsqueeze(0).to(device)
            with torch.no_grad():
                quantized_prediction = int(model(quantized).argmax(dim=1).item())
            if quantized_prediction == target_label:
                break
            logger.info("attack on %s undone by quantization, retrying", file_path.name)
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
        attacked_image.save(case_directory / f"{target_label}-attacked.png")
        mask.save(case_directory / f"{target_label}-mask.png")
        (case_directory / f"{target_label}-label.txt").write_text(
            CLASS_NAMES[target_label]
        )


def _case_heatmaps(
    *,
    model: torch.nn.Module,
    batch: torch.Tensor,
    original: torch.Tensor,
    target: int,
    seed: int,
) -> dict[str, tuple[str, np.ndarray]]:
    """All heatmaps for one case: Captum methods, then calibration baselines."""
    heatmaps: dict[str, tuple[str, np.ndarray]] = {}
    for method, args in EXPLANATION_METHODS:
        heatmap = explain(
            method=method, model=model, inputs=batch, targets=target, args=args
        )
        heatmaps[method.__name__] = ("method", heatmap)

    attacked = batch.squeeze(0)
    height, width = attacked.shape[1:]
    rng = np.random.default_rng(seed)
    heatmaps["Random"] = ("baseline", random_heatmap(height, width, rng))
    heatmaps["HighPass"] = ("baseline", highpass_heatmap(attacked))
    heatmaps["DiffOracle"] = ("oracle", diff_oracle_heatmap(attacked, original))
    return heatmaps


def compute_explanations(
    *, model: torch.nn.Module, gold_directory: Path
) -> list[dict[str, Any]]:
    """Explain each attacked image and score everything against its mask.

    ``model`` must be the same [0, 1]-space model used for the attack. The
    attacked image is re-verified after the PNG round trip; cases where the
    attack no longer holds are skipped. Returns one record per
    (case, target, heatmap) and writes them to ``results.csv``.
    """
    device = next(model.parameters()).device
    records: list[dict[str, Any]] = []

    for case_directory in sorted(gold_directory.iterdir()):
        if not case_directory.is_dir():
            continue
        for image_path in sorted(case_directory.glob("*-attacked.png")):
            target = int(image_path.name.split("-")[0])

            attacked_image = Image.open(image_path).convert("RGB")
            batch = _to_tensor(attacked_image).unsqueeze(0).to(device)
            original = _to_tensor(
                Image.open(case_directory / "original.png").convert("RGB")
            ).to(device)

            with torch.no_grad():
                probabilities = model(batch).softmax(dim=1).squeeze(0)
            prediction = int(probabilities.argmax().item())
            confidence = float(probabilities[target].item())
            if prediction != target:
                logger.warning(
                    "%s: attack did not survive the round trip "
                    "(predicts %d, expected %d), skipping",
                    image_path,
                    prediction,
                    target,
                )
                continue

            mask_array = np.array(Image.open(case_directory / f"{target}-mask.png"))
            n_pixels_in_mask = int(np.count_nonzero(mask_array))

            heatmaps = _case_heatmaps(
                model=model,
                batch=batch,
                original=original,
                target=target,
                seed=target,
            )
            for name, (kind, heatmap) in heatmaps.items():
                _heatmap_to_image(heatmap).save(case_directory / f"{target}-{name}.png")

                # Binarize: brightest k pixels, k = ground-truth mask size.
                top_k = mask_matrix(heatmap, k=n_pixels_in_mask)
                Image.fromarray(top_k, mode="L").save(
                    case_directory / f"{target}-{name}-mask.png"
                )

                records.append(
                    {
                        "case": case_directory.name,
                        "target": target,
                        "method": name,
                        "kind": kind,
                        "iou": intersection_over_union(top_k, mask_array),
                        "relevance_mass": relevance_mass(heatmap, mask_array),
                        "auc": pixel_auc(heatmap, mask_array),
                        "confidence": confidence,
                    }
                )

    write_records(gold_directory, records)
    return records
