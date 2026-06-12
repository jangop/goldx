"""Targeted adversarial attacks with the perturbation confined to a mask.

The mask is the ground-truth explanation: only pixels inside it are allowed
to change, so only they can explain the model's new prediction.
"""

import torch
from torch.nn import functional


class NormalizedModel(torch.nn.Module):
    """Wrap a classifier so it accepts images in [0, 1] and normalizes internally.

    Keeps the attack (and Captum attributions) in image space, where the
    epsilon bound and the mask are meaningful.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        model: torch.nn.Module,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def attack_image_with_mask(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = 0.05,
    rel_stepsize: float = 0.2,
    steps: int = 10,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[torch.Tensor, bool]:
    """Run a targeted L-infinity basic iterative attack restricted to ``mask``.

    ``model`` must accept images in ``bounds`` space (wrap with
    :class:`NormalizedModel` if it expects normalized input). ``mask`` is
    broadcast against the image; pixels where it is zero are never changed.

    Returns the adversarial image and whether the model now predicts ``target``.
    """
    image = image.detach()
    mask = mask.to(device=image.device, dtype=image.dtype)
    step_size = rel_stepsize * epsilon

    adversarial = image.clone()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = functional.cross_entropy(model(adversarial), target)
        (gradient,) = torch.autograd.grad(loss, adversarial)

        with torch.no_grad():
            # Targeted: descend the loss toward the target class.
            adversarial = adversarial - step_size * gradient.sign() * mask
            adversarial = image + (adversarial - image).clamp(-epsilon, epsilon) * mask
            adversarial = adversarial.clamp(*bounds)

    with torch.no_grad():
        prediction = model(adversarial).argmax(dim=1)
    success = bool((prediction == target).all().item())

    return adversarial.detach(), success
