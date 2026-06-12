import numpy as np
import torch

from goldx.attacking import NormalizedModel, attack_image_with_mask
from goldx.masking import generate_mask


class TinyClassifier(torch.nn.Module):
    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.conv(x)).flatten(1))


def test_perturbation_confined_to_mask_and_epsilon():
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    image = torch.rand(1, 3, 32, 32)
    mask = torch.from_numpy(np.array(generate_mask(32, 32)))
    target = torch.tensor([1])
    epsilon = 0.05

    adversarial, _ = attack_image_with_mask(model, image, target, mask, epsilon=epsilon)

    perturbation = adversarial - image
    outside = perturbation[:, :, ~mask.bool()]
    assert torch.all(outside == 0)
    assert perturbation.abs().max() <= epsilon + 1e-6
    assert adversarial.min() >= 0
    assert adversarial.max() <= 1


def test_attack_can_succeed_on_easy_target():
    torch.manual_seed(0)
    model = TinyClassifier(n_classes=2).eval()
    image = torch.rand(1, 3, 16, 16)
    mask = torch.ones(16, 16)

    with torch.no_grad():
        label = model(image).argmax(dim=1)
    target = torch.tensor([1 - label.item()])

    _, success = attack_image_with_mask(
        model, image, target, mask, epsilon=0.5, steps=50
    )
    assert success


def test_normalized_model_normalizes():
    inner = torch.nn.Identity()
    model = NormalizedModel(inner)
    x = torch.zeros(1, 3, 2, 2)
    out = model(x)
    expected = (0 - model.mean) / model.std
    assert torch.allclose(out, expected.expand_as(out))
