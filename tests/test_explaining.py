import numpy as np
import pytest
import torch

from goldx.explaining import ContrastiveLogit, normalize_attributions


class FixedLogits(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Logit i is the mean of channel i, scaled — differentiable and simple.
        return x.mean(dim=(2, 3)) * 10


def test_contrastive_logit_value():
    model = FixedLogits()
    x = torch.zeros(1, 3, 4, 4)
    x[:, 1] = 1.0  # logits: [0, 10, 0]
    contrastive = ContrastiveLogit(model, target=1, reference=2)
    assert contrastive(x).shape == (1, 1)
    assert contrastive(x).item() == pytest.approx(10.0)


def test_contrastive_logit_is_differentiable():
    model = FixedLogits()
    x = torch.rand(1, 3, 4, 4, requires_grad=True)
    score = ContrastiveLogit(model, target=0, reference=1)(x)
    score.sum().backward()
    assert x.grad is not None
    # d(score)/dx is +c on channel 0, -c on channel 1, 0 on channel 2.
    assert x.grad[0, 0].sum() > 0
    assert x.grad[0, 1].sum() < 0
    assert torch.allclose(x.grad[0, 2], torch.zeros(4, 4))


def test_normalize_attributions_shape_and_range():
    attributions = np.random.default_rng(0).normal(size=(3, 8, 8))
    heatmap = normalize_attributions(attributions)
    assert heatmap.shape == (8, 8)
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1


def test_normalize_attributions_rejects_2d():
    with pytest.raises(ValueError):
        normalize_attributions(np.zeros((8, 8)))
