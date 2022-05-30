from typing import Callable, Tuple

import eagerpy as ep
import foolbox
import numpy as np
import torch
from foolbox.attacks import LinfBasicIterativeAttack


def value_and_grad(
    self,
    loss_fn: Callable[[ep.Tensor], ep.Tensor],
    x: ep.Tensor,
) -> Tuple[ep.Tensor, ep.Tensor]:
    value, grad = ep.value_and_grad(loss_fn, x)

    grad = grad * self.mask

    return value, grad


LinfBasicIterativeAttack.value_and_grad = value_and_grad


def attack_image_with_mask(fmodel, image, label, mask):
    attack = LinfBasicIterativeAttack()
    attack.mask = torch.from_numpy(np.array(mask)).to(fmodel.device)

    return attack(fmodel, image, label, epsilons=0.05)
