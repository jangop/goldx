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

    print(grad.dtype)

    print(f"{type(grad) = }, {type(self.mask) = }")
    print(f"{grad.dtype = }, {self.mask.dtype = }")

    print(grad)

    grad = grad * self.mask

    print(grad.dtype)

    return value, grad


LinfBasicIterativeAttack.value_and_grad = value_and_grad


def attack_image_with_mask(fmodel, image, label, mask):
    attack = LinfBasicIterativeAttack()
    attack.mask = torch.from_numpy(np.array(mask))

    return attack(fmodel, image, label, epsilons=0.3)
