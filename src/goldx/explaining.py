import numpy as np
from captum.attr._utils.visualization import _normalize_image_attr


def explain(*, Method, model, inputs, targets, args):

    inputs = inputs.cpu()

    explainer = Method(model)
    attributions = (
        explainer.attribute(inputs, target=targets, **args)
        .squeeze()
        .cpu()
        .detach()
        .numpy()
    )

    # Average over channels.

    if True:
        attributions = _normalize_image_attr(
            np.transpose(attributions, (1, 2, 0)), sign="positive"
        )
    else:
        attributions = attributions.mean(axis=0)

    return attributions
