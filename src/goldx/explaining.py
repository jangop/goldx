from captum.attr import GuidedBackprop, visualization


def explain(model, inputs, targets):
    guided = GuidedBackprop(model)
    attributions = guided.attribute(inputs, targets)
    for attr, image, target in zip(attributions, inputs, targets):
        fig, ax = visualization.visualize_image_attr(attr, image)

    return fig, ax
