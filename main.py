from pathlib import Path

import foolbox
import torchvision

import goldx

model = torchvision.models.vgg16(pretrained=True)
model.eval()

model2 = torchvision.models.vgg16(pretrained=True)
model2.eval()

preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)

fmodel = foolbox.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

source_directory = Path("data/source")
gold_directory = Path("data/gold")
evaluation_directory = Path("data/evaluation")

goldx.pipeline.prepare_ground_truths(
    source_directory=source_directory,
    target_directory=gold_directory,
    fmodel=fmodel,
    image_size=224,
)

goldx.pipeline.compute_explanations(
    gold_directory=gold_directory,
    fmodel=fmodel,
    model=model2,
    evaluation_directory=evaluation_directory,
)

goldx.pipeline.compare_explanations(directory=gold_directory)
