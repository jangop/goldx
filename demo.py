import foolbox
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

import goldx

mask = goldx.masking.generate_mask(224, 224)

model = torchvision.models.resnet18(pretrained=True)
model.eval()

preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)

data_dir = "./data/"

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        data_dir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
)

fmodel = foolbox.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(inputs.shape)
    print(targets.shape)

    targets = torch.tensor([0], dtype=torch.long)

    adv1, adv2, suc = goldx.attacking.attack_image_with_mask(
        fmodel=fmodel, image=inputs, label=targets, mask=mask
    )

    foolbox.plot.images(adv1)

plt.gcf().savefig("adv1.png")


fig, ax = goldx.explaining.explain(model, inputs, targets)
