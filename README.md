# GoldX: Ground-Truth Explanations for Visual Classifiers
Most popular explanations for visual classifiers come in the form of heatmaps.
[Captum](https://captum.ai) provides implementations of several methods to produce such explanations.

The usefulness/validity of such explanations has been called into question frequently [[Adebayo et al. 2020](https://arxiv.org/abs/2011.05429)].

I am not going to solve the problem of explanations that are difficult to understand and need explanations themselves.
I will not propose a method to produce _the best explanations_.
Instead, here I reimplement a way to generate ground-truth explanations.
These can then be used to at least evaluate explanations **quantitatively**.

Note that I did so before for a publication, with limited scope [[Göpfert et al. 2019](https://arxiv.org/abs/1910.09239)].

Eventually, I would like to provide a tool to evaluate further explanation methods, but for now I am going to focus my limited programming skills on a broad evaluation.

## How It Works

1. **Attack** — pick an image, pick a random target class, and run a targeted
   L∞ BIM attack whose perturbation is confined to a random elliptical mask.
   The mask *is* the ground-truth explanation: only pixels inside it changed,
   so only they can explain the new prediction.
2. **Explain** — run Captum methods (Saliency, Integrated Gradients, Occlusion)
   on the attacked image for the target class.
3. **Score** — binarize each heatmap (top-k pixels, k = mask size) and compute
   its IoU with the ground-truth mask.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

```console
$ uv sync
$ mkdir -p data/source   # drop .png/.jpg images here
$ uv run goldx run --model resnet18 --attacks-per-image 2
$ uv run goldx report data/gold                      # re-render figures
$ uv run goldx compare r18=data/gold-r18 vit=data/gold-vit --output data
```

`goldx run` writes everything into the gold directory (default `data/gold`):
per-case artifacts and figures, `results.csv`, `RESULTS.md`, `summary.png`.
Models: `resnet18`, `resnet50`, `vgg16`, `vit_b_16`.

Each Captum method is evaluated in two regimes: **standard** (attribute the
target logit) and **contrastive** (attribute the *target − original* logit
difference — the "why t rather than y" question that the ground truth
actually answers).

Development:

```console
$ uv run pre-commit install
$ uv run pytest
```

## Results

Three calibration references accompany every run: **Random** (chance floor,
IoU ≈ 0.05 at 10% mask area), **HighPass** (a model-blind Laplacian filter —
any method that doesn't beat it is only detecting perturbation noise, not
explaining the model), and **DiffOracle** (pixel difference to the clean
image — privileged information, the ceiling).

Smoke run, ResNet-18, 3 images ([full table](examples/RESULTS.md)):

![](examples/summary.png)

Occlusion clearly beats the noise detector; Integrated Gradients does not —
on pixel AUC it scores *below* HighPass. Treat gradient-method results on
this benchmark with suspicion.

Per-case view — green is the method's top-k selection, red the ground truth:

![](examples/bee-958-plot.png)
![](examples/snail-652-plot.png)

Each run writes `results.csv`, `RESULTS.md`, `summary.png`, and one such plot
per case into the gold directory.
