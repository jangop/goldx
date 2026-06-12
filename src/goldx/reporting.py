"""Visual and tabular reports for a finished pipeline run.

Per case: one figure showing original, attacked image, amplified perturbation,
and every heatmap with the ground-truth mask outlined and its scores in the
title. Per run: ``results.csv`` (written by the pipeline), ``summary.png``,
and ``RESULTS.md``.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from .imagenet import CLASS_NAMES

logger = logging.getLogger(__name__)

BASELINE_KINDS = {"baseline", "oracle"}

KIND_COLORS = {
    "method": "#2b6cb0",
    "contrastive": "#2c7a7b",
    "baseline": "#a0aec0",
    "oracle": "#718096",
}


def load_records(gold_directory: Path) -> list[dict[str, Any]]:
    with (gold_directory / "results.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def write_records(gold_directory: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        logger.warning("no records to write")
        return
    path = gold_directory / "results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _amplified_difference(attacked: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Per-pixel difference scaled to use the full display range."""
    difference = attacked.astype(np.float32) - original.astype(np.float32)
    magnitude = np.abs(difference).max()
    if magnitude == 0:
        return np.full(attacked.shape[:2], 0.5)
    return (difference.sum(axis=2) / (2 * magnitude * 3)) + 0.5


def render_case_figure(
    *, directory: Path, target: int, records: list[dict[str, Any]]
) -> None:
    """One figure per (case, target): context row on top, heatmap row below."""
    original = np.array(Image.open(directory / "original.png"))
    attacked = np.array(Image.open(directory / f"{target}-attacked.png"))
    mask = np.array(Image.open(directory / f"{target}-mask.png"))
    original_label = (directory / "original-label.txt").read_text().split(",")[0]
    target_label = CLASS_NAMES[target].split(",")[0]

    records = sorted(records, key=lambda r: -float(r["iou"]))
    n_columns = max(4, len(records))

    fig, axes = plt.subplots(nrows=2, ncols=n_columns, figsize=(n_columns * 2.6, 5.8))
    for ax in axes.flat:
        ax.axis("off")

    confidence = float(records[0]["confidence"]) if records else float("nan")
    context = [
        (original, f"Original\n{original_label}", {}),
        (attacked, f"Attacked\n{target_label} ({confidence:.0%})", {}),
        (
            _amplified_difference(attacked, original),
            "Perturbation\n(amplified)",
            {"cmap": "coolwarm", "vmin": 0, "vmax": 1},
        ),
        (mask, "Ground truth", {"cmap": "gray"}),
    ]
    for ax, (image, title, kwargs) in zip(axes[0], context, strict=False):
        ax.imshow(image, **kwargs)
        ax.set_title(title, fontsize=9)

    for ax, record in zip(axes[1], records, strict=False):
        name = record["method"]
        heatmap = np.array(Image.open(directory / f"{target}-{name}.png"))
        top_k = np.array(Image.open(directory / f"{target}-{name}-mask.png"))

        ax.imshow(heatmap, cmap="gray")
        # Method's own top-k selection in green, ground truth outline in red.
        overlay = np.zeros((*top_k.shape, 4))
        overlay[top_k != 0] = (0.0, 0.8, 0.2, 0.45)
        ax.imshow(overlay)
        ax.contour(mask, levels=[127], colors="red", linewidths=2.0)

        kind = str(record["kind"])
        suffix = f" ({kind})" if kind in BASELINE_KINDS else ""
        ax.set_title(
            f"{name}{suffix}\n"
            f"IoU {float(record['iou']):.2f} · "
            f"mass {float(record['relevance_mass']):.2f} · "
            f"AUC {float(record['auc']):.2f}",
            fontsize=8,
        )

    fig.suptitle(
        f"{directory.name} → {target_label}   "
        "(green = method top-k, red = ground truth)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(directory / f"{target}-plot.png", dpi=150)
    plt.close(fig)


def render_summary(gold_directory: Path, records: list[dict[str, Any]]) -> None:
    """Aggregate over all cases: bar chart + markdown table."""
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[record["method"]].append(record)

    rows = []
    for method, group in by_method.items():
        iou = np.array([float(r["iou"]) for r in group])
        mass = np.array([float(r["relevance_mass"]) for r in group])
        auc = np.array([float(r["auc"]) for r in group])
        rows.append(
            {
                "method": method,
                "kind": group[0]["kind"],
                "n": len(group),
                "iou_mean": iou.mean(),
                "iou_std": iou.std(),
                "mass_mean": mass.mean(),
                "auc_mean": auc.mean(),
            }
        )
    rows.sort(key=lambda row: row["iou_mean"])

    # Bar chart: standard methods blue, contrastive regime teal,
    # model-blind baselines gray, oracle hatched.
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    colors = KIND_COLORS
    for i, row in enumerate(rows):
        ax.barh(
            i,
            row["iou_mean"],
            xerr=row["iou_std"],
            color=colors.get(str(row["kind"]), "#2b6cb0"),
            hatch="//" if row["kind"] == "oracle" else None,
        )
        ax.text(
            row["iou_mean"] + row["iou_std"] + 0.01,
            i,
            f"{row['iou_mean']:.2f}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(rows)), [str(row["method"]) for row in rows])
    ax.set_xlabel("IoU with ground-truth mask (mean ± std)")
    ax.set_xlim(0, 1)
    ax.set_title(f"GoldX summary — {len({r['case'] for r in records})} cases")
    fig.tight_layout()
    fig.savefig(gold_directory / "summary.png", dpi=150)
    plt.close(fig)

    lines = [
        "# GoldX Results",
        "",
        "| Method | Kind | n | IoU | Relevance mass | Pixel AUC |",
        "|---|---|---|---|---|---|",
    ]
    for row in reversed(rows):
        lines.append(
            f"| {row['method']} | {row['kind']} | {row['n']} "
            f"| {row['iou_mean']:.3f} ± {row['iou_std']:.3f} "
            f"| {row['mass_mean']:.3f} | {row['auc_mean']:.3f} |"
        )
    lines += [
        "",
        "Kinds: *method* sees only model + attacked image. *baseline* is "
        "model-blind. *oracle* reads the clean reference image (upper bound).",
        "",
        "![summary](summary.png)",
    ]
    (gold_directory / "RESULTS.md").write_text("\n".join(lines) + "\n")


def render_comparison(
    named_directories: dict[str, Path], output_directory: Path
) -> None:
    """Cross-model comparison from finished runs: grouped bars + table.

    ``named_directories`` maps a display label (e.g. model name) to the gold
    directory of a finished run.
    """
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    kinds: dict[str, str] = {}
    for label, directory in named_directories.items():
        by_method: dict[str, list[float]] = defaultdict(list)
        for record in load_records(directory):
            by_method[record["method"]].append(float(record["iou"]))
            kinds[record["method"]] = str(record["kind"])
        summaries[label] = {
            method: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
            for method, values in by_method.items()
        }

    methods = sorted(
        {method for summary in summaries.values() for method in summary},
        key=lambda method: np.mean(
            [
                summary[method]["mean"]
                for summary in summaries.values()
                if method in summary
            ]
        ),
    )
    labels = list(summaries)

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(methods) * len(labels) + 1.5))
    bar_height = 0.8 / len(labels)
    for column, label in enumerate(labels):
        positions = [
            i + (column - (len(labels) - 1) / 2) * bar_height
            for i in range(len(methods))
        ]
        means = [summaries[label].get(m, {}).get("mean", 0.0) for m in methods]
        stds = [summaries[label].get(m, {}).get("std", 0.0) for m in methods]
        ax.barh(positions, means, height=bar_height * 0.9, xerr=stds, label=label)
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlabel("IoU with ground-truth mask (mean ± std)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("GoldX model comparison")
    fig.tight_layout()
    output_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_directory / "comparison.png", dpi=150)
    plt.close(fig)

    header = "| Method | Kind | " + " | ".join(labels) + " |"
    divider = "|---|---|" + "---|" * len(labels)
    lines = ["# GoldX Model Comparison", "", header, divider]
    for method in reversed(methods):
        cells = []
        for label in labels:
            entry = summaries[label].get(method)
            cells.append(
                f"{entry['mean']:.3f} ± {entry['std']:.3f} (n={entry['n']})"
                if entry
                else "—"
            )
        lines.append(
            f"| {method} | {kinds.get(method, '?')} | " + " | ".join(cells) + " |"
        )
    lines += ["", "![comparison](comparison.png)"]
    (output_directory / "COMPARISON.md").write_text("\n".join(lines) + "\n")


def render_report(
    gold_directory: Path, records: list[dict[str, Any]] | None = None
) -> None:
    if records is None:
        records = load_records(gold_directory)
    if not records:
        logger.warning("no records found in %s", gold_directory)
        return

    by_case_target: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case_target[(str(record["case"]), int(record["target"]))].append(record)

    for (case, target), group in sorted(by_case_target.items()):
        render_case_figure(
            directory=gold_directory / case, target=target, records=group
        )

    render_summary(gold_directory, records)
