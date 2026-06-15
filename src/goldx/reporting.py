"""Visual and tabular reports for a finished pipeline run.

Per case: one figure showing original, attacked image, amplified perturbation,
and every heatmap with the ground-truth mask outlined and its scores in the
title. Per run: ``results.parquet`` (written by the pipeline), ``summary.svg``
(theme-aware), and ``RESULTS.md``.
"""

import io
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from PIL import Image

from .imagenet import CLASS_NAMES

logger = logging.getLogger(__name__)

# One scored explanation per row. Typed so the parquet round trip preserves
# ints/floats instead of the everything-is-a-string CSV behaviour.
RESULTS_SCHEMA = pa.schema(
    [
        ("case", pa.string()),
        ("target", pa.int64()),
        ("method", pa.string()),
        ("kind", pa.string()),
        ("iou", pa.float64()),
        ("relevance_mass", pa.float64()),
        ("auc", pa.float64()),
        ("confidence", pa.float64()),
    ]
)
RESULTS_FILENAME = "results.parquet"
MANIFEST_FILENAME = "run.json"

# Charts are saved as theme-aware SVGs: transparent background, all text and
# axes drawn in this sentinel color, which is then rewritten to currentColor
# with a prefers-color-scheme stylesheet — so GitHub renders them legibly in
# both light and dark mode.
_SENTINEL_COLOR = "#010203"
_THEME_STYLE = (
    "<style>:root{color:#1f2328}@media (prefers-color-scheme:dark){:root{color:#e6edf3}}</style>"
)
_CHART_RC: dict[Any, Any] = {
    "text.color": _SENTINEL_COLOR,
    "axes.edgecolor": _SENTINEL_COLOR,
    "axes.labelcolor": _SENTINEL_COLOR,
    "xtick.color": _SENTINEL_COLOR,
    "ytick.color": _SENTINEL_COLOR,
    "svg.fonttype": "none",
}


def _save_theme_aware_svg(fig: Any, path: Path) -> None:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", transparent=True)
    svg = buffer.getvalue().replace(_SENTINEL_COLOR, "currentColor")
    svg = re.sub(r"(<svg\b[^>]*>)", r"\1" + _THEME_STYLE, svg, count=1)
    path.write_text(svg)


BASELINE_KINDS = {"baseline", "oracle"}

KIND_COLORS = {
    "method": "#2b6cb0",
    "contrastive": "#2c7a7b",
    "baseline": "#a0aec0",
    "oracle": "#718096",
}


def load_records(gold_directory: Path) -> list[dict[str, Any]]:
    path = gold_directory / RESULTS_FILENAME
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def write_records(gold_directory: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        logger.warning("no records to write")
        return
    gold_directory.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=RESULTS_SCHEMA)
    pq.write_table(table, gold_directory / RESULTS_FILENAME)


def write_manifest(gold_directory: Path, manifest: dict[str, Any]) -> None:
    gold_directory.mkdir(parents=True, exist_ok=True)
    (gold_directory / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")


def load_manifest(gold_directory: Path) -> dict[str, Any] | None:
    path = gold_directory / MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _n_cases(records: list[dict[str, Any]]) -> int:
    """Distinct (case, target) pairs that reached scoring."""
    return len({(r["case"], r["target"]) for r in records})


def _success_from_manifest(manifest: dict[str, Any] | None) -> str | None:
    """``"15/18"`` = attacks that succeeded / attacks attempted, or None.

    Uses the manifest's own ``attack_successes`` (attacks that produced a saved
    adversarial) so the "attacks succeeded" wording is literally true — this is
    distinct from the count of cases that later survived the PNG round trip to
    reach scoring. Returns None if the manifest is absent or lacks either count.
    """
    if manifest is None:
        return None
    successes = manifest.get("attack_successes")
    attempts = manifest.get("attack_attempts")
    if successes is None or attempts is None:
        return None
    return f"{successes}/{attempts}"


def _success_label(gold_directory: Path) -> str | None:
    return _success_from_manifest(load_manifest(gold_directory))


def _amplified_difference(attacked: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Per-pixel difference scaled to use the full display range."""
    difference = attacked.astype(np.float32) - original.astype(np.float32)
    magnitude = np.abs(difference).max()
    if magnitude == 0:
        return np.full(attacked.shape[:2], 0.5)
    return (difference.sum(axis=2) / (2 * magnitude * 3)) + 0.5


def render_case_figure(*, directory: Path, target: int, records: list[dict[str, Any]]) -> None:
    """One figure per (case, target): context row on top, heatmap row below."""
    original = np.array(Image.open(directory / "original.png"))
    attacked = np.array(Image.open(directory / f"{target}-attacked.png"))
    mask = np.array(Image.open(directory / f"{target}-mask.png"))
    original_label = (directory / "original-label.txt").read_text().split(",")[0]
    target_label = CLASS_NAMES[target].split(",")[0]

    records = sorted(records, key=lambda r: -float(r["iou"]))
    n_columns = max(4, len(records))

    # Mid-gray text reads on both GitHub themes; background stays transparent.
    with plt.rc_context({"text.color": "#888888"}):
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
            # GT mask may load as bool (mode "1") or uint8 0/255 (mode "L");
            # threshold to a 0/1 field and contour at 0.5 so the red outline
            # fires regardless of how the mask was saved.
            ax.contour((mask > 0).astype(float), levels=[0.5], colors="red", linewidths=2.0)

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
            f"{directory.name} → {target_label}   (green = method top-k, red = ground truth)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(directory / f"{target}-plot.png", dpi=150, transparent=True)
        plt.close(fig)


def render_summary(gold_directory: Path, records: list[dict[str, Any]]) -> None:
    """Aggregate over all cases: bar chart + markdown table."""
    manifest = load_manifest(gold_directory)
    success = _success_from_manifest(manifest)
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
    with plt.rc_context(_CHART_RC):
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
        for i, row in enumerate(rows):
            ax.barh(
                i,
                row["iou_mean"],
                xerr=row["iou_std"],
                color=KIND_COLORS.get(str(row["kind"]), "#2b6cb0"),
                hatch="//" if row["kind"] == "oracle" else None,
                error_kw={"ecolor": _SENTINEL_COLOR},
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
        n_cases = _n_cases(records)
        title = f"GoldX summary — {n_cases} cases"
        if success is not None:
            title += f" ({success} attacks succeeded)"
        ax.set_title(title)
        fig.tight_layout()
        _save_theme_aware_svg(fig, gold_directory / "summary.svg")
        plt.close(fig)

    lines = ["# GoldX Results", ""]
    if success is not None and manifest is not None:
        sources = manifest.get("source_images")
        per_image = manifest.get("attacks_per_image")
        detail = (
            f" ({sources} source images, {per_image} attacks each)"
            if sources is not None and per_image is not None
            else ""
        )
        lines += [f"**Attack success:** {success}{detail}.", ""]
    lines += [
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
        "![summary](summary.svg)",
    ]
    (gold_directory / "RESULTS.md").write_text("\n".join(lines) + "\n")


def render_comparison(named_directories: dict[str, Path], output_directory: Path) -> None:
    """Cross-model comparison from finished runs: grouped bars + table.

    ``named_directories`` maps a display label (e.g. model name) to the gold
    directory of a finished run.
    """
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    kinds: dict[str, str] = {}
    success: dict[str, str | None] = {}
    for label, directory in named_directories.items():
        records = load_records(directory)
        success[label] = _success_label(directory)
        by_method: dict[str, list[float]] = defaultdict(list)
        for record in records:
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
            [summary[method]["mean"] for summary in summaries.values() if method in summary]
        ),
    )
    labels = list(summaries)

    def display(label: str) -> str:
        rate = success[label]
        return f"{label} ({rate} attacks)" if rate else label

    with plt.rc_context(_CHART_RC):
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(methods) * len(labels) + 1.5))
        bar_height = 0.8 / len(labels)
        for column, label in enumerate(labels):
            positions = [
                i + (column - (len(labels) - 1) / 2) * bar_height for i in range(len(methods))
            ]
            means = [summaries[label].get(m, {}).get("mean", 0.0) for m in methods]
            stds = [summaries[label].get(m, {}).get("std", 0.0) for m in methods]
            ax.barh(
                positions,
                means,
                height=bar_height * 0.9,
                xerr=stds,
                label=display(label),
                error_kw={"ecolor": _SENTINEL_COLOR},
            )
        ax.set_yticks(range(len(methods)), methods)
        ax.set_xlabel("IoU with ground-truth mask (mean ± std)")
        ax.set_xlim(0, 1)
        legend = ax.legend(loc="lower right")
        legend.get_frame().set_alpha(0)
        ax.set_title("GoldX model comparison")
        fig.tight_layout()
        output_directory.mkdir(parents=True, exist_ok=True)
        _save_theme_aware_svg(fig, output_directory / "comparison.svg")
        plt.close(fig)

    header = "| Method | Kind | " + " | ".join(labels) + " |"
    divider = "|---|---|" + "---|" * len(labels)
    lines = ["# GoldX Model Comparison", "", header, divider]
    if any(success[label] for label in labels):
        rates = [success[label] or "—" for label in labels]
        lines.append("| **Attack success** | — | " + " | ".join(rates) + " |")
    for method in reversed(methods):
        cells = []
        for label in labels:
            entry = summaries[label].get(method)
            cells.append(
                f"{entry['mean']:.3f} ± {entry['std']:.3f} (n={entry['n']})" if entry else "—"
            )
        lines.append(f"| {method} | {kinds.get(method, '?')} | " + " | ".join(cells) + " |")
    lines += ["", "![comparison](comparison.svg)"]
    (output_directory / "COMPARISON.md").write_text("\n".join(lines) + "\n")


def render_report(gold_directory: Path, records: list[dict[str, Any]] | None = None) -> None:
    if records is None:
        records = load_records(gold_directory)
    if not records:
        logger.warning("no records found in %s", gold_directory)
        return

    by_case_target: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_case_target[(str(record["case"]), int(record["target"]))].append(record)

    for (case, target), group in sorted(by_case_target.items()):
        render_case_figure(directory=gold_directory / case, target=target, records=group)

    render_summary(gold_directory, records)
