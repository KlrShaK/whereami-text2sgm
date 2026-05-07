#!/usr/bin/env python3
"""Create rebuttal figures for fine-localization threshold analysis."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


MODELS_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = MODELS_DIR / "eval"
OUT_DIR = EVAL_DIR / "rebuttal_figures"

DATASETS = {
    "3RScan subset": EVAL_DIR / "eval_metrics_mk5_all_weighted_3rscan_subset.json",
    "ScanNet": EVAL_DIR / "eval_metrics_mk5_weighted_scannet.json",
}

COLORS = {
    "3RScan subset": "#0072B2",
    "ScanNet": "#D55E00",
}

POSITION_THRESHOLDS = [0.75, 1.0, 1.5, 2.0]
POSE_THRESHOLDS = [(0.3, 30), (0.5, 30), (1.0, 45), (1.5, 45), (2.0, 45)]


def load_metric_arrays(path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(path.read_text())
    metrics = data["metrics"]
    hit_radii = sorted(float(r) for r in metrics[0]["hit_masses"])
    return {
        "distance_error": np.asarray(
            [item["distance_error"] for item in metrics], dtype=np.float64
        ),
        "topk_min_dist": np.asarray(
            [item["topk_min_dist"] for item in metrics], dtype=np.float64
        ),
        "angular_error_deg": np.asarray(
            [
                np.nan
                if item.get("angular_error_deg") is None
                else item["angular_error_deg"]
                for item in metrics
            ],
            dtype=np.float64,
        ),
        "hit_radii": np.asarray(hit_radii, dtype=np.float64),
        "hit_masses": np.asarray(
            [
                [item["hit_masses"][str(radius)] for radius in hit_radii]
                for item in metrics
            ],
            dtype=np.float64,
        ),
    }


def ecdf(values: np.ndarray, xmax: float) -> Tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values[np.isfinite(values)])
    n = len(sorted_values)
    if n == 0:
        return np.asarray([0.0, xmax]), np.asarray([0.0, 0.0])
    x = np.concatenate(([0.0], sorted_values, [xmax]))
    y = np.concatenate(([0.0], np.arange(1, n + 1, dtype=np.float64) / n, [1.0]))
    return x, y


def accuracy(values: np.ndarray, threshold: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite <= threshold))


def pose_accuracy(pos: np.ndarray, ang: np.ndarray, pos_threshold: float, ang_threshold: float) -> float:
    valid = np.isfinite(pos) & np.isfinite(ang)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((pos[valid] <= pos_threshold) & (ang[valid] <= ang_threshold)))


def pct(value: float) -> str:
    return f"{100.0 * value:.0f}%"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")


def plot_position_ecdf(arrays: Dict[str, Dict[str, np.ndarray]]) -> None:
    xmax = 4.0
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for dataset_name, values in arrays.items():
        color = COLORS[dataset_name]
        x, y = ecdf(values["distance_error"], xmax=xmax)
        ax.step(
            x,
            y,
            where="post",
            color=color,
            linewidth=2.4,
            label=f"{dataset_name}: final pose",
        )
        x, y = ecdf(values["topk_min_dist"], xmax=xmax)
        ax.step(
            x,
            y,
            where="post",
            color=color,
            linewidth=2.0,
            linestyle=(0, (4, 2)),
            alpha=0.9,
            label=f"{dataset_name}: top-10 grid oracle",
        )

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("Position error threshold (m)")
    ax.set_ylabel("Fraction localized within threshold")
    ax.set_title("Fine-localization accuracy vs. error threshold")
    ax.set_xticks([0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, which="major", color="#D8D8D8", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout()
    save_figure(fig, "fine_localization_position_ecdf")
    plt.close(fig)


def plot_hit_mass_curve(arrays: Dict[str, Dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    for dataset_name, values in arrays.items():
        color = COLORS[dataset_name]
        radii = values["hit_radii"]
        hit_masses = values["hit_masses"]
        mean = np.mean(hit_masses, axis=0)
        median = np.median(hit_masses, axis=0)
        q25 = np.percentile(hit_masses, 25, axis=0)
        q75 = np.percentile(hit_masses, 75, axis=0)

        ax.fill_between(
            radii,
            q25,
            q75,
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )
        ax.plot(
            radii,
            mean,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=5.0,
            label=f"{dataset_name}: mean Hit@r",
        )
        ax.plot(
            radii,
            median,
            color=color,
            linewidth=2.0,
            marker="s",
            markersize=4.5,
            linestyle=(0, (4, 2)),
            label=f"{dataset_name}: median Hit@r",
        )

    ax.set_xlim(0.7, 2.55)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Radius around ground-truth position (m)")
    ax.set_ylabel("Probability mass within radius")
    ax.set_title("Hit@r: localization probability mass near ground truth")
    ax.set_xticks([0.75, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, which="major", color="#D8D8D8", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")

    fig.tight_layout()
    save_figure(fig, "fine_localization_hit_at_radius")
    plt.close(fig)


def add_table(
    ax: Axes,
    title: str,
    row_labels: List[str],
    col_labels: List[str],
    cell_text: List[List[str]],
) -> None:
    ax.axis("off")
    ax.set_title(title, pad=8, fontweight="bold")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold")
        if col == -1 and row > 0:
            label = row_labels[row - 1]
            cell.set_facecolor("#F7F7F7")
            cell.set_text_props(weight="bold", color=COLORS[label])


def plot_threshold_table(arrays: Dict[str, Dict[str, np.ndarray]]) -> None:
    row_labels = list(arrays.keys())

    position_col_labels = [f"<= {thr:g} m" for thr in POSITION_THRESHOLDS]
    position_cells: List[List[str]] = []
    for dataset_name in row_labels:
        pos = arrays[dataset_name]["distance_error"]
        position_cells.append([pct(accuracy(pos, thr)) for thr in POSITION_THRESHOLDS])

    pose_col_labels = [f"<= {pt:g} m\n& <= {at:g} deg" for pt, at in POSE_THRESHOLDS]
    pose_cells: List[List[str]] = []
    for dataset_name in row_labels:
        pos = arrays[dataset_name]["distance_error"]
        ang = arrays[dataset_name]["angular_error_deg"]
        pose_cells.append([pct(pose_accuracy(pos, ang, pt, at)) for pt, at in POSE_THRESHOLDS])

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.7))
    add_table(
        axes[0],
        "Position-only threshold accuracy",
        row_labels,
        position_col_labels,
        position_cells,
    )
    add_table(
        axes[1],
        "Position + orientation threshold accuracy",
        row_labels,
        pose_col_labels,
        pose_cells,
    )
    fig.suptitle("Thresholded fine-localization accuracy", y=0.98, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Values are percentages over 100 evaluated frames per dataset.",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    save_figure(fig, "fine_localization_threshold_accuracy_table")
    plt.close(fig)


def write_summary_csv(arrays: Dict[str, Dict[str, np.ndarray]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "threshold_accuracy_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "metric", "threshold", "accuracy"])
        for dataset_name, values in arrays.items():
            pos = values["distance_error"]
            ang = values["angular_error_deg"]
            topk = values["topk_min_dist"]
            for thr in POSITION_THRESHOLDS:
                writer.writerow([dataset_name, "final_position", f"{thr:g}m", f"{accuracy(pos, thr):.6f}"])
            for thr in POSITION_THRESHOLDS:
                writer.writerow([dataset_name, "top10_grid_oracle", f"{thr:g}m", f"{accuracy(topk, thr):.6f}"])
            for radius, mean_hit, median_hit in zip(
                values["hit_radii"],
                np.mean(values["hit_masses"], axis=0),
                np.median(values["hit_masses"], axis=0),
            ):
                writer.writerow([dataset_name, "hit_mass_mean", f"{radius:g}m", f"{mean_hit:.6f}"])
                writer.writerow([dataset_name, "hit_mass_median", f"{radius:g}m", f"{median_hit:.6f}"])
            for pt, at in POSE_THRESHOLDS:
                writer.writerow(
                    [
                        dataset_name,
                        "final_position_and_angle",
                        f"{pt:g}m_{at:g}deg",
                        f"{pose_accuracy(pos, ang, pt, at):.6f}",
                    ]
                )


def main() -> None:
    configure_matplotlib()
    arrays = {name: load_metric_arrays(path) for name, path in DATASETS.items()}
    plot_position_ecdf(arrays)
    plot_hit_mass_curve(arrays)
    plot_threshold_table(arrays)
    write_summary_csv(arrays)
    print(f"Wrote rebuttal figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
