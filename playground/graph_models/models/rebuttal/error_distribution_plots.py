#!/usr/bin/env python3
"""Error-distribution visualisations for the rebuttal.

Reviewer hook: EEmB —
  "Mean position error is high (~half the midpoint error) — likely
   skewed distribution.  It would be good to have more visualizations
   of this.  I suspect that, when the system identifies the correct
   cluster, the errors are very low; but when the incorrect cluster
   is found, then it is very high."

Produces four panel families per dataset:

  1. Position-error KDE per method, including LangLoc top-10 grid oracle.
  2. Position-error CDF per method, including LangLoc top-10 grid oracle.
  3. Distance-error × angular-error scatter (separates wrong-room
     failures from near-but-wrong-heading failures).
  4. Mean-vs-median bar chart per method to make the skew obvious.

Reads the cached eval JSONs under ``playground/graph_models/models/eval``;
no new model runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Tuple

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

DATASETS: Mapping[str, Mapping[str, Path]] = {
    "3rscan_subset": {
        "Midpoint": EVAL_DIR / "baseline_eval_metrics_mid_point_3rscan_subset.json",
        "Qwen baseline": EVAL_DIR / "baseline_eval_metrics_qwen_3rscan_subset.json",
        "LangLoc": EVAL_DIR / "eval_metrics_mk5_all_weighted_3rscan_subset.json",
    },
    "scannet": {
        "Midpoint": EVAL_DIR / "baseline_eval_metrics_mid_point_scannet.json",
        "Qwen baseline": EVAL_DIR / "baseline_eval_metrics_qwen_scannet.json",
        "LangLoc": EVAL_DIR / "eval_metrics_mk5_weighted_scannet.json",
    },
    "3rscan_full": {
        "Midpoint": EVAL_DIR / "baseline_eval_metrics_mid_point_3rscan.json",
        "Qwen baseline": EVAL_DIR / "baseline_eval_metrics_qwen_3rscan.json",
        "LangLoc": EVAL_DIR / "eval_metrics_mk5_all_3rscan.json",
    },
}


def load_per_query(path: Path, position_key: str = "distance_error") -> List[Tuple[float, float]]:
    d = json.loads(path.read_text())
    items = d["metrics"] if isinstance(d, dict) and "metrics" in d else d
    out: List[Tuple[float, float]] = []
    for it in items:
        de = it.get(position_key)
        ae = it.get("angular_error_deg")
        if de is None:
            continue
        angle = float("nan") if position_key != "distance_error" or ae is None else float(ae)
        out.append((float(de), angle))
    return out


COLOR = {
    "Midpoint": "#888888",
    "Qwen baseline": "#009E73",
    "LangLoc": "#0072B2",
    "LangLoc top-10 oracle": "#E69F00",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def method_tick_label(name: str) -> str:
    return {
        "Qwen baseline": "Qwen\nbaseline",
        "LangLoc top-10 oracle": "LangLoc\ntop-10\noracle",
    }.get(name, name)


def gather(dataset: str) -> Dict[str, List[Tuple[float, float]]]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {tuple(DATASETS)}")

    samples: Dict[str, List[Tuple[float, float]]] = {}
    for method, path in DATASETS[dataset].items():
        if not path.exists():
            print(f"skipping missing input: {path}")
            continue
        samples[method] = load_per_query(path)
        if method == "LangLoc":
            samples["LangLoc top-10 oracle"] = load_per_query(path, "topk_min_dist")
    return samples


def plot_kde_position(
    ax: Axes, samples: Dict[str, List[Tuple[float, float]]], xmax: float = 6.0
) -> None:
    """Position-error KDE per method."""
    grid = np.linspace(0, xmax, 600)
    for name, data in samples.items():
        if not data:
            continue
        pos = np.array([p for p, _ in data])
        # Silverman bandwidth
        n = len(pos)
        sigma = pos.std(ddof=1) if n > 1 else 1.0
        bw = max(1.06 * sigma * n ** (-1 / 5), 1e-3)
        kde = np.exp(-((grid[:, None] - pos[None, :]) ** 2) / (2 * bw * bw))
        density = kde.mean(axis=1) / (bw * (2 * np.pi) ** 0.5)
        c = COLOR[name]
        ax.plot(grid, density, label=name, color=c, lw=2)
        peak_x = grid[int(np.argmax(density))]
        ax.axvline(peak_x, color=c, ls="--", lw=1, alpha=0.7)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.set_title("Position-error KDE")


def plot_cdf(
    ax: Axes, samples: Dict[str, List[Tuple[float, float]]], xmax: float = 5.0
) -> None:
    grid = np.linspace(0, xmax, 1200)
    for name, data in samples.items():
        if not data:
            continue
        pos = np.array(sorted(p for p, _ in data))
        n = len(pos)
        cdf = np.searchsorted(pos, grid, side="right") / n
        ax.plot(grid, cdf, label=name, color=COLOR[name], lw=2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    for tau in (0.3, 0.5, 1.0):
        ax.axvline(tau, color="grey", ls=":", lw=0.6)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Recall (= CDF)")
    ax.legend(loc="lower right")
    ax.set_title("Position-error CDF / Recall@τ_pos")


def plot_scatter(
    ax: Axes, samples: Dict[str, List[Tuple[float, float]]], xmax: float = 5.0
) -> None:
    for name, data in samples.items():
        if not data:
            continue
        valid = [(p, r) for p, r in data if r == r]  # drop NaN angles
        if not valid:
            continue
        pos = np.array([p for p, _ in valid])
        rot = np.array([r for _, r in valid])
        ax.scatter(pos, rot, label=name, color=COLOR[name], alpha=0.5, s=14)
    ax.axhline(30, color="grey", ls=":", lw=0.6)
    ax.axvline(0.3, color="grey", ls=":", lw=0.6)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 180)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Angular error (°)")
    ax.legend(loc="upper right")
    ax.set_title("dist × ang per query (gates: 0.3 m / 30°)")


def plot_mean_median(ax: Axes, samples: Dict[str, List[Tuple[float, float]]]) -> None:
    names = list(samples.keys())
    means = [np.mean([p for p, _ in samples[n]]) if samples[n] else float("nan") for n in names]
    meds = [np.median([p for p, _ in samples[n]]) if samples[n] else float("nan") for n in names]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, means, w, label="Mean", color="#888888")
    ax.bar(x + w / 2, meds, w, label="Median", color="#d62728")
    for i, (m, md) in enumerate(zip(means, meds)):
        ax.text(i - w / 2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, md, f"{md:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([method_tick_label(n) for n in names], fontsize=8)
    ax.set_ylabel("Position error (m)")
    ax.legend()
    ax.set_title("Mean vs median (skew = mean − median)")


def render_dataset(
    dataset: str, samples: Dict[str, List[Tuple[float, float]]], xmax_kde: float = 6.0
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    plot_kde_position(axs[0, 0], samples, xmax=xmax_kde)
    plot_cdf(axs[0, 1], samples, xmax=5.0)
    plot_scatter(axs[1, 0], samples, xmax=5.0)
    plot_mean_median(axs[1, 1], samples)
    fig.suptitle(f"Error distribution — {dataset}", fontsize=13)
    fig.tight_layout()
    save_figure(fig, f"error_distribution_{dataset}")
    plt.close(fig)


def render_panel(
    dataset: str,
    samples: Dict[str, List[Tuple[float, float]]],
    suffix: str,
    plot_fn: Callable[[Axes, Dict[str, List[Tuple[float, float]]]], None],
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    plot_fn(ax, samples)
    fig.tight_layout()
    save_figure(fig, f"error_distribution_{dataset}_{suffix}")
    plt.close(fig)


def render_individual_panels(
    dataset: str, samples: Dict[str, List[Tuple[float, float]]]
) -> None:
    render_panel(dataset, samples, "kde", plot_kde_position)
    render_panel(dataset, samples, "cdf", plot_cdf)
    render_panel(dataset, samples, "scatter", plot_scatter)
    render_panel(dataset, samples, "mean_median", plot_mean_median)


def main() -> None:
    configure_matplotlib()
    for ds in DATASETS:
        s = gather(ds)
        # Skip empty rows so the legend isn't cluttered
        s = {k: v for k, v in s.items() if v}
        if not s:
            print(f"skipping {ds}: no samples found")
            continue
        render_dataset(ds, s)
        render_individual_panels(ds, s)


if __name__ == "__main__":
    main()
