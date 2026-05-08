#!/usr/bin/env python3
"""Create LangLoc/Qwen position + orientation threshold comparison tables."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping

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
    "3RScan subset": {
        "LangLoc": EVAL_DIR / "eval_metrics_mk5_all_weighted_3rscan_subset.json",
        "Qwen": EVAL_DIR / "baseline_eval_metrics_qwen_3rscan_subset.json",
    },
    "ScanNet": {
        "LangLoc": EVAL_DIR / "eval_metrics_mk5_weighted_scannet.json",
        "Qwen": EVAL_DIR / "baseline_eval_metrics_qwen_scannet.json",
    },
}

DIALOG_DIR = EVAL_DIR / "dialog_eval"
DIALOG_DATASETS: Mapping[str, Mapping[str, Path | str]] = {
    "3RScan subset": {
        "path": DIALOG_DIR / "replayed_metrics_with_vectors_iou.csv",
        "schema": "replayed",
    },
    "ScanNet": {
        "path": DIALOG_DIR / "qwen_results_all_3.csv",
        "schema": "qwen_results",
    },
}

COLORS = {
    "3RScan subset": "#0072B2",
    "ScanNet": "#D55E00",
}

POSE_THRESHOLDS = [(0.3, 30), (0.5, 30), (1.0, 45), (1.5, 45), (2.0, 45)]
COMPACT_POSE_THRESHOLDS = [(1.0, 45), (1.5, 45), (2.0, 45)]


def load_pose_arrays(path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(path.read_text())
    metrics = data["metrics"]
    return {
        "distance_error": np.asarray(
            [item["distance_error"] for item in metrics], dtype=np.float64
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
    }


def parse_metric(value: str) -> float:
    return float("nan") if value == "" or value == "nan" else float(value)


def load_replayed_dialog_arrays(
    path: Path, backend: str = "A3", which: str = "MAP"
) -> Dict[str, np.ndarray]:
    pos_errors: List[float] = []
    ang_errors: List[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["backend"] != backend or row["which"] != which:
                continue
            pos_errors.append(parse_metric(row["pos_err_m"]))
            ang_errors.append(parse_metric(row["rot_err_deg"]))

    return {
        "distance_error": np.asarray(pos_errors, dtype=np.float64),
        "angular_error_deg": np.asarray(ang_errors, dtype=np.float64),
    }


def load_qwen_results_dialog_arrays(path: Path) -> Dict[str, np.ndarray]:
    pos_errors: List[float] = []
    ang_errors: List[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("returncode") != "0" or row.get("has_error") not in {"0", "False", "false"}:
                continue
            pos_errors.append(parse_metric(row["a3_map_pos_err"]))
            ang_errors.append(parse_metric(row["a3_map_rot_err"]))

    return {
        "distance_error": np.asarray(pos_errors, dtype=np.float64),
        "angular_error_deg": np.asarray(ang_errors, dtype=np.float64),
    }


def load_dialog_arrays(config: Mapping[str, Path | str]) -> Dict[str, np.ndarray]:
    path = Path(config["path"])
    schema = str(config["schema"])
    if schema == "replayed":
        return load_replayed_dialog_arrays(path)
    if schema == "qwen_results":
        return load_qwen_results_dialog_arrays(path)
    raise ValueError(f"Unknown dialog schema {schema!r} for {path}")


def pose_accuracy(
    pos: np.ndarray, ang: np.ndarray, pos_threshold: float, ang_threshold: float
) -> float:
    valid = np.isfinite(pos) & np.isfinite(ang)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((pos[valid] <= pos_threshold) & (ang[valid] <= ang_threshold)))


def pct(value: float) -> str:
    return f"{100.0 * value:.0f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.0f} pp"


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


def add_table(
    ax: Axes,
    title: str,
    row_labels: List[str],
    col_labels: List[str],
    cell_text: List[List[str]],
    row_colors: Mapping[str, str],
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
            cell.set_text_props(weight="bold", color=row_colors[label])


def add_grouped_table(
    ax: Axes,
    title: str,
    col_labels: List[str],
    cell_text: List[List[str]],
) -> None:
    ax.axis("off")
    ax.set_title(title, pad=8, fontweight="bold")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold")
            continue

        row_kind = cell_text[row - 1][1]
        dataset = cell_text[row - 1][0]
        if col == 0 and dataset:
            cell.set_facecolor("#F7F7F7")
            cell.set_text_props(weight="bold", color=COLORS[dataset])
        elif col == 1:
            cell.set_facecolor("#F7F7F7")
            cell.set_text_props(weight="bold")

        if row_kind.startswith("Delta"):
            cell.set_facecolor("#FFF7E6")
            cell.set_text_props(weight="bold", color="#8A4B00")


def load_rows() -> Dict[str, Dict[str, np.ndarray]]:
    rows: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset_name, methods in DATASETS.items():
        for method_name, path in methods.items():
            rows[f"{dataset_name} ({method_name})"] = load_pose_arrays(path)
        if dataset_name in DIALOG_DATASETS:
            rows[f"{dataset_name} (LangLoc w/ dialog)"] = load_dialog_arrays(DIALOG_DATASETS[dataset_name])
    return rows


def accuracy_for(
    rows: Dict[str, Dict[str, np.ndarray]],
    dataset_name: str,
    method_name: str,
    pos_threshold: float,
    ang_threshold: float,
) -> float:
    values = rows[f"{dataset_name} ({method_name})"]
    return pose_accuracy(
        values["distance_error"],
        values["angular_error_deg"],
        pos_threshold,
        ang_threshold,
    )


def plot_full_table(rows: Dict[str, Dict[str, np.ndarray]]) -> None:
    row_labels = list(rows.keys())
    row_colors = {
        row_label: COLORS["3RScan subset"] if row_label.startswith("3RScan") else COLORS["ScanNet"]
        for row_label in row_labels
    }
    col_labels = [f"<= {pos_thr:g} m\n& <= {ang_thr:g} deg" for pos_thr, ang_thr in POSE_THRESHOLDS]
    cell_text: List[List[str]] = []
    for row_label in row_labels:
        pos = rows[row_label]["distance_error"]
        ang = rows[row_label]["angular_error_deg"]
        cell_text.append(
            [pct(pose_accuracy(pos, ang, pos_thr, ang_thr)) for pos_thr, ang_thr in POSE_THRESHOLDS]
        )

    fig, ax = plt.subplots(figsize=(10.2, 3.3))
    add_table(
        ax,
        "Position + orientation threshold accuracy",
        row_labels,
        col_labels,
        cell_text,
        row_colors,
    )
    fig.tight_layout()
    save_figure(fig, "position_orientation_threshold_accuracy_langloc_qwen")
    plt.close(fig)


def plot_compact_delta_table(rows: Dict[str, Dict[str, np.ndarray]]) -> None:
    row_labels = list(DATASETS.keys())
    row_colors = {dataset_name: COLORS[dataset_name] for dataset_name in row_labels}
    col_labels = [
        f"<= {pos_thr:g} m\n& <= {ang_thr:g} deg"
        for pos_thr, ang_thr in COMPACT_POSE_THRESHOLDS
    ]
    cell_text: List[List[str]] = []

    for dataset_name in row_labels:
        row: List[str] = []
        comparison_method = (
            "LangLoc w/ dialog"
            if f"{dataset_name} (LangLoc w/ dialog)" in rows
            else "LangLoc"
        )
        for pos_thr, ang_thr in COMPACT_POSE_THRESHOLDS:
            langloc = accuracy_for(rows, dataset_name, comparison_method, pos_thr, ang_thr)
            qwen = accuracy_for(rows, dataset_name, "Qwen", pos_thr, ang_thr)
            row.append(f"{pct(langloc)} / {pct(qwen)}\n({pp(langloc - qwen)})")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(7.6, 2.3))
    add_table(
        ax,
        "Position + orientation threshold accuracy: LangLoc / Qwen",
        row_labels,
        col_labels,
        cell_text,
        row_colors,
    )
    fig.text(
        0.5,
        0.03,
        "Each cell reports LangLoc / Qwen accuracy; parentheses show LangLoc - Qwen in percentage points.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_figure(fig, "position_orientation_threshold_accuracy_langloc_qwen_delta")
    plt.close(fig)


def plot_grouped_delta_table(rows: Dict[str, Dict[str, np.ndarray]]) -> None:
    cell_text: List[List[str]] = []
    col_labels = [
        "Dataset",
        "Method / Delta",
        *[f"<= {pos_thr:g} m\n& <= {ang_thr:g} deg" for pos_thr, ang_thr in POSE_THRESHOLDS],
    ]

    for dataset_name in DATASETS:
        qwen_row: List[str] = []
        langloc_row: List[str] = []
        dialog_row: List[str] = []
        delta_row: List[str] = []
        has_dialog = f"{dataset_name} (LangLoc w/ dialog)" in rows
        for pos_thr, ang_thr in POSE_THRESHOLDS:
            qwen = accuracy_for(rows, dataset_name, "Qwen", pos_thr, ang_thr)
            langloc = accuracy_for(rows, dataset_name, "LangLoc", pos_thr, ang_thr)
            comparison = langloc
            if has_dialog:
                comparison = accuracy_for(rows, dataset_name, "LangLoc w/ dialog", pos_thr, ang_thr)
                dialog_row.append(pct(comparison))
            qwen_row.append(pct(qwen))
            langloc_row.append(pct(langloc))
            delta_row.append(pp(comparison - qwen))

        cell_text.extend([[dataset_name, "Qwen", *qwen_row], ["", "LangLoc", *langloc_row]])
        if has_dialog:
            cell_text.append(["", "LangLoc w/ dialog", *dialog_row])
            cell_text.append(["", "Delta Dialog-Qwen", *delta_row])
        else:
            cell_text.append(["", "Delta LangLoc-Qwen", *delta_row])

    fig, ax = plt.subplots(figsize=(11.6, 4.0))
    add_grouped_table(
        ax,
        "Position + orientation threshold accuracy",
        col_labels,
        cell_text,
    )
    fig.tight_layout()
    save_figure(fig, "position_orientation_threshold_accuracy_langloc_qwen_grouped_delta")
    plt.close(fig)


def write_csv(rows: Dict[str, Dict[str, np.ndarray]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "position_orientation_threshold_accuracy_langloc_qwen.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "position_threshold_m",
                "angle_threshold_deg",
                "langloc_accuracy",
                "langloc_dialog_accuracy",
                "qwen_accuracy",
                "delta_comparison_minus_qwen",
            ]
        )
        for dataset_name in DATASETS:
            comparison_method = (
                "LangLoc w/ dialog"
                if f"{dataset_name} (LangLoc w/ dialog)" in rows
                else "LangLoc"
            )
            for pos_thr, ang_thr in POSE_THRESHOLDS:
                langloc = accuracy_for(rows, dataset_name, "LangLoc", pos_thr, ang_thr)
                dialog = (
                    accuracy_for(rows, dataset_name, "LangLoc w/ dialog", pos_thr, ang_thr)
                    if f"{dataset_name} (LangLoc w/ dialog)" in rows
                    else float("nan")
                )
                qwen = accuracy_for(rows, dataset_name, "Qwen", pos_thr, ang_thr)
                comparison = accuracy_for(rows, dataset_name, comparison_method, pos_thr, ang_thr)
                writer.writerow(
                    [
                        dataset_name,
                        f"{pos_thr:g}",
                        f"{ang_thr:g}",
                        f"{langloc:.6f}",
                        "" if np.isnan(dialog) else f"{dialog:.6f}",
                        f"{qwen:.6f}",
                        f"{comparison - qwen:.6f}",
                    ]
                )


def main() -> None:
    configure_matplotlib()
    rows = load_rows()
    plot_full_table(rows)
    plot_compact_delta_table(rows)
    plot_grouped_delta_table(rows)
    write_csv(rows)
    print(f"Wrote position + orientation threshold table to {OUT_DIR}")


if __name__ == "__main__":
    main()
