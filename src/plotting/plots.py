from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.plotting.io import first_existing, load_csv_if_exists
from src.plotting.registry import REQUIRED_PLOTS, PlotSpec
from src.plotting.theme import CMAP_SUNSET, CUSTOM_COLORS, apply_theme

logger = logging.getLogger(__name__)


def _missing(ax: plt.Axes, spec: PlotSpec, reason: str) -> None:
    logger.warning("Cannot generate %s: %s", spec.plot_id, reason)
    ax.set_title(spec.title)
    ax.text(
        0.5,
        0.5,
        f"Missing data\n{', '.join(spec.required_data)}",
        ha="center",
        va="center",
        fontsize=12,
        color="#616161",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _metrics(run_dir: Path) -> pd.DataFrame | None:
    return load_csv_if_exists(run_dir / "metrics.csv")


def _plot_scalability(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["scalability.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"num_clients", "final_accuracy"}.issubset(df.columns):
        _missing(ax, spec, "scalability.csv requires num_clients, final_accuracy")
        return
    sns.lineplot(
        data=df,
        x="num_clients",
        y="final_accuracy",
        marker="o",
        linewidth=3,
        color=CUSTOM_COLORS[0],
        ax=ax,
    )
    ax.set_title(spec.title)
    ax.set_xlabel("Number of Agents (N)")
    ax.set_ylabel("Accuracy")


def _plot_non_iid(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["client_class_distribution.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or "client_id" not in df.columns:
        _missing(ax, spec, "client_class_distribution.csv requires client_id plus class columns")
        return
    df = df.set_index("client_id")
    row_sums = df.sum(axis=1).replace(0, np.nan)
    df.div(row_sums, axis=0).fillna(0).plot(kind="barh", stacked=True, ax=ax, width=0.8)
    ax.set_title(spec.title)
    ax.set_xlabel("Proportion of Data per Class")


def _plot_convergence(ax: plt.Axes, run_dir: Path, spec: PlotSpec, alpha_label: str) -> None:
    path = first_existing(
        run_dir, ["comparison_metrics.csv", "federated_round_metrics.csv", "metrics.csv"]
    )
    df = load_csv_if_exists(path) if path else None
    if df is None:
        _missing(ax, spec, "requires round/epoch and validation/global accuracy columns")
        return
    x_col = "round" if "round" in df.columns else "epoch" if "epoch" in df.columns else "step"
    y_candidates = ["federated/global_accuracy", "val/accuracy", "train/accuracy", "test/accuracy"]
    y_col = next((col for col in y_candidates if col in df.columns), None)
    if x_col not in df.columns or y_col is None:
        _missing(ax, spec, "no usable convergence columns found")
        return
    if "method" in df.columns:
        sns.lineplot(data=df, x=x_col, y=y_col, hue="method", ax=ax)
    else:
        sns.lineplot(data=df, x=x_col, y=y_col, ax=ax, color=CUSTOM_COLORS[0])
    ax.set_title(f"{spec.title} ({alpha_label})")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Accuracy")


def _plot_score_distribution(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["open_set_scores.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"unknown_score", "is_unknown"}.issubset(df.columns):
        _missing(ax, spec, "open_set_scores.csv requires unknown_score,is_unknown")
        return
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=0)
    sns.histplot(
        data=df,
        x="unknown_score",
        hue="is_unknown",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        ax=ax,
    )
    ax.set_title(spec.title)
    ax.set_xlabel("Unknown Score")


def _plot_xy(
    ax: plt.Axes,
    run_dir: Path,
    spec: PlotSpec,
    filename: str,
    x: str,
    y: str,
    hue: str | None = None,
) -> None:
    path = first_existing(run_dir, [filename])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {x, y}.issubset(df.columns):
        _missing(ax, spec, f"{filename} requires {x},{y}")
        return
    if len(df) > 5000:
        stride = int(np.ceil(len(df) / 5000))
        df = df.iloc[::stride].copy()
    if hue is not None and hue in df.columns:
        for label, group in df.groupby(hue):
            ax.plot(group[x].to_numpy(), group[y].to_numpy(), marker="o", label=str(label))
        ax.legend()
    else:
        ax.plot(df[x].to_numpy(), df[y].to_numpy(), marker="o", color=CUSTOM_COLORS[0])
    ax.set_title(spec.title)


def _plot_matrix(ax: plt.Axes, run_dir: Path, spec: PlotSpec, filename: str) -> None:
    path = first_existing(run_dir, [filename])
    if path is None:
        _missing(ax, spec, f"{filename} not found")
        return
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim != 2:
        _missing(ax, spec, f"{filename} is not a 2D matrix")
        return
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        matrix = np.nan_to_num(matrix)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=CMAP_SUNSET, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(spec.title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")


def _plot_box(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["seed_robustness.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"heterogeneity", "accuracy"}.issubset(df.columns):
        _missing(ax, spec, "seed_robustness.csv requires heterogeneity,accuracy")
        return
    sns.boxplot(data=df, x="heterogeneity", y="accuracy", ax=ax)
    ax.set_title(spec.title)


def _plot_latent(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["latent_embeddings.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"x", "y", "label"}.issubset(df.columns):
        _missing(ax, spec, "latent_embeddings.csv requires x,y,label")
        return
    sns.scatterplot(data=df, x="x", y="y", hue="label", s=18, ax=ax)
    ax.set_title(spec.title)
    ax.set_xticks([])
    ax.set_yticks([])


def render_q1_dashboard(
    run_dir: Path, output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    apply_theme()
    fig, axes = plt.subplots(7, 2, figsize=(24, 52), dpi=dpi)
    fig.suptitle(
        "Federated Learning & Open-Set Recognition Complete Q1 Analytics",
        fontsize=34,
        weight="bold",
        color="#424242",
        y=0.99,
    )
    flat_axes = axes.flatten()
    specs = list(REQUIRED_PLOTS)
    for idx, spec in enumerate(specs):
        ax = flat_axes[idx]
        if spec.plot_id == "scalability_nodes_vs_accuracy":
            _plot_scalability(ax, run_dir, spec)
        elif spec.plot_id == "non_iid_data_distribution":
            _plot_non_iid(ax, run_dir, spec)
        elif spec.plot_id == "convergence_mild_non_iid":
            _plot_convergence(ax, run_dir, spec, "alpha=10")
        elif spec.plot_id == "convergence_hard_non_iid":
            _plot_convergence(ax, run_dir, spec, "alpha=0.1")
        elif spec.plot_id == "known_unknown_score_distribution":
            _plot_score_distribution(ax, run_dir, spec)
        elif spec.plot_id == "openness_vs_auroc":
            _plot_xy(ax, run_dir, spec, "openness_metrics.csv", "openness", "auroc", "method")
        elif spec.plot_id == "unknown_detection_roc":
            _plot_xy(ax, run_dir, spec, "open_set_roc_curve.csv", "fpr", "tpr")
        elif spec.plot_id == "cross_dataset_generalization":
            _plot_xy(
                ax, run_dir, spec, "cross_dataset_metrics.csv", "dataset", "metric_value", "metric"
            )
        elif spec.plot_id == "confusion_matrix_before_osr":
            _plot_matrix(ax, run_dir, spec, "before_osr_confusion_matrix.csv")
        elif spec.plot_id == "confusion_matrix_after_osr":
            _plot_matrix(ax, run_dir, spec, "test_confusion_matrix.csv")
        elif spec.plot_id == "seed_robustness_boxplot":
            _plot_box(ax, run_dir, spec)
        elif spec.plot_id == "latent_space_separation":
            _plot_latent(ax, run_dir, spec)
        elif spec.plot_id == "communication_efficiency":
            _plot_xy(
                ax,
                run_dir,
                spec,
                "communication_metrics.csv",
                "cumulative_mb",
                "accuracy",
                "method",
            )
        elif spec.plot_id == "architectural_ablation":
            path = first_existing(run_dir, ["ablation_metrics.csv"])
            df = load_csv_if_exists(path) if path else None
            if df is None or not {"configuration", "macro_f1"}.issubset(df.columns):
                _missing(ax, spec, "ablation_metrics.csv requires configuration,macro_f1")
            else:
                sns.barplot(data=df, x="configuration", y="macro_f1", ax=ax, color=CUSTOM_COLORS[0])
                ax.tick_params(axis="x", rotation=20)
                ax.set_title(spec.title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        out = output_dir / f"complete_Q1_dashboard.{fmt}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        paths.append(out)
    plt.close(fig)
    return paths


def render_training_plots(
    run_dir: Path, output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    apply_theme()
    df = _metrics(run_dir)
    if df is None:
        logger.warning("No metrics.csv found in %s; skipping standalone training plots.", run_dir)
        return []
    paths: list[Path] = []
    x_col = "epoch" if "epoch" in df.columns else "step" if "step" in df.columns else None
    if x_col is None:
        return paths
    plot_specs = [
        ("training_loss", "Training Loss", ["train/loss", "train/double_q_loss"]),
        ("training_accuracy", "Train/Validation Accuracy", ["train/accuracy", "val/accuracy"]),
        ("reward_curve", "Reward Curve", ["train/reward"]),
        ("epsilon_schedule", "Epsilon Schedule", ["train/epsilon"]),
    ]
    for filename, title, columns in plot_specs:
        present = [col for col in columns if col in df.columns]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
        for col in present:
            sns.lineplot(data=df, x=x_col, y=col, label=col, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x_col)
        fig.tight_layout()
        for fmt in formats:
            out = output_dir / f"{filename}.{fmt}"
            fig.savefig(out, bbox_inches="tight", facecolor="white")
            paths.append(out)
        plt.close(fig)
    return paths
