from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from src.plotting.io import first_existing, load_csv_if_exists
from src.plotting.registry import REQUIRED_PLOTS, PlotSpec
from src.plotting.theme import CMAP_SUNSET, CUSTOM_COLORS, apply_theme

logger = logging.getLogger(__name__)

PLOT_FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "scalability_nodes_vs_accuracy": (8.6, 5.6),
    "non_iid_data_distribution": (10.2, 6.4),
    "convergence_mild_non_iid": (9.0, 5.5),
    "convergence_hard_non_iid": (9.0, 5.5),
    "known_unknown_score_distribution": (8.8, 5.4),
    "openness_vs_auroc": (8.5, 5.4),
    "unknown_detection_roc": (8.5, 5.4),
    "cross_dataset_generalization": (8.8, 5.6),
    "confusion_matrix_before_osr": (7.2, 6.4),
    "confusion_matrix_after_osr": (7.2, 6.4),
    "seed_robustness_boxplot": (8.6, 5.4),
    "latent_space_separation": (7.4, 6.4),
    "communication_efficiency": (8.8, 5.4),
    "architectural_ablation": (8.8, 5.4),
}

TRAINING_FIGURE_SIZE = (8.4, 4.8)


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


def _figure_size(plot_id: str) -> tuple[float, float]:
    return PLOT_FIGURE_SIZES.get(plot_id, (8.5, 5.25))


def _maybe_percent(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(finite) <= 1.0 + 1e-9 and np.nanmin(finite) >= -1e-9:
        return arr * 100.0
    return arr


def _first_existing_column(columns: pd.Index | list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _first_existing_in_dirs(dirs: tuple[Path, ...], names: list[str]) -> Path | None:
    for base_dir in dirs:
        path = first_existing(base_dir, names)
        if path is not None:
            return path
    return None


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        paths.append(out)
    return paths


def _select_metric_frame(
    df: pd.DataFrame, metric_candidates: tuple[str, ...]
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    metric_name_col = _first_existing_column(df.columns, ("metric_name", "metric"))
    metric_value_col = _first_existing_column(df.columns, ("metric_value", "value"))
    if metric_name_col is not None and metric_value_col is not None:
        metric_names = df[metric_name_col].astype(str)
        for candidate in metric_candidates:
            subset = df.loc[metric_names == candidate].copy()
            if not subset.empty:
                return subset, metric_value_col, candidate
        if metric_names.nunique() == 1:
            return df.copy(), metric_value_col, metric_names.iloc[0]
        return None, None, None

    for candidate in metric_candidates:
        if candidate in df.columns:
            return df.copy(), candidate, candidate

    return None, None, None


def _metrics(run_dir: Path) -> pd.DataFrame | None:
    return load_csv_if_exists(run_dir / "metrics.csv")


def _plot_scalability(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["scalability.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None:
        _missing(ax, spec, "scalability.csv not found")
        return
    x_col = _first_existing_column(df.columns, ("num_clients", "num_nodes", "clients"))
    y_col = _first_existing_column(df.columns, ("final_accuracy", "accuracy"))
    if x_col is None or y_col is None:
        _missing(ax, spec, "scalability.csv requires num_clients/num_nodes and final_accuracy")
        return
    plot_df = df[[x_col, y_col]].dropna().copy()
    plot_df = plot_df.sort_values(x_col)
    plot_df["_plot_accuracy"] = _maybe_percent(plot_df[y_col])
    sns.lineplot(
        data=plot_df,
        x=x_col,
        y="_plot_accuracy",
        marker="o",
        markersize=8,
        linewidth=2.8,
        color=CUSTOM_COLORS[0],
        ax=ax,
    )
    ax.set_title(spec.title)
    ax.set_xlabel("Number of Clients")
    ax.set_ylabel("Final Accuracy (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(alpha=0.25)
    for x_val, y_val in zip(plot_df[x_col], plot_df["_plot_accuracy"], strict=False):
        ax.annotate(f"{y_val:.1f}%", (x_val, y_val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)


def _plot_non_iid(
    ax: plt.Axes,
    run_dir: Path,
    spec: PlotSpec,
    *,
    preprocess_dir: Path | None = None,
) -> None:
    search_dirs = (preprocess_dir, run_dir) if preprocess_dir is not None else (run_dir,)
    path = _first_existing_in_dirs(search_dirs, ["client_class_distribution.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or "client_id" not in df.columns:
        _missing(ax, spec, "client_class_distribution.csv requires client_id plus class columns")
        return
    class_cols = [col for col in df.columns if col != "client_id"]
    if not class_cols:
        _missing(ax, spec, "client_class_distribution.csv requires class columns")
        return
    plot_df = df.sort_values("client_id").set_index("client_id")[class_cols]
    row_sums = plot_df.sum(axis=1).replace(0, np.nan)
    proportions = plot_df.div(row_sums, axis=0).fillna(0) * 100.0
    proportions.index = [f"Client {int(client_id)}" for client_id in proportions.index]
    proportions.plot(kind="barh", stacked=True, ax=ax, width=0.82)
    ax.set_title(spec.title)
    ax.set_xlabel("Class Share (%)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.invert_yaxis()


def _plot_convergence(ax: plt.Axes, run_dir: Path, spec: PlotSpec, alpha_label: str) -> None:
    path = first_existing(
        run_dir,
        ["comparison_metrics.csv", "federated_history.csv", "federated_round_metrics.csv", "metrics.csv"],
    )
    df = load_csv_if_exists(path) if path else None
    if df is None:
        _missing(ax, spec, "requires round/epoch and validation/global accuracy columns")
        return
    metric_candidates = (
        "federated/global_accuracy",
        "federated/accuracy",
        "val/accuracy",
        "test/accuracy",
        "train/accuracy",
        "accuracy",
        "macro_f1",
        "federated/macro_f1",
    )
    metric_df, y_col, metric_name = _select_metric_frame(df, metric_candidates)
    if metric_df is None or y_col is None:
        _missing(ax, spec, "no usable convergence columns found")
        return
    x_col = _first_existing_column(
        metric_df.columns,
        ("round", "epoch", "step", "federated_round", "global_step", "federated/round"),
    )
    if x_col is None:
        _missing(ax, spec, "no usable convergence x-axis found")
        return
    plot_df = metric_df.copy()
    plot_df["_plot_value"] = (
        _maybe_percent(plot_df[y_col])
        if "accuracy" in str(metric_name).lower() or "f1" in str(metric_name).lower()
        else plot_df[y_col].astype(float)
    )
    hue_col = _first_existing_column(metric_df.columns, ("method", "run_id", "alpha"))
    if hue_col is not None:
        sns.lineplot(data=plot_df, x=x_col, y="_plot_value", hue=hue_col, marker="o", linewidth=2.5, ax=ax)
    else:
        sns.lineplot(data=plot_df, x=x_col, y="_plot_value", marker="o", linewidth=2.5, color=CUSTOM_COLORS[0], ax=ax)
    ax.set_title(f"{spec.title} ({alpha_label})")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Accuracy (%)" if "accuracy" in str(metric_name).lower() or "f1" in str(metric_name).lower() else "Metric Value")
    if "accuracy" in str(metric_name).lower() or "f1" in str(metric_name).lower():
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(alpha=0.25)


def _plot_score_distribution(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["open_set_scores.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"unknown_score", "is_unknown"}.issubset(df.columns):
        _missing(ax, spec, "open_set_scores.csv requires unknown_score,is_unknown")
        return
    plot_df = df[["unknown_score", "is_unknown"]].copy()
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(n=5000, random_state=0)
    plot_df["group"] = np.where(plot_df["is_unknown"].astype(int) == 1, "Unknown", "Known")
    sns.histplot(
        data=plot_df,
        x="unknown_score",
        hue="group",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        ax=ax,
    )
    threshold_path = first_existing(run_dir, ["open_set_metrics.json", "evaluation_metrics.json"])
    threshold_value: float | None = None
    if threshold_path is not None and threshold_path.exists():
        try:
            metrics = json.loads(threshold_path.read_text(encoding="utf-8"))
            raw_threshold = metrics.get("openset_global_delta", metrics.get("open_set/global_delta"))
            if raw_threshold is not None:
                threshold_value = float(raw_threshold)
        except Exception as exc:
            logger.warning("Could not read EVT threshold from %s: %s", threshold_path, exc)
    if threshold_value is not None and np.isfinite(threshold_value):
        ax.axvline(
            threshold_value,
            linestyle="--",
            linewidth=2.0,
            color=CUSTOM_COLORS[5],
            label="Calibrated threshold",
        )
        ax.text(
            threshold_value + 0.01,
            ax.get_ylim()[1] * 0.92,
            "Threshold",
            color=CUSTOM_COLORS[5],
            fontsize=9,
            fontweight="bold",
            va="top",
        )
    ax.legend(frameon=False, title="")
    ax.set_title(spec.title)
    ax.set_xlabel("Unknown Score")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)


def _plot_openness_vs_auroc(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["openness_metrics.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"openness", "auroc"}.issubset(df.columns):
        _missing(ax, spec, "openness_metrics.csv requires openness,auroc")
        return
    plot_df = df.sort_values("openness").copy()
    hue_col = _first_existing_column(plot_df.columns, ("method", "baseline", "configuration"))
    if hue_col is not None:
        sns.lineplot(data=plot_df, x="openness", y="auroc", hue=hue_col, marker="o", linewidth=2.5, ax=ax)
        ax.legend(frameon=False)
    else:
        sns.lineplot(data=plot_df, x="openness", y="auroc", marker="o", linewidth=2.5, color=CUSTOM_COLORS[0], ax=ax)
    ax.set_title(spec.title)
    ax.set_xlabel("Openness (O)")
    ax.set_ylabel("AUROC")
    ax.set_xlim(plot_df["openness"].min() - 0.02, plot_df["openness"].max() + 0.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)


def _plot_roc_curve(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["open_set_roc_curve.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"fpr", "tpr"}.issubset(df.columns):
        _missing(ax, spec, "open_set_roc_curve.csv requires fpr,tpr")
        return
    hue_col = _first_existing_column(df.columns, ("method", "baseline", "configuration"))
    if hue_col is not None:
        sns.lineplot(data=df, x="fpr", y="tpr", hue=hue_col, linewidth=2.5, ax=ax)
        ax.legend(frameon=False)
    else:
        sns.lineplot(data=df, x="fpr", y="tpr", linewidth=2.8, color=CUSTOM_COLORS[0], ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8C8C8C", linewidth=1.3)
    ax.set_title(spec.title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)


def _plot_cross_dataset(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["cross_dataset_metrics.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or "dataset" not in df.columns:
        _missing(ax, spec, "cross_dataset_metrics.csv requires dataset plus metric columns")
        return

    plot_df: pd.DataFrame | None = None
    if {"metric", "metric_value"}.issubset(df.columns):
        plot_df = df[["dataset", "metric", "metric_value"]].copy()
    else:
        numeric_cols = [
            col for col in df.columns if col != "dataset" and pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(numeric_cols) >= 2:
            plot_df = df.melt(
                id_vars=["dataset"],
                value_vars=numeric_cols,
                var_name="metric",
                value_name="metric_value",
            )

    if plot_df is None or plot_df.empty:
        _missing(ax, spec, "cross_dataset_metrics.csv needs metric/value columns")
        return

    plot_df = plot_df.copy()
    plot_df["metric_value"] = _maybe_percent(plot_df["metric_value"])
    metric_labels = {
        "f1": "F1",
        "f1_score": "F1",
        "closed_f1": "Known F1",
        "known_f1": "Known F1",
        "auroc": "AUROC",
        "open_auroc": "Open AUROC",
        "open_unknown_auroc": "Open AUROC",
    }
    plot_df["metric"] = plot_df["metric"].map(lambda value: metric_labels.get(str(value), str(value)))
    sns.barplot(data=plot_df, x="dataset", y="metric_value", hue="metric", ax=ax)
    ax.set_title(spec.title)
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.tick_params(axis="x", rotation=15)
    ax.legend(frameon=False, title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)


def _plot_communication_efficiency(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["communication_metrics.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"cumulative_mb", "accuracy"}.issubset(df.columns):
        _missing(ax, spec, "communication_metrics.csv requires cumulative_mb,accuracy")
        return
    plot_df = df.sort_values("cumulative_mb").copy()
    plot_df["_plot_accuracy"] = _maybe_percent(plot_df["accuracy"])
    hue_col = _first_existing_column(plot_df.columns, ("method", "configuration", "variant"))
    if hue_col is not None:
        sns.lineplot(
            data=plot_df,
            x="cumulative_mb",
            y="_plot_accuracy",
            hue=hue_col,
            marker="o",
            linewidth=2.5,
            ax=ax,
        )
        ax.legend(frameon=False)
    else:
        sns.lineplot(
            data=plot_df,
            x="cumulative_mb",
            y="_plot_accuracy",
            marker="o",
            linewidth=2.5,
            color=CUSTOM_COLORS[0],
            ax=ax,
        )
    ax.set_title(spec.title)
    ax.set_xlabel("Cumulative Data Transmitted (MB)")
    ax.set_ylabel("Global Accuracy (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(alpha=0.25)


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
    plot_df = df.copy()
    if len(plot_df) > 5000:
        stride = int(np.ceil(len(plot_df) / 5000))
        plot_df = plot_df.iloc[::stride].copy()
    if any(token in y.lower() for token in ("accuracy", "f1", "precision", "recall")):
        plot_df["_plot_y"] = _maybe_percent(plot_df[y])
        y_label = f"{y.replace('_', ' ').title()} (%)"
    else:
        plot_df["_plot_y"] = plot_df[y].astype(float)
        y_label = y.replace("_", " ").title()
    if hue is not None and hue in df.columns:
        for label, group in plot_df.groupby(hue):
            ax.plot(group[x].to_numpy(), group["_plot_y"].to_numpy(), marker="o", label=str(label), linewidth=2.2)
        ax.legend(frameon=False)
    else:
        ax.plot(plot_df[x].to_numpy(), plot_df["_plot_y"].to_numpy(), marker="o", color=CUSTOM_COLORS[0], linewidth=2.4)
    ax.set_title(spec.title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y_label)
    if any(token in y.lower() for token in ("accuracy", "f1", "precision", "recall")):
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(alpha=0.25)


def _plot_matrix(ax: plt.Axes, run_dir: Path, spec: PlotSpec, filename: str) -> None:
    path = first_existing(run_dir, [filename])
    if path is None:
        _missing(ax, spec, f"{filename} not found")
        return
    try:
        frame = pd.read_csv(path, index_col=0)
    except Exception as exc:
        _missing(ax, spec, f"failed to load {filename}: {exc}")
        return
    if frame.empty or frame.shape[0] != frame.shape[1]:
        _missing(ax, spec, f"{filename} is not a square labeled matrix")
        return
    matrix = frame.to_numpy(dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=CMAP_SUNSET,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        square=True,
        linewidths=0.8,
        linecolor="white",
        xticklabels=list(frame.columns),
        yticklabels=list(frame.index),
    )
    ax.set_title(spec.title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="x", rotation=30)


def _plot_box(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["seed_robustness.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None:
        _missing(ax, spec, "seed_robustness.csv not found")
        return
    metric_df, y_col, metric_name = _select_metric_frame(df, ("accuracy", "macro_f1", "f1"))
    if metric_df is not None and y_col is not None:
        df = metric_df
    x_col = _first_existing_column(df.columns, ("heterogeneity", "alpha", "setting"))
    if x_col is None or y_col is None:
        _missing(ax, spec, "seed_robustness.csv requires heterogeneity/alpha and accuracy")
        return
    plot_df = df[[x_col, y_col]].dropna().copy()
    scale_to_percent = any(token in str(metric_name or y_col).lower() for token in ("accuracy", "f1"))
    plot_df[y_col] = _maybe_percent(plot_df[y_col]) if scale_to_percent else plot_df[y_col].astype(float)
    sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax, color=CUSTOM_COLORS[0], width=0.6)
    ax.set_title(spec.title)
    ax.set_xlabel("Dirichlet alpha / Heterogeneity")
    ax.set_ylabel("Final Accuracy (%)" if scale_to_percent else y_col)
    if scale_to_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(alpha=0.25, axis="y")


def _plot_latent(ax: plt.Axes, run_dir: Path, spec: PlotSpec) -> None:
    path = first_existing(run_dir, ["latent_embeddings.csv"])
    df = load_csv_if_exists(path) if path else None
    if df is None or not {"x", "y", "label"}.issubset(df.columns):
        _missing(ax, spec, "latent_embeddings.csv requires x,y,label")
        return
    sns.scatterplot(data=df, x="x", y="y", hue="label", s=18, ax=ax, alpha=0.8, edgecolor="none")
    ax.set_title(spec.title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")


def _render_required_plot(
    ax: plt.Axes,
    run_dir: Path,
    spec: PlotSpec,
    *,
    preprocess_dir: Path | None = None,
) -> None:
    if spec.plot_id == "scalability_nodes_vs_accuracy":
        _plot_scalability(ax, run_dir, spec)
    elif spec.plot_id == "non_iid_data_distribution":
        _plot_non_iid(ax, run_dir, spec, preprocess_dir=preprocess_dir)
    elif spec.plot_id == "convergence_mild_non_iid":
        _plot_convergence(ax, run_dir, spec, "alpha=10")
    elif spec.plot_id == "convergence_hard_non_iid":
        _plot_convergence(ax, run_dir, spec, "alpha=0.1")
    elif spec.plot_id == "known_unknown_score_distribution":
        _plot_score_distribution(ax, run_dir, spec)
    elif spec.plot_id == "openness_vs_auroc":
        _plot_openness_vs_auroc(ax, run_dir, spec)
    elif spec.plot_id == "unknown_detection_roc":
        _plot_roc_curve(ax, run_dir, spec)
    elif spec.plot_id == "cross_dataset_generalization":
        _plot_cross_dataset(ax, run_dir, spec)
    elif spec.plot_id == "confusion_matrix_before_osr":
        _plot_matrix(ax, run_dir, spec, "before_osr_confusion_matrix.csv")
    elif spec.plot_id == "confusion_matrix_after_osr":
        _plot_matrix(ax, run_dir, spec, "after_osr_confusion_matrix.csv")
    elif spec.plot_id == "seed_robustness_boxplot":
        _plot_box(ax, run_dir, spec)
    elif spec.plot_id == "latent_space_separation":
        _plot_latent(ax, run_dir, spec)
    elif spec.plot_id == "communication_efficiency":
        _plot_communication_efficiency(ax, run_dir, spec)
    elif spec.plot_id == "architectural_ablation":
        path = first_existing(run_dir, ["ablation_metrics.csv"])
        df = load_csv_if_exists(path) if path else None
        if df is None:
            _missing(ax, spec, "ablation_metrics.csv not found")
        else:
            config_col = _first_existing_column(df.columns, ("configuration", "module_set", "variant"))
            metric_df, metric_col, metric_name = _select_metric_frame(df, ("macro_f1", "accuracy", "f1"))
            if metric_df is not None and metric_col is not None:
                df = metric_df
            if config_col is None or metric_col is None:
                _missing(ax, spec, "ablation_metrics.csv requires configuration and metric columns")
            else:
                plot_df = df[[config_col, metric_col]].dropna().copy()
                scale_to_percent = any(
                    token in str(metric_name or metric_col).lower() for token in ("f1", "accuracy")
                )
                plot_df[metric_col] = (
                    _maybe_percent(plot_df[metric_col]) if scale_to_percent else plot_df[metric_col].astype(float)
                )
                sns.barplot(data=plot_df, x=config_col, y=metric_col, ax=ax, color=CUSTOM_COLORS[0])
                ax.tick_params(axis="x", rotation=20)
                ax.set_title(spec.title)
                ax.set_xlabel("Configuration")
                ax.set_ylabel("Macro F1 (%)" if scale_to_percent else metric_col)
                if scale_to_percent:
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)


def render_required_plots(
    run_dir: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    *,
    preprocess_dir: Path | None = None,
) -> list[Path]:
    apply_theme()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for idx, spec in enumerate(REQUIRED_PLOTS, start=1):
        fig, ax = plt.subplots(figsize=_figure_size(spec.plot_id), dpi=dpi)
        _render_required_plot(ax, run_dir, spec, preprocess_dir=preprocess_dir)
        fig.tight_layout()
        stem = f"{idx:02d}_{spec.plot_id}"
        generated.extend(_save_figure(fig, output_dir, stem, formats))
        plt.close(fig)
    return generated


def render_q1_dashboard(
    run_dir: Path,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    *,
    preprocess_dir: Path | None = None,
) -> list[Path]:
    return render_required_plots(
        run_dir,
        output_dir,
        formats,
        dpi,
        preprocess_dir=preprocess_dir,
    )


def render_training_plots(
    run_dir: Path, output_dir: Path, formats: list[str], dpi: int
) -> list[Path]:
    apply_theme()
    output_dir.mkdir(parents=True, exist_ok=True)
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
        fig, ax = plt.subplots(figsize=TRAINING_FIGURE_SIZE, dpi=dpi)
        for col in present:
            plot_df = df[[x_col, col]].dropna().copy()
            is_percent = any(token in col.lower() for token in ("accuracy", "f1", "precision", "recall"))
            if is_percent:
                plot_df[col] = _maybe_percent(plot_df[col])
            sns.lineplot(data=plot_df, x=x_col, y=col, label=col, ax=ax, linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel(x_col.replace("_", " ").title())
        if any(token in " ".join(present).lower() for token in ("accuracy", "f1", "precision", "recall")):
            ax.set_ylabel("Score (%)")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        else:
            ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        for fmt in formats:
            out = output_dir / f"{filename}.{fmt}"
            fig.savefig(out, bbox_inches="tight", facecolor="white")
            paths.append(out)
        plt.close(fig)
    return paths
