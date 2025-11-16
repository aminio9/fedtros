import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from .evt import EVTModel, load_evt_collection, load_evt_meta, save_evt_collection, save_evt_meta
from .models import OpenSetQChainModelFactory
from .utils import get_device

logger = logging.getLogger("OpenSetEval")

UNKNOWN_LABEL_ID = -1
OPEN_SET_LABEL_ID = 99


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compute_reconstruction_errors(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    value_net_main: torch.nn.Module,
    generation_net: torch.nn.Module,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    loss_fn = nn.MSELoss(reduction="none")
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    per_class_errs: Dict[int, list[np.ndarray]] = {}
    with torch.no_grad():
        prior_net.eval()
        recognition_net.eval()
        value_net_main.eval()
        generation_net.eval()

        for states_s, true_actions in loader:
            states_s = states_s.to(device)
            true_actions = true_actions.to(device)

            mu_p, _ = prior_net(states_s)
            preds = value_net_main(mu_p, states_s).argmax(dim=1)

            a_onehot = F.one_hot(preds, num_classes=value_net_main.num_actions).float()
            mu_q, _ = recognition_net(states_s, a_onehot)
            s_recon = generation_net(mu_q, a_onehot)
            errs = loss_fn(s_recon, states_s).mean(dim=1)

            for cls_id in torch.unique(true_actions).tolist():
                mask = (true_actions == cls_id) & (preds == cls_id)
                if mask.any():
                    per_class_errs.setdefault(int(cls_id), []).append(errs[mask].cpu().numpy())

    return {cls_id: np.concatenate(chunks) for cls_id, chunks in per_class_errs.items() if chunks}


def fit_evt_models(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    evt_cfg,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    value_net_main: torch.nn.Module,
    generation_net: torch.nn.Module,
    device: torch.device,
) -> Dict[int, EVTModel]:
    per_class_errors = _compute_reconstruction_errors(
        features,
        labels,
        batch_size=batch_size,
        prior_net=prior_net,
        recognition_net=recognition_net,
        value_net_main=value_net_main,
        generation_net=generation_net,
        device=device,
    )
    evt_models: Dict[int, EVTModel] = {}
    tail_percent = float(getattr(evt_cfg, "tail_size_percent", 0.1))
    min_errs = int(getattr(evt_cfg, "min_errors_per_class", 5))
    min_tail = int(getattr(evt_cfg, "min_tail_size", 5))
    q = float(getattr(evt_cfg, "q", 0.95))

    for cls_id, errs in per_class_errors.items():
        if len(errs) < min_errs:
            logger.warning("Insufficient training errors for class %d; skipping EVT.", cls_id)
            continue
        threshold = float(np.quantile(errs, q))
        tail = errs[errs > threshold] - threshold
        if tail.size < min_tail:
            logger.warning("Insufficient tail for class %d; relaxing percentile.", cls_id)
            threshold = float(np.quantile(errs, min(0.99, q + 0.02)))
            tail = errs[errs > threshold] - threshold
            if tail.size < min_tail:
                logger.warning("Still insufficient tail for class %d; skipping.", cls_id)
                continue

        evt_model = EVTModel(tail_percent)
        evt_model.threshold_u = threshold
        evt_model.gpd_params = (0.0, 0.0, 1.0)  # temporary placeholder
        evt_model.fit(errs)
        evt_models[cls_id] = evt_model
        logger.info(
            "Fitted EVT for class %d: %d samples, threshold %.6f",
            cls_id,
            len(errs),
            evt_model.threshold_u,
        )

    if not evt_models:
        raise RuntimeError("Failed to fit any EVT models; ensure data and predictions are valid.")
    return evt_models


def calibrate_evt_thresholds(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    evt_models: Dict[int, EVTModel],
    evt_cfg,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    value_net_main: torch.nn.Module,
    generation_net: torch.nn.Module,
    device: torch.device,
) -> Dict:
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.MSELoss(reduction="none")

    pvals_all: list[float] = []
    labels_all: list[int] = []
    preds_all: list[int] = []

    with torch.no_grad():
        for states_s, lbls in loader:
            states_s = states_s.to(device)
            lbls = lbls.to(device)
            mu_p, _ = prior_net(states_s)
            preds = value_net_main(mu_p, states_s).argmax(dim=1)
            one_hot = F.one_hot(preds, num_classes=value_net_main.num_actions).float()
            mu_r, _ = recognition_net(states_s, one_hot)
            recon = generation_net(mu_r, one_hot)
            errs = loss_fn(recon, states_s).mean(dim=1)
            for i in range(states_s.size(0)):
                pred_label = int(preds[i].item())
                model = evt_models.get(pred_label)
                prob_unknown = model.predict_probability_unknown(float(errs[i].item())) if model else 0.0
                pvals_all.append(prob_unknown)
                labels_all.append(int(lbls[i].item()))
                preds_all.append(pred_label)

    pvals_all = np.array(pvals_all)
    labels_all = np.array(labels_all)
    preds_all = np.array(preds_all)
    correct_mask = preds_all == labels_all

    target_fpr = float(getattr(evt_cfg, "target_known_fpr", 0.01))
    delta_grid = getattr(evt_cfg, "delta_grid", [1e-6, 0.99, 200])
    lo, hi, npoints = float(delta_grid[0]), float(delta_grid[1]), int(delta_grid[2])

    p_corr = pvals_all[correct_mask]
    if p_corr.size >= 10:
        qf = 1.0 - target_fpr
        qf = min(max(qf, 0.0), 1.0)
        delta_global = float(np.quantile(p_corr, qf))
        logger.info(
            "Calibrated global EVT threshold δ=%.6g (target known-FPR≈%.3f).",
            delta_global,
            target_fpr,
        )
    else:
        grid = np.logspace(np.log10(lo), np.log10(hi), num=npoints)
        fprs = []
        for d in grid:
            rej = (pvals_all >= d) & correct_mask
            fprs.append(rej.mean())
        ok = [d for d, fpr in zip(grid, fprs) if fpr <= target_fpr]
        delta_global = float(ok[-1] if ok else grid[-1])
        logger.info(
            "Calibrated global EVT threshold δ=%.6g via grid search (known-FPR≤%.3f).",
            delta_global,
            target_fpr,
        )

    per_class_enabled = bool(getattr(evt_cfg, "per_class_thresholds", False))
    delta_per_class: Dict[int, float] = {}
    if per_class_enabled:
        qf = 1.0 - target_fpr
        for cls_id in np.unique(preds_all):
            mask = correct_mask & (preds_all == cls_id)
            if mask.sum() < 10:
                continue
            delta_k = float(np.quantile(pvals_all[mask], qf))
            delta_per_class[int(cls_id)] = delta_k
            logger.info("Per-class EVT threshold δ[%d]=%.6g", cls_id, delta_k)

    meta = {
        "global_delta": delta_global,
        "delta_per_class": delta_per_class,
        "per_class": per_class_enabled and bool(delta_per_class),
    }
    return meta


def evaluate_open_set(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    value_net_main: torch.nn.Module,
    generation_net: torch.nn.Module,
    evt_models: Dict[int, EVTModel],
    evt_meta: Dict,
    class_names: Dict[int, str],
    output_dir: Path,
    device: torch.device,
    report_to_stdout: bool = False,
) -> Dict[str, float]:
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.MSELoss(reduction="none")
    output_dir = _ensure_dir(output_dir)

    delta_global = float(evt_meta.get("global_delta", 0.1))
    delta_per_class = {int(k): float(v) for k, v in evt_meta.get("delta_per_class", {}).items()}
    use_per_class = bool(evt_meta.get("per_class", False) and delta_per_class)

    all_true = []
    final_preds = []
    pre_evt_preds = []
    recon_probs = []
    recon_records = []

    with torch.no_grad():
        for states_s, lbls in loader:
            states_s = states_s.to(device)
            lbls = lbls.to(device)
            mu_p, _ = prior_net(states_s)
            preds = value_net_main(mu_p, states_s).argmax(dim=1)
            pre_evt_preds.extend(preds.cpu().numpy())

            one_hot = F.one_hot(preds, num_classes=value_net_main.num_actions).float()
            mu_q, _ = recognition_net(states_s, one_hot)
            recon = generation_net(mu_q, one_hot)
            errs = loss_fn(recon, states_s).mean(dim=1)

            adjusted_preds = preds.clone()
            for i in range(states_s.size(0)):
                pred_label = int(preds[i].item())
                true_label = int(lbls[i].item())
                model = evt_models.get(pred_label)
                prob = model.predict_probability_unknown(float(errs[i].item())) if model else 0.0
                threshold = (
                    delta_per_class.get(pred_label, delta_global) if use_per_class else delta_global
                )
                if prob >= threshold:
                    adjusted_preds[i] = OPEN_SET_LABEL_ID
                recon_probs.append((prob, true_label == UNKNOWN_LABEL_ID))
                recon_records.append(
                    (
                        true_label,
                        pred_label,
                        float(errs[i].item()),
                    )
                )
            final_preds.extend(adjusted_preds.cpu().numpy())
            all_true.extend(lbls.cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(final_preds)
    y_true[y_true == UNKNOWN_LABEL_ID] = OPEN_SET_LABEL_ID

    all_class_names = [class_names.get(i, f"class_{i}") for i in range(len(class_names))]
    all_class_names.append("Unknown")

    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    logger.info("Open-set evaluation | F1 micro=%.4f | F1 macro=%.4f", f1_micro, f1_macro)

    report = classification_report(
        y_true, y_pred, target_names=all_class_names, digits=4, zero_division=0
    )
    (output_dir / "openset_report.txt").write_text(report)
    if report_to_stdout:
        print("\nClassification Report (All Classes):\n")
        print(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))) + [OPEN_SET_LABEL_ID])
    cm_df = pd.DataFrame(cm, index=all_class_names, columns=all_class_names)
    if report_to_stdout:
        print("\nConfusion Matrix (All Classes):\n")
        print(cm_df)
    _plot_confusion_matrix(cm, all_class_names, output_dir / "fig10_cm_open_set.png")

    if recon_probs:
        _plot_roc_curve(recon_probs, output_dir / "openset_roc.png")

    _save_histograms(
        recon_records,
        class_names,
        output_dir,
    )

    pre_evt_arr = np.array(pre_evt_preds)
    y_true_known_module = y_true.copy()
    cm_known = confusion_matrix(
        y_true_known_module,
        pre_evt_arr,
        labels=list(range(len(class_names))) + [OPEN_SET_LABEL_ID],
    )
    _plot_known_module_confusion(
        cm_known,
        all_class_names,
        list(class_names.get(i, f"class_{i}") for i in range(len(class_names))),
        output_dir / "fig7_cm_known.png",
    )

    known_mask = y_true != OPEN_SET_LABEL_ID
    unknown_mask = ~known_mask
    known_acc = float(np.mean(y_pred[known_mask] == y_true[known_mask])) if known_mask.any() else 1.0
    unknown_recall = (
        float(np.mean(y_pred[unknown_mask] == OPEN_SET_LABEL_ID)) if unknown_mask.any() else 1.0
    )
    overall_acc = float(np.mean(y_pred == y_true))

    return {
        "openset_f1_micro": f1_micro,
        "openset_f1_macro": f1_macro,
        "openset_known_acc": known_acc,
        "openset_unknown_recall": unknown_recall,
        "openset_overall_acc": overall_acc,
    }


def _plot_confusion_matrix(cm: np.ndarray, labels: Iterable[str], path: Path) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(np.nan_to_num(cm_norm), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
                fontsize=9,
            )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Open-set confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_major_formatter(
        mtick.FormatStrFormatter("%.1f")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_known_module_confusion(
    cm: np.ndarray, all_labels: Iterable[str], known_labels: Iterable[str], path: Path
) -> None:
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1.0, a_max=None)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(np.nan_to_num(cm_norm), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(known_labels)))
    ax.set_yticks(np.arange(len(known_labels)))
    ax.set_xticklabels(list(known_labels), rotation=45, ha="right")
    ax.set_yticklabels(list(known_labels))
    for i in range(cm.shape[0]):
        for j in range(min(cm.shape[1], len(known_labels))):
            val = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
                fontsize=9,
            )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix of the known traffic classification module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_major_formatter(
        mtick.FormatStrFormatter("%.1f")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_roc_curve(probs, path: Path) -> None:
    y_true = np.array([int(flag) for _, flag in probs])
    y_score = np.array([prob for prob, _ in probs])
    if y_true.max() == y_true.min():
        logger.warning("Skipping ROC plot because only one class is present.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.plot(fpr, tpr, color="#d62728", linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#4d4d4d", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("FPR (known traffic)")
    ax.set_ylabel("TPR (unknown attacks)")
    ax.set_title("Open-set ROC")
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_histograms(records, class_names: Dict[int, str], output_dir: Path) -> None:
    df = pd.DataFrame(records, columns=["true", "pred_known", "recon_error"])
    df["true"] = df["true"].replace(UNKNOWN_LABEL_ID, OPEN_SET_LABEL_ID)

    inv_class = {v: k for k, v in class_names.items()}

    def _plot_for_class(key: str, filename: str) -> None:
        class_id = inv_class.get(key)
        if class_id is None:
            logger.warning("Class %s not found for histogram plotting.", key)
            return
        known_err = df[df["true"] == class_id]["recon_error"].to_numpy()
        unknown_err = df[(df["true"] == OPEN_SET_LABEL_ID) & (df["pred_known"] == class_id)][
            "recon_error"
        ].to_numpy()
        if known_err.size == 0 and unknown_err.size == 0:
            logger.warning("No histogram data for class %s", key)
            return
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        bins = 40
        max_err = np.percentile(
            np.concatenate([known_err, unknown_err]) if known_err.size + unknown_err.size else np.array([0.0]),
            99.5,
        )
        if known_err.size:
            ax.hist(
                known_err[known_err <= max_err],
                bins=bins,
                color="#4c78a8",
                alpha=0.85,
                edgecolor="white",
                density=True,
                label=f"{key} reconstruction error",
            )
        if unknown_err.size:
            ax.hist(
                unknown_err[unknown_err <= max_err],
                bins=bins,
                color="#f5a623",
                alpha=0.7,
                edgecolor="white",
                density=True,
                label="Unknown attacks",
            )
        ax.set_title(f"Reconstruction error distribution ({key})")
        ax.set_xlabel("Reconstruction error")
        ax.set_ylabel("Relative quantity")
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

    _plot_for_class("Normal", "fig8_hist_normal_vs_unknown.png")
    _plot_for_class("MitM", "fig9_hist_mitm_vs_unknown.png")


def run_evaluate(cfg) -> Dict[str, float]:
    """Standalone evaluation entry-point mirroring the reference OpenSetQ-Chain script."""
    device = get_device()
    logger.info("--- Starting OpenSetQChain Evaluation ---")

    paths_cfg = cfg.paths
    test_data_path = Path(
        to_absolute_path(getattr(paths_cfg, "open_set_test_data", paths_cfg.get("test_data")))
    )
    class_names_path = Path(to_absolute_path(paths_cfg.class_names))
    figure_dir = _ensure_dir(Path(to_absolute_path(paths_cfg.figures_dir)))

    model_factory = OpenSetQChainModelFactory(cfg.model)

    value_stack = model_factory.create_value_network().to(device)
    encoder = value_stack.encoder
    decoder = value_stack.decoder

    prior_net = encoder.prior
    recognition_net = encoder.recognition
    value_net_main = decoder.main_q

    prior_net.load_state_dict(torch.load(to_absolute_path(paths_cfg.prior_network), map_location=device))
    recognition_net.load_state_dict(
        torch.load(to_absolute_path(paths_cfg.recognition_network), map_location=device)
    )
    value_net_main.load_state_dict(
        torch.load(to_absolute_path(paths_cfg.value_network_main), map_location=device)
    )
    decoder.target_q.load_state_dict(value_net_main.state_dict())

    prior_net.eval()
    recognition_net.eval()
    value_net_main.eval()

    generation_net = model_factory.create_generation_network().to(device)
    generation_net.load_state_dict(torch.load(to_absolute_path(paths_cfg.generation_network), map_location=device))
    generation_net.eval()

    evt_dir = Path(to_absolute_path(paths_cfg.evt_dir))
    evt_models = load_evt_collection(evt_dir / "evt_models.pkl")

    evt_meta_path = evt_dir / "evt_meta.json"
    if evt_meta_path.exists():
        try:
            evt_meta = load_evt_meta(evt_meta_path)
        except Exception:  # pragma: no cover
            logger.warning("Failed to load EVT meta; using config threshold.", exc_info=True)
            evt_meta = {"global_delta": float(getattr(cfg.evt, "decision_threshold", 0.1))}
    else:
        evt_meta = {"global_delta": float(getattr(cfg.evt, "decision_threshold", 0.1))}

    data = torch.load(test_data_path)
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    metrics = evaluate_open_set(
        features=data["features"].float(),
        labels=data["labels"].long(),
        batch_size=int(cfg.training.batch_size),
        prior_net=prior_net,
        recognition_net=recognition_net,
        value_net_main=value_net_main,
        generation_net=generation_net,
        evt_models=evt_models,
        evt_meta=evt_meta,
        class_names=class_map,
        output_dir=figure_dir,
        device=device,
        report_to_stdout=True,
    )
    logger.info(
        "Summary: Known acc=%.4f | Unknown recall=%.4f | Overall=%.4f",
        metrics["openset_known_acc"],
        metrics["openset_unknown_recall"],
        metrics["openset_overall_acc"],
    )
    return metrics
