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
from sklearn.metrics import (
    auc,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

# Adjust imports based on your folder structure
try:
    from .evt import EVTModel, load_evt_collection, load_evt_meta, save_evt_collection, save_evt_meta
    from .models import OpenSetQChainModelFactory
    from .utils import resolve_device_from_config, to_one_hot
except ImportError:
    from evt import EVTModel, load_evt_collection, load_evt_meta, save_evt_collection, save_evt_meta
    from models import OpenSetQChainModelFactory
    from utils import resolve_device_from_config, to_one_hot

logger = logging.getLogger("OpenSetEval")

UNKNOWN_LABEL_ID = -1
OPEN_SET_LABEL_ID = 99

# CRITICAL: Scaling factor to prevent precision collapse on tiny errors
ERROR_SCALE_FACTOR = 100000.0


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

            a_onehot = to_one_hot(preds, value_net_main.num_actions)
            mu_q, _ = recognition_net(states_s, a_onehot)
            s_recon = generation_net(mu_q, a_onehot)
            
            # --- APPLY SCALING HERE ---
            errs = loss_fn(s_recon, states_s).mean(dim=1) * ERROR_SCALE_FACTOR

            for cls_id in torch.unique(true_actions).tolist():
                mask = (true_actions == cls_id) & (preds == cls_id)
                if mask.any():
                    per_class_errs.setdefault(int(cls_id), []).append(
                        errs[mask].cpu().numpy()
                    )

    return {
        cls_id: np.concatenate(chunks)
        for cls_id, chunks in per_class_errs.items()
        if chunks
    }


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

    # Load config
    tail_percent = float(getattr(evt_cfg, "tail_size_percent", 1.0))
    min_errs = int(getattr(evt_cfg, "min_errors_per_class", 10))
    min_tail = int(getattr(evt_cfg, "min_tail_size", 10))

    # CRITICAL: Calculate q directly from tail_percent.
    # If tail_percent is 1.0, q becomes 0.0 (use all data).
    q = max(0.0, 1.0 - tail_percent)

    for cls_id, errs in per_class_errors.items():
        if len(errs) < min_errs:
            logger.warning("Insufficient training errors for class %d; skipping EVT.", cls_id)
            continue

        # Determine Threshold
        threshold = float(np.quantile(errs, q))

        # IMPORTANT: If using 100% tail (q=0), threshold is min(errs).
        # We subtract epsilon to ensure the min value is included in 'tail' > threshold
        if q == 0.0:
            threshold -= 1e-9

        tail = errs[errs > threshold] - threshold

        # Fallback: If tail is somehow too small, relax q
        if tail.size < min_tail:
            logger.warning("Insufficient tail for class %d; relaxing percentile.", cls_id)
            new_q = max(0.0, q - 0.2)
            threshold = float(np.quantile(errs, new_q))
            if new_q == 0.0:
                threshold -= 1e-9

            tail = errs[errs > threshold] - threshold

            if tail.size < min_tail:
                logger.warning("Still insufficient tail for class %d; skipping.", cls_id)
                continue

        # Fit EVT with the FIXED threshold
        evt_model = EVTModel(tail_percent)
        evt_model.fit(errs, fixed_threshold=threshold)

        evt_models[cls_id] = evt_model
        logger.info(
            "Fitted EVT for class %d: %d tail samples (q=%.2f), threshold %.6f",
            cls_id, tail.size, q, threshold
        )

    if not evt_models:
        raise RuntimeError("Failed to fit any EVT models.")

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

    probs_unknown = []
    with torch.no_grad():
        for states_s, lbls in loader:
            states_s = states_s.to(device)
            lbls = lbls.to(device)
            mu_p, _ = prior_net(states_s)
            preds = value_net_main(mu_p, states_s).argmax(dim=1)
            one_hot = to_one_hot(preds, value_net_main.num_actions)
            mu_r, _ = recognition_net(states_s, one_hot)
            recon = generation_net(mu_r, one_hot)
            
            # --- APPLY SCALING HERE ---
            errs = loss_fn(recon, states_s).mean(dim=1) * ERROR_SCALE_FACTOR
            
            for i in range(states_s.size(0)):
                pred_label = int(preds[i].item())
                true_label = int(lbls[i].item())
                if pred_label == true_label and pred_label in evt_models:
                    model = evt_models[pred_label]
                    prob = model.predict_probability_unknown(float(errs[i].item()))
                    probs_unknown.append(prob)

    probs_unknown = np.array(probs_unknown)
    target_fpr = float(getattr(evt_cfg, "target_known_fpr", 0.05))
    
    if probs_unknown.size > 0:
        qf = 1.0 - target_fpr
        qf = min(max(qf, 0.0), 1.0)
        delta_global = float(np.quantile(probs_unknown, qf))
    else:
        delta_global = 0.5

    logger.info(
        "Calibrated global EVT threshold δ=%.6f (target known-FPR≈%.3f).",
        delta_global,
        target_fpr,
    )

    return {"global_delta": delta_global}


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
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    y_score_unknown: list[float] = []

    with torch.no_grad():
        for states_s, lbls in loader:
            states_s = states_s.to(device)
            lbls = lbls.to(device)
            mu_p, _ = prior_net(states_s)
            preds = value_net_main(mu_p, states_s).argmax(dim=1)

            one_hot = to_one_hot(preds, value_net_main.num_actions)
            mu_q, _ = recognition_net(states_s, one_hot)
            recon = generation_net(mu_q, one_hot)
            
            # --- APPLY SCALING HERE ---
            errs = loss_fn(recon, states_s).mean(dim=1) * ERROR_SCALE_FACTOR

            for i in range(states_s.size(0)):
                pred_label = int(preds[i].item())
                true_label = int(lbls[i].item())
                model = evt_models.get(pred_label)
                
                prob = model.predict_probability_unknown(float(errs[i].item())) if model else 0.0
                y_score_unknown.append(prob)

                final_pred = pred_label
                if prob >= delta_global:
                    final_pred = OPEN_SET_LABEL_ID

                mapped_true = OPEN_SET_LABEL_ID if true_label == UNKNOWN_LABEL_ID else true_label
                y_true_list.append(mapped_true)
                y_pred_list.append(final_pred)

    y_true = np.array(y_true_list, dtype=int)
    y_pred = np.array(y_pred_list, dtype=int)
    y_scores = np.array(y_score_unknown, dtype=float)
    y_binary = (y_true == OPEN_SET_LABEL_ID).astype(int)

    try:
        auroc = float(roc_auc_score(y_binary, y_scores))
    except ValueError:
        auroc = 0.0

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    known_mask = y_true != OPEN_SET_LABEL_ID
    unknown_mask = ~known_mask

    known_acc = accuracy_score(y_true[known_mask], y_pred[known_mask]) if known_mask.any() else 0.0
    unknown_recall = accuracy_score(y_true[unknown_mask], y_pred[unknown_mask]) if unknown_mask.any() else 0.0
    overall_acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0

    known_label_ids = sorted(class_names.keys())
    class_name_list = [class_names.get(k, f"class_{k}") for k in known_label_ids]
    class_name_list.append("Unknown")
    report_labels = known_label_ids + [OPEN_SET_LABEL_ID]

    report = classification_report(
        y_true,
        y_pred,
        labels=report_labels,
        target_names=class_name_list,
        digits=4,
        zero_division=0,
    )
    (output_dir / "openset_report.txt").write_text(report)

    logger.info(
        "Open-set metrics | AUROC=%.4f | F1_macro=%.4f | Known_Acc=%.4f | Unknown_Recall=%.4f | Overall_Acc=%.4f",
        auroc, f1_macro, known_acc, unknown_recall, overall_acc,
    )
    
    if report_to_stdout:
        print("\nOpen-Set Classification Report:\n")
        print(report)

    return {
        "openset_f1_macro": f1_macro,
        "openset_auroc": auroc,
        "openset_known_acc": known_acc,
        "openset_unknown_recall": unknown_recall,
        "openset_overall_acc": overall_acc,
    }