import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from src.openset.evt import EVTModel
from src.utils.utils import to_one_hot

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
) -> dict[int, np.ndarray]:
    loss_fn = nn.MSELoss(reduction="none")
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    per_class_errs: dict[int, list[np.ndarray]] = {}

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
) -> dict[int, EVTModel]:

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

    evt_models: dict[int, EVTModel] = {}

    # Load config
    tail_percent = float(evt_cfg.tail_size_percent)
    min_errs = int(evt_cfg.min_errors_per_class)
    min_tail = int(evt_cfg.min_tail_size)

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
            cls_id,
            tail.size,
            q,
            threshold,
        )

    if not evt_models:
        raise RuntimeError("Failed to fit any EVT models.")

    return evt_models


def calibrate_evt_thresholds(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    evt_models: dict[int, EVTModel],
    evt_cfg,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    value_net_main: torch.nn.Module,
    generation_net: torch.nn.Module,
    device: torch.device,
) -> dict:
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.MSELoss(reduction="none")

    probs_unknown = []
    with torch.no_grad():
        prior_net.eval()
        recognition_net.eval()
        value_net_main.eval()
        generation_net.eval()

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
    target_fpr = float(evt_cfg.target_known_fpr)

    if probs_unknown.size > 0:
        qf = 1.0 - target_fpr
        qf = min(max(qf, 0.0), 1.0)
        delta_global = float(np.quantile(probs_unknown, qf))
    else:
        delta_global = 0.5

    logger.info(
        "Calibrated global EVT threshold delta=%.6f (target known-FPR~=%.3f).",
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
    evt_models: dict[int, EVTModel],
    evt_meta: dict,
    class_names: dict[int, str],
    output_dir: Path,
    device: torch.device,
    report_to_stdout: bool = False,
) -> dict[str, float]:
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.MSELoss(reduction="none")
    output_dir = _ensure_dir(output_dir)

    delta_global = float(evt_meta.get("global_delta", 0.1))
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    y_score_unknown: list[float] = []
    missing_evt_model_count = 0

    with torch.no_grad():
        prior_net.eval()
        recognition_net.eval()
        value_net_main.eval()
        generation_net.eval()

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

                if model is None:
                    # A predicted class without a calibrated EVT tail is outside
                    # the fitted support and should not be silently accepted as known.
                    prob = 1.0
                    missing_evt_model_count += 1
                else:
                    prob = model.predict_probability_unknown(float(errs[i].item()))
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

    if np.unique(y_binary).size < 2:
        auroc = 0.0
        auprc = 0.0
        fpr95 = 1.0
    else:
        try:
            auroc = float(roc_auc_score(y_binary, y_scores))
        except ValueError:
            auroc = 0.0
        if not np.isfinite(auroc):
            auroc = 0.0
        try:
            auprc = float(average_precision_score(y_binary, y_scores))
        except ValueError:
            auprc = 0.0
        if not np.isfinite(auprc):
            auprc = 0.0
        try:
            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            valid = np.where(tpr >= 0.95)[0]
            fpr95 = float(fpr[valid[0]]) if valid.size else 1.0
        except ValueError:
            fpr95 = 1.0

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    unknown_f1 = f1_score(y_binary, (y_pred == OPEN_SET_LABEL_ID).astype(int), zero_division=0)
    known_mask = y_true != OPEN_SET_LABEL_ID
    unknown_mask = ~known_mask

    known_acc = accuracy_score(y_true[known_mask], y_pred[known_mask]) if known_mask.any() else 0.0
    unknown_recall = (
        accuracy_score(y_true[unknown_mask], y_pred[unknown_mask]) if unknown_mask.any() else 0.0
    )
    overall_acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0

    observed_known_ids = set(y_true[y_true != OPEN_SET_LABEL_ID].tolist())
    observed_known_ids.update(y_pred[y_pred != OPEN_SET_LABEL_ID].tolist())
    observed_known_ids.update(class_names.keys())
    known_label_ids = sorted(int(k) for k in observed_known_ids)
    class_name_list = [class_names.get(k, f"class_{k}") for k in known_label_ids]
    class_name_list.append("Unknown")
    report_labels = [*known_label_ids, OPEN_SET_LABEL_ID]

    report = classification_report(
        y_true,
        y_pred,
        labels=report_labels,
        target_names=class_name_list,
        digits=4,
        zero_division=0,
    )
    (output_dir / "openset_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(
        output_dir / "open_set_scores.csv",
        np.column_stack([y_true, y_pred, y_scores, y_binary]),
        delimiter=",",
        header="y_true,y_pred,unknown_score,is_unknown",
        comments="",
    )
    if np.unique(y_binary).size >= 2:
        fpr, tpr, roc_thresholds = roc_curve(y_binary, y_scores)
        np.savetxt(
            output_dir / "open_set_roc_curve.csv",
            np.column_stack([fpr, tpr, roc_thresholds]),
            delimiter=",",
            header="fpr,tpr,threshold",
            comments="",
        )
        precision, recall, pr_thresholds = precision_recall_curve(y_binary, y_scores)
        padded_thresholds = np.concatenate([pr_thresholds, [np.nan]])
        np.savetxt(
            output_dir / "open_set_pr_curve.csv",
            np.column_stack([precision, recall, padded_thresholds]),
            delimiter=",",
            header="precision,recall,threshold",
            comments="",
        )

    logger.info(
        "Open-set metrics | AUROC=%.4f | AUPRC=%.4f | FPR95=%.4f | F1_macro=%.4f | Known_Acc=%.4f | Unknown_Recall=%.4f | Overall_Acc=%.4f",
        auroc,
        auprc,
        fpr95,
        f1_macro,
        known_acc,
        unknown_recall,
        overall_acc,
    )

    if report_to_stdout:
        print("\nOpen-Set Classification Report:\n")
        print(report)

    metrics = {
        "openset_f1_macro": f1_macro,
        "openset_auroc": auroc,
        "openset_auprc": auprc,
        "openset_fpr95": fpr95,
        "openset_unknown_f1": float(unknown_f1),
        "openset_known_acc": known_acc,
        "openset_unknown_recall": unknown_recall,
        "openset_overall_acc": overall_acc,
        "openset_missing_evt_model_count": float(missing_evt_model_count),
        "open_set/auroc": auroc,
        "open_set/auprc": auprc,
        "open_set/fpr95": fpr95,
        "open_set/unknown_detection_rate": unknown_recall,
        "open_set/unknown_f1": float(unknown_f1),
    }
    (output_dir / "open_set_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics
