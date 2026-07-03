import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from src.openset.evt import EVTModel, resolve_tail_fraction
from src.utils.utils import to_one_hot

logger = logging.getLogger("OpenSetEval")

UNKNOWN_LABEL_ID = -1
OPEN_SET_LABEL_ID = 99

# CRITICAL: Scaling factor to prevent precision collapse on tiny errors
ERROR_SCALE_FACTOR = 100000.0


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cfg_value(cfg, key: str, default):
    return getattr(cfg, key, default) if cfg is not None else default


def _error_scale_factor(evt_cfg) -> float:
    return float(_cfg_value(evt_cfg, "error_scale_factor", ERROR_SCALE_FACTOR))


def _threshold_mode(evt_cfg) -> str:
    return str(_cfg_value(evt_cfg, "threshold_mode", "validation_known_fpr")).lower()


def _known_only(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    unknown_label_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    known_mask = labels.long() != int(unknown_label_id)
    dropped = int((~known_mask).sum().item())
    if dropped:
        return features[known_mask], labels[known_mask], dropped
    return features, labels, 0


def _open_set_label_order(
    class_names: dict[int, str], open_set_label_id: int
) -> tuple[list[int], list[str]]:
    known_ids = sorted(int(k) for k in class_names)
    return known_ids + [open_set_label_id], [class_names[k] for k in known_ids] + ["Unknown"]


def _save_labeled_confusion_matrix(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_ids: list[int],
    label_names: list[str],
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    frame = pd.DataFrame(matrix, index=label_names, columns=label_names)
    frame.to_csv(path)
    return frame


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    y_values = np.asarray(y, dtype=float)
    x_values = np.asarray(x, dtype=float)
    if y_values.shape != x_values.shape:
        raise ValueError("x and y must have the same shape for trapezoid integration.")
    if y_values.size < 2:
        return 0.0
    widths = x_values[1:] - x_values[:-1]
    heights = (y_values[1:] + y_values[:-1]) * 0.5
    return float(np.sum(widths * heights))


def _compute_oscr_curve(
    y_true: np.ndarray,
    y_raw_pred: np.ndarray,
    y_scores: np.ndarray,
    *,
    open_set_label_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute OSCR-style CCR-vs-FPR curve for unknown-score thresholding.

    Higher ``y_scores`` means more unknown. A sample is accepted as known when
    its score is at or below the threshold.
    """
    known_mask = y_true != open_set_label_id
    unknown_mask = ~known_mask
    known_count = int(known_mask.sum())
    unknown_count = int(unknown_mask.sum())
    if known_count == 0 or unknown_count == 0 or y_scores.size == 0:
        thresholds = np.array([-np.inf, np.inf], dtype=float)
        zeros = np.array([0.0, 0.0], dtype=float)
        return np.array([0.0, 1.0], dtype=float), zeros, thresholds, 0.0

    unique_scores = np.unique(y_scores.astype(float))
    thresholds = np.concatenate(([-np.inf], unique_scores, [np.inf])).astype(float)
    fpr_values: list[float] = []
    ccr_values: list[float] = []
    for threshold in thresholds:
        accepted_as_known = y_scores <= threshold
        false_known_accepts = unknown_mask & accepted_as_known
        correct_known_accepts = known_mask & (y_raw_pred == y_true) & accepted_as_known
        fpr_values.append(float(false_known_accepts.sum() / unknown_count))
        ccr_values.append(float(correct_known_accepts.sum() / known_count))

    fpr = np.asarray(fpr_values, dtype=float)
    ccr = np.asarray(ccr_values, dtype=float)
    order = np.argsort(fpr, kind="stable")
    fpr = fpr[order]
    ccr = ccr[order]
    thresholds = thresholds[order]

    # Keep the best CCR for repeated FPR values to avoid staircase artifacts.
    dedup_fpr: list[float] = []
    dedup_ccr: list[float] = []
    dedup_thresholds: list[float] = []
    for current_fpr in np.unique(fpr):
        mask = fpr == current_fpr
        best_idx = int(np.argmax(ccr[mask]))
        original_indices = np.flatnonzero(mask)
        selected_idx = int(original_indices[best_idx])
        dedup_fpr.append(float(current_fpr))
        dedup_ccr.append(float(ccr[selected_idx]))
        dedup_thresholds.append(float(thresholds[selected_idx]))

    fpr_out = np.asarray(dedup_fpr, dtype=float)
    ccr_out = np.asarray(dedup_ccr, dtype=float)
    threshold_out = np.asarray(dedup_thresholds, dtype=float)
    auoscr = _trapezoid_integral(ccr_out, fpr_out)
    return fpr_out, ccr_out, threshold_out, auoscr


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
    error_scale_factor: float,
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

            errs = loss_fn(s_recon, states_s).mean(dim=1) * error_scale_factor

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
    logger: logging.Logger | None = None,
) -> dict[int, EVTModel]:
    active_logger = logger or logging.getLogger("OpenSetEval")
    error_scale = _error_scale_factor(evt_cfg)
    unknown_label_id = int(_cfg_value(evt_cfg, "unknown_label_id", UNKNOWN_LABEL_ID))
    features, labels, dropped_unknown = _known_only(
        features,
        labels,
        unknown_label_id=unknown_label_id,
    )
    if dropped_unknown:
        active_logger.warning(
            "Dropped %d unknown-labeled samples before EVT fitting; EVT tails must use known data.",
            dropped_unknown,
        )

    per_class_errors = _compute_reconstruction_errors(
        features,
        labels,
        batch_size=batch_size,
        prior_net=prior_net,
        recognition_net=recognition_net,
        value_net_main=value_net_main,
        generation_net=generation_net,
        device=device,
        error_scale_factor=error_scale,
    )

    evt_models: dict[int, EVTModel] = {}

    tail_fraction, tail_source = resolve_tail_fraction(evt_cfg)
    min_errs = int(evt_cfg.min_errors_per_class)
    min_tail = int(evt_cfg.min_tail_size)

    q = max(0.0, 1.0 - tail_fraction)

    for cls_id, errs in per_class_errors.items():
        if len(errs) < min_errs:
            active_logger.warning("Insufficient training errors for class %d; skipping EVT.", cls_id)
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
            active_logger.warning("Insufficient tail for class %d; relaxing percentile.", cls_id)
            new_q = max(0.0, q - 0.2)
            threshold = float(np.quantile(errs, new_q))
            if new_q == 0.0:
                threshold -= 1e-9

            tail = errs[errs > threshold] - threshold

            if tail.size < min_tail:
                active_logger.warning("Still insufficient tail for class %d; skipping.", cls_id)
                continue

        # Fit EVT with the FIXED threshold
        evt_model = EVTModel(tail_fraction=tail_fraction)
        evt_model.fit(errs, fixed_threshold=threshold, logger=active_logger)

        evt_models[cls_id] = evt_model
        active_logger.info(
            "Fitted EVT for class %d: %d tail samples (tail_fraction=%.4f via %s, q=%.4f), threshold %.6f",
            cls_id,
            tail.size,
            tail_fraction,
            tail_source,
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
    logger: logging.Logger | None = None,
) -> dict:
    active_logger = logger or logging.getLogger("OpenSetEval")
    error_scale = _error_scale_factor(evt_cfg)
    unknown_label_id = int(_cfg_value(evt_cfg, "unknown_label_id", UNKNOWN_LABEL_ID))
    features, labels, dropped_unknown = _known_only(
        features,
        labels,
        unknown_label_id=unknown_label_id,
    )
    if dropped_unknown:
        active_logger.warning(
            "Dropped %d unknown-labeled samples before EVT threshold calibration; "
            "validation-known-FPR calibration must use known validation data.",
            dropped_unknown,
        )
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

            errs = loss_fn(recon, states_s).mean(dim=1) * error_scale

            for i in range(states_s.size(0)):
                pred_label = int(preds[i].item())
                true_label = int(lbls[i].item())
                if pred_label == true_label and pred_label in evt_models:
                    model = evt_models[pred_label]
                    prob = model.predict_probability_unknown(float(errs[i].item()))
                    probs_unknown.append(prob)

    probs_unknown = np.array(probs_unknown)
    target_fpr = float(evt_cfg.target_known_fpr)
    mode = _threshold_mode(evt_cfg)
    if mode == "fixed":
        delta_global = float(_cfg_value(evt_cfg, "fixed_threshold", _cfg_value(evt_cfg, "decision_threshold", 0.5)))
    elif probs_unknown.size > 0:
        qf = 1.0 - target_fpr
        qf = min(max(qf, 0.0), 1.0)
        delta_global = float(np.quantile(probs_unknown, qf))
    elif mode in {"validation_known_fpr", "known_fpr", "validation"}:
        delta_global = float(_cfg_value(evt_cfg, "decision_threshold", 0.5))
    else:
        raise ValueError(
            "open_set.evt.threshold_mode must be 'validation_known_fpr' or 'fixed'."
        )

    active_logger.info(
        "Calibrated global EVT threshold delta=%.6f (mode=%s, target known-FPR~=%.3f).",
        delta_global,
        mode,
        target_fpr,
    )

    tail_fraction, tail_source = resolve_tail_fraction(evt_cfg)
    return {
        "global_delta": delta_global,
        "threshold_mode": mode,
        "calibration_protocol": str(_cfg_value(evt_cfg, "calibration_protocol", "validation_only")),
        "target_known_fpr": target_fpr,
        "target_unknown_tpr": _cfg_value(evt_cfg, "target_unknown_tpr", None),
        "decision_threshold": float(_cfg_value(evt_cfg, "decision_threshold", delta_global)),
        "score_direction": str(_cfg_value(evt_cfg, "score_direction", "higher_unknown")),
        "tail_fraction": tail_fraction,
        "tail_source": tail_source,
        "error_scale_factor": error_scale,
        "unknown_label_id": int(_cfg_value(evt_cfg, "unknown_label_id", UNKNOWN_LABEL_ID)),
        "open_set_label_id": int(_cfg_value(evt_cfg, "open_set_label_id", OPEN_SET_LABEL_ID)),
    }


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
    evt_cfg=None,
    report_to_stdout: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    active_logger = logger or logging.getLogger("OpenSetEval")
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.MSELoss(reduction="none")
    output_dir = _ensure_dir(output_dir)

    decision_threshold = float(_cfg_value(evt_cfg, "decision_threshold", 0.1))
    delta_global = float(evt_meta.get("global_delta", decision_threshold))
    error_scale = float(evt_meta.get("error_scale_factor", _error_scale_factor(evt_cfg)))
    unknown_label_id = int(
        evt_meta.get(
            "unknown_label_id",
            _cfg_value(evt_cfg, "unknown_label_id", UNKNOWN_LABEL_ID),
        )
    )
    open_set_label_id = int(
        evt_meta.get(
            "open_set_label_id",
            _cfg_value(evt_cfg, "open_set_label_id", OPEN_SET_LABEL_ID),
        )
    )
    y_true_list: list[int] = []
    raw_pred_list: list[int] = []
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

            errs = loss_fn(recon, states_s).mean(dim=1) * error_scale

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
                raw_pred_list.append(pred_label)

                final_pred = pred_label
                if prob > delta_global:
                    final_pred = open_set_label_id

                mapped_true = open_set_label_id if true_label == unknown_label_id else true_label
                y_true_list.append(mapped_true)
                y_pred_list.append(final_pred)

    y_true = np.array(y_true_list, dtype=int)
    y_raw_pred = np.array(raw_pred_list, dtype=int)
    y_pred = np.array(y_pred_list, dtype=int)
    y_scores = np.array(y_score_unknown, dtype=float)
    y_binary = (y_true == open_set_label_id).astype(int)

    if np.unique(y_binary).size < 2:
        auroc = 0.0
        aupr_out = 0.0
        aupr_in = 0.0
        fpr95 = 1.0
    else:
        try:
            auroc = float(roc_auc_score(y_binary, y_scores))
        except ValueError:
            auroc = 0.0
        if not np.isfinite(auroc):
            auroc = 0.0
        try:
            aupr_out = float(average_precision_score(y_binary, y_scores))
        except ValueError:
            aupr_out = 0.0
        if not np.isfinite(aupr_out):
            aupr_out = 0.0
        try:
            aupr_in = float(average_precision_score(1 - y_binary, -y_scores))
        except ValueError:
            aupr_in = 0.0
        if not np.isfinite(aupr_in):
            aupr_in = 0.0
        try:
            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            valid = np.where(tpr >= 0.95)[0]
            fpr95 = float(fpr[valid[0]]) if valid.size else 1.0
        except ValueError:
            fpr95 = 1.0
    auprc = aupr_out

    oscr_fpr, oscr_ccr, oscr_thresholds, auoscr = _compute_oscr_curve(
        y_true,
        y_raw_pred,
        y_scores,
        open_set_label_id=open_set_label_id,
    )

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    y_pred_unknown = (y_pred == open_set_label_id).astype(int)
    unknown_f1 = f1_score(y_binary, y_pred_unknown, zero_division=0)
    unknown_tp = int(((y_binary == 1) & (y_pred_unknown == 1)).sum())
    unknown_fp = int(((y_binary == 0) & (y_pred_unknown == 1)).sum())
    unknown_precision = unknown_tp / max(unknown_tp + unknown_fp, 1)
    known_mask = y_true != open_set_label_id
    unknown_mask = ~known_mask

    known_acc = accuracy_score(y_true[known_mask], y_pred[known_mask]) if known_mask.any() else 0.0
    known_rejection_rate = (
        float((y_pred[known_mask] == open_set_label_id).mean()) if known_mask.any() else 0.0
    )
    unknown_recall = (
        accuracy_score(y_true[unknown_mask], y_pred[unknown_mask]) if unknown_mask.any() else 0.0
    )
    overall_acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0

    report_labels, class_name_list = _open_set_label_order(class_names, open_set_label_id)

    before_cm_path = output_dir / "before_osr_confusion_matrix.csv"
    after_cm_path = output_dir / "after_osr_confusion_matrix.csv"
    _save_labeled_confusion_matrix(
        before_cm_path,
        y_true,
        y_raw_pred,
        label_ids=report_labels,
        label_names=class_name_list,
    )
    _save_labeled_confusion_matrix(
        after_cm_path,
        y_true,
        y_pred,
        label_ids=report_labels,
        label_names=class_name_list,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=report_labels,
        target_names=class_name_list,
        digits=4,
        zero_division=0,
    )
    (output_dir / "openset_report.txt").write_text(report, encoding="utf-8")
    scores_df = pd.DataFrame(
        {
            "y_true": y_true,
            "raw_pred": y_raw_pred,
            "y_pred": y_pred,
            "unknown_score": y_scores,
            "is_unknown": y_binary,
        }
    )
    scores_df.to_csv(output_dir / "open_set_scores.csv", index=False)
    if np.unique(y_binary).size >= 2:
        fpr, tpr, roc_thresholds = roc_curve(y_binary, y_scores)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
            output_dir / "open_set_roc_curve.csv",
            index=False,
        )
        precision, recall, pr_thresholds = precision_recall_curve(y_binary, y_scores)
        padded_thresholds = np.concatenate([pr_thresholds, [np.nan]])
        pd.DataFrame(
            {"precision": precision, "recall": recall, "threshold": padded_thresholds}
        ).to_csv(output_dir / "open_set_pr_curve.csv", index=False)
        sensitivity_rows = []
        for threshold in np.unique(np.quantile(y_scores, np.linspace(0.0, 1.0, 51))):
            pred_unknown = (y_scores > threshold).astype(int)
            tp = int(((y_binary == 1) & (pred_unknown == 1)).sum())
            fp = int(((y_binary == 0) & (pred_unknown == 1)).sum())
            fn = int(((y_binary == 1) & (pred_unknown == 0)).sum())
            sensitivity_rows.append(
                {
                    "threshold": float(threshold),
                    "unknown_precision": float(tp / max(tp + fp, 1)),
                    "unknown_recall": float(tp / max(tp + fn, 1)),
                    "unknown_f1": float(f1_score(y_binary, pred_unknown, zero_division=0)),
                    "known_rejection_rate": float(
                        fp / max(int((y_binary == 0).sum()), 1)
                    ),
                }
            )
        pd.DataFrame(sensitivity_rows).to_csv(
            output_dir / "open_set_threshold_sensitivity.csv",
            index=False,
        )
    pd.DataFrame(
        {"fpr": oscr_fpr, "ccr": oscr_ccr, "threshold": oscr_thresholds}
    ).to_csv(output_dir / "open_set_oscr_curve.csv", index=False)

    active_logger.info(
        "Open-set metrics | AUROC=%.4f | AUPR-Out=%.4f | AUOSCR=%.4f | "
        "FPR95=%.4f | F1_macro=%.4f | Known_Acc=%.4f | Unknown_Recall=%.4f | "
        "Overall_Acc=%.4f",
        auroc,
        aupr_out,
        auoscr,
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
        "openset_aupr_out": aupr_out,
        "openset_aupr_in": aupr_in,
        "openset_auoscr": auoscr,
        "openset_fpr95": fpr95,
        "openset_unknown_f1": float(unknown_f1),
        "openset_unknown_precision": float(unknown_precision),
        "openset_known_acc": known_acc,
        "openset_known_rejection_rate": known_rejection_rate,
        "openset_unknown_recall": unknown_recall,
        "openset_overall_acc": overall_acc,
        "openset_missing_evt_model_count": float(missing_evt_model_count),
        "openset_global_delta": delta_global,
        "openset_error_scale_factor": error_scale,
        "open_set/auroc": auroc,
        "open_set/auprc": auprc,
        "open_set/aupr_out": aupr_out,
        "open_set/aupr_in": aupr_in,
        "open_set/auoscr": auoscr,
        "open_set/fpr95": fpr95,
        "open_set/unknown_detection_rate": unknown_recall,
        "open_set/unknown_f1": float(unknown_f1),
        "open_set/unknown_precision": float(unknown_precision),
        "open_set/known_accuracy_after_rejection": known_acc,
        "open_set/known_rejection_rate": known_rejection_rate,
        "open_set/global_delta": delta_global,
        "open_set/error_scale_factor": error_scale,
    }
    (output_dir / "open_set_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics
