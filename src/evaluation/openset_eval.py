from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

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

from src.openset.evt import EVTModel
from src.utils.utils import to_one_hot

logger = logging.getLogger("OpenSetEval")

UNKNOWN_LABEL_ID = -1
OPEN_SET_LABEL_ID = 99
ERROR_SCALE_FACTOR = 100000.0

BackendName = Literal["teacher_generator"]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cfg_value(cfg, key: str, default):
    return getattr(cfg, key, default) if cfg is not None else default


def _error_scale_factor(evt_cfg) -> float:
    return float(_cfg_value(evt_cfg, "error_scale_factor", ERROR_SCALE_FACTOR))


def _evt_backend(evt_cfg) -> BackendName:
    # Legacy evaluator supports only the local teacher/generator reconstruction
    # backend. DKD-FedOS open-set uses src.openset.feature_evt instead.
    return "teacher_generator"

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


def _compute_teacher_reconstruction_errors(
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
    dataset = TensorDataset(features.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_class_errs: dict[int, list[np.ndarray]] = {}

    with torch.no_grad():
        prior_net.eval()
        recognition_net.eval()
        value_net_main.eval()
        generation_net.eval()

        for states_s, true_actions in loader:
            states_s = states_s.to(device).float()
            true_actions = true_actions.to(device).long()
            known_mask = (true_actions >= 0) & (true_actions < int(value_net_main.num_actions))
            if not bool(known_mask.any().item()):
                continue
            states_s = states_s[known_mask]
            true_actions = true_actions[known_mask]

            mu_p, _ = prior_net(states_s)
            preds = value_net_main(mu_p, states_s).argmax(dim=1)
            a_onehot = to_one_hot(true_actions, value_net_main.num_actions)
            mu_q, _ = recognition_net(states_s, a_onehot)
            s_recon = generation_net(mu_q, a_onehot)
            errs = loss_fn(s_recon, states_s).mean(dim=1) * error_scale_factor

            for cls_id in torch.unique(true_actions).tolist():
                mask = (true_actions == cls_id) & (preds == cls_id)
                if mask.any():
                    per_class_errs.setdefault(int(cls_id), []).append(errs[mask].cpu().numpy())

    return {cls_id: np.concatenate(chunks) for cls_id, chunks in per_class_errs.items() if chunks}


# Backward-compatible name used by older tests/imports.
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
    return _compute_teacher_reconstruction_errors(
        features,
        labels,
        batch_size=batch_size,
        prior_net=prior_net,
        recognition_net=recognition_net,
        value_net_main=value_net_main,
        generation_net=generation_net,
        device=device,
        error_scale_factor=error_scale_factor,
    )


def _fit_evt_from_errors(
    per_class_errors: dict[int, np.ndarray], *, evt_cfg, logger: logging.Logger) -> dict[int, EVTModel]:
    evt_models: dict[int, EVTModel] = {}
    tail_percent = float(_cfg_value(evt_cfg, "tail_size_percent", 0.10))
    min_errs = int(_cfg_value(evt_cfg, "min_errors_per_class", 50))
    min_tail = int(_cfg_value(evt_cfg, "min_tail_size", 20))
    target_fpr = float(_cfg_value(evt_cfg, "target_known_fpr", 0.05))
    threshold_method = str(_cfg_value(evt_cfg, "threshold_method", "mef"))
    mef_min_q = float(_cfg_value(evt_cfg, "mef_min_quantile", 0.60))
    mef_max_q = float(_cfg_value(evt_cfg, "mef_max_quantile", 0.95))
    mef_candidates = int(_cfg_value(evt_cfg, "mef_num_candidates", 40))

    for cls_id, errs in sorted(per_class_errors.items()):
        errs = np.asarray(errs, dtype=np.float64)
        errs = errs[np.isfinite(errs)]
        if len(errs) < min_errs:
            logger.warning(
                "Insufficient reconstruction errors for class %d; skipping EVT | have=%d min=%d",
                cls_id,
                len(errs),
                min_errs,
            )
            continue
        evt_model = EVTModel(
            tail_size_percent=tail_percent,
            threshold_method=threshold_method,
            target_fpr=target_fpr,
        )
        evt_model.fit(
            errs,
            target_fpr=target_fpr,
            min_tail_size=min_tail,
            threshold_method=threshold_method,
            mef_min_quantile=mef_min_q,
            mef_max_quantile=mef_max_q,
            mef_num_candidates=mef_candidates,
            logger=logger,
        )
        evt_models[int(cls_id)] = evt_model
        logger.info(
            "Fitted class-wise EVT | class=%d | errors=%d | u=%.6g | decision_T=%.6g | tail=%d | method=%s",
            int(cls_id),
            int(evt_model.num_errors),
            float(evt_model.threshold_u or 0.0),
            float(evt_model.decision_threshold or 0.0),
            int(evt_model.tail_size),
            str(evt_model.threshold_selection.get("method", threshold_method)),
        )

    if not evt_models:
        raise RuntimeError("Failed to fit any EVT models.")
    return evt_models


def fit_evt_models(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    evt_cfg,
    prior_net: torch.nn.Module | None = None,
    recognition_net: torch.nn.Module | None = None,
    value_net_main: torch.nn.Module | None = None,
    generation_net: torch.nn.Module | None = None,
    student_model: torch.nn.Module | None = None,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> dict[int, EVTModel]:
    active_logger = logger or logging.getLogger("OpenSetEval")
    error_scale = _error_scale_factor(evt_cfg)
    backend = _evt_backend(evt_cfg)

    if any(m is None for m in (prior_net, recognition_net, value_net_main, generation_net)):
        raise ValueError("teacher_generator EVT requires prior/recognition/value/generation nets.")
    per_class_errors = _compute_teacher_reconstruction_errors(
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

    active_logger.info(
        "EVT fitting backend=%s | classes_with_errors=%s | error_scale=%.3g",
        backend,
        sorted(per_class_errors.keys()),
        error_scale,
    )
    return _fit_evt_from_errors(per_class_errors, evt_cfg=evt_cfg, logger=active_logger)


def calibrate_evt_thresholds(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    evt_models: dict[int, EVTModel],
    evt_cfg,
    prior_net: torch.nn.Module | None = None,
    recognition_net: torch.nn.Module | None = None,
    value_net_main: torch.nn.Module | None = None,
    generation_net: torch.nn.Module | None = None,
    student_model: torch.nn.Module | None = None,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Return metadata for already-fitted class-wise EVT models.

    Phase 2 no longer tunes a global probability delta.  Each class owns its own
    reconstruction-error rejection threshold derived from GPD/MEF calibration.
    The function is kept for call-site compatibility and artifact metadata.
    """
    active_logger = logger or logging.getLogger("OpenSetEval")
    _ = features, labels, batch_size, prior_net, recognition_net, value_net_main, generation_net, student_model, device
    thresholds = {
        str(cls_id): {
            "threshold_u": model.threshold_u,
            "decision_threshold": model.decision_threshold,
            "target_fpr": model.target_fpr,
            "tail_size": model.tail_size,
            "num_errors": model.num_errors,
            "tail_fraction": model.tail_fraction,
            "threshold_selection": model.threshold_selection,
        }
        for cls_id, model in sorted(evt_models.items())
    }
    active_logger.info(
        "EVT class-wise calibration complete | backend=%s | classes=%s",
        _evt_backend(evt_cfg),
        sorted(evt_models.keys()),
    )
    return {
        "backend": _evt_backend(evt_cfg),
        "decision_rule": "reconstruction_error_gt_class_evt_threshold",
        "target_known_fpr": float(_cfg_value(evt_cfg, "target_known_fpr", 0.05)),
        "threshold_method": str(_cfg_value(evt_cfg, "threshold_method", "mef")),
        "error_scale_factor": _error_scale_factor(evt_cfg),
        "unknown_label_id": int(_cfg_value(evt_cfg, "unknown_label_id", UNKNOWN_LABEL_ID)),
        "open_set_label_id": int(_cfg_value(evt_cfg, "open_set_label_id", OPEN_SET_LABEL_ID)),
        "class_thresholds": thresholds,
    }


def _teacher_predict_reconstruct_batch(
    *,
    states_s: torch.Tensor,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    value_net_main: torch.nn.Module,
    generation_net: torch.nn.Module,
    loss_fn: nn.Module,
    error_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_p, _ = prior_net(states_s)
    logits = value_net_main(mu_p, states_s)
    preds = logits.argmax(dim=1)
    one_hot = to_one_hot(preds, value_net_main.num_actions)
    mu_q, _ = recognition_net(states_s, one_hot)
    recon = generation_net(mu_q, one_hot)
    errs = loss_fn(recon, states_s).mean(dim=1) * error_scale
    return preds, logits, errs


def evaluate_open_set(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    prior_net: torch.nn.Module | None = None,
    recognition_net: torch.nn.Module | None = None,
    value_net_main: torch.nn.Module | None = None,
    generation_net: torch.nn.Module | None = None,
    student_model: torch.nn.Module | None = None,
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
    dataset = TensorDataset(features.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.MSELoss(reduction="none")
    output_dir = _ensure_dir(output_dir)

    backend = str(evt_meta.get("backend", _evt_backend(evt_cfg))).lower()
    if backend != "teacher_generator":
        backend = "teacher_generator"
    error_scale = float(evt_meta.get("error_scale_factor", _error_scale_factor(evt_cfg)))
    unknown_label_id = int(evt_meta.get("unknown_label_id", _cfg_value(evt_cfg, "unknown_label_id", UNKNOWN_LABEL_ID)))
    open_set_label_id = int(evt_meta.get("open_set_label_id", _cfg_value(evt_cfg, "open_set_label_id", OPEN_SET_LABEL_ID)))

    y_true_list: list[int] = []
    raw_pred_list: list[int] = []
    y_pred_list: list[int] = []
    y_score_unknown: list[float] = []
    reconstruction_errors: list[float] = []
    evt_thresholds_used: list[float] = []
    missing_evt_model_count = 0

    with torch.no_grad():
        if any(m is None for m in (prior_net, recognition_net, value_net_main, generation_net)):
            raise ValueError("teacher_generator open-set eval requires teacher/generator modules.")
        prior_net.eval()
        recognition_net.eval()
        value_net_main.eval()
        generation_net.eval()

        for states_s, lbls in loader:
            states_s = states_s.to(device).float()
            lbls = lbls.to(device).long()
            preds, _logits, errs = _teacher_predict_reconstruct_batch(
                states_s=states_s,
                prior_net=prior_net,
                recognition_net=recognition_net,
                value_net_main=value_net_main,
                generation_net=generation_net,
                loss_fn=loss_fn,
                error_scale=error_scale,
            )

            for i in range(states_s.size(0)):
                pred_label = int(preds[i].item())
                true_label = int(lbls[i].item())
                err_value = float(errs[i].item())
                model = evt_models.get(pred_label)

                if model is None:
                    unknown_score = 1.0
                    final_pred = open_set_label_id
                    threshold_value = float("nan")
                    missing_evt_model_count += 1
                else:
                    unknown_score = model.predict_probability_unknown(err_value)
                    final_pred = open_set_label_id if model.is_unknown(err_value) else pred_label
                    threshold_value = float(model.decision_threshold or model.threshold_u or np.nan)

                mapped_true = open_set_label_id if true_label == unknown_label_id else true_label
                y_true_list.append(mapped_true)
                raw_pred_list.append(pred_label)
                y_pred_list.append(final_pred)
                y_score_unknown.append(float(unknown_score))
                reconstruction_errors.append(err_value)
                evt_thresholds_used.append(threshold_value)

    y_true = np.array(y_true_list, dtype=int)
    y_raw_pred = np.array(raw_pred_list, dtype=int)
    y_pred = np.array(y_pred_list, dtype=int)
    y_scores = np.array(y_score_unknown, dtype=float)
    rec_errors = np.array(reconstruction_errors, dtype=float)
    y_binary = (y_true == open_set_label_id).astype(int)

    # For ROC/AUPRC, reconstruction error is the most direct monotonic Yang-style
    # score.  EVT tail probability is still exported for inspection.
    roc_scores = rec_errors if rec_errors.size else y_scores
    if np.unique(y_binary).size < 2:
        auroc = 0.0
        auprc = 0.0
        fpr95 = 1.0
    else:
        try:
            auroc = float(roc_auc_score(y_binary, roc_scores))
        except ValueError:
            auroc = 0.0
        if not np.isfinite(auroc):
            auroc = 0.0
        try:
            auprc = float(average_precision_score(y_binary, roc_scores))
        except ValueError:
            auprc = 0.0
        if not np.isfinite(auprc):
            auprc = 0.0
        try:
            fpr, tpr, _ = roc_curve(y_binary, roc_scores)
            valid = np.where(tpr >= 0.95)[0]
            fpr95 = float(fpr[valid[0]]) if valid.size else 1.0
        except ValueError:
            fpr95 = 1.0

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    unknown_f1 = f1_score(y_binary, (y_pred == open_set_label_id).astype(int), zero_division=0)
    known_mask = y_true != open_set_label_id
    unknown_mask = ~known_mask

    known_acc = accuracy_score(y_true[known_mask], y_pred[known_mask]) if known_mask.any() else 0.0
    unknown_recall = accuracy_score(y_true[unknown_mask], y_pred[unknown_mask]) if unknown_mask.any() else 0.0
    overall_acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0

    report_labels, class_name_list = _open_set_label_order(class_names, open_set_label_id)
    _save_labeled_confusion_matrix(
        output_dir / "before_osr_confusion_matrix.csv",
        y_true,
        y_raw_pred,
        label_ids=report_labels,
        label_names=class_name_list,
    )
    _save_labeled_confusion_matrix(
        output_dir / "after_osr_confusion_matrix.csv",
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
            "reconstruction_error": rec_errors,
            "evt_threshold": np.array(evt_thresholds_used, dtype=float),
            "is_unknown": y_binary,
            "backend": backend,
        }
    )
    scores_df.to_csv(output_dir / "open_set_scores.csv", index=False)
    scores_df[scores_df["is_unknown"] == 0].to_csv(
        output_dir / "teacher_reconstruction_errors_known.csv", index=False
    )
    scores_df[scores_df["is_unknown"] == 1].to_csv(
        output_dir / "teacher_reconstruction_errors_unknown.csv", index=False
    )

    evt_threshold_payload = {
        str(k): v.to_payload() for k, v in sorted(evt_models.items())
    }
    (output_dir / "evt_thresholds.json").write_text(
        json.dumps(evt_threshold_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if np.unique(y_binary).size >= 2:
        fpr, tpr, roc_thresholds = roc_curve(y_binary, roc_scores)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
            output_dir / "open_set_roc_curve.csv",
            index=False,
        )
        precision, recall, pr_thresholds = precision_recall_curve(y_binary, roc_scores)
        padded_thresholds = np.concatenate([pr_thresholds, [np.nan]])
        pd.DataFrame(
            {"precision": precision, "recall": recall, "threshold": padded_thresholds}
        ).to_csv(output_dir / "open_set_pr_curve.csv", index=False)

    active_logger.info(
        "Open-set metrics | backend=%s | AUROC=%.4f | AUPRC=%.4f | FPR95=%.4f | F1_macro=%.4f | Known_Acc=%.4f | Unknown_Recall=%.4f | Overall_Acc=%.4f | missing_evt=%d",
        backend,
        auroc,
        auprc,
        fpr95,
        f1_macro,
        known_acc,
        unknown_recall,
        overall_acc,
        missing_evt_model_count,
    )

    if report_to_stdout:
        print("\nOpen-Set Classification Report:\n")
        print(report)

    metrics = {
        "openset_f1_macro": float(f1_macro),
        "openset_auroc": float(auroc),
        "openset_auprc": float(auprc),
        "openset_fpr95": float(fpr95),
        "openset_unknown_f1": float(unknown_f1),
        "openset_known_acc": float(known_acc),
        "openset_unknown_recall": float(unknown_recall),
        "openset_overall_acc": float(overall_acc),
        "openset_missing_evt_model_count": float(missing_evt_model_count),
        "openset_error_scale_factor": float(error_scale),
        "openset_evt_backend": 0.0,
        "open_set/auroc": float(auroc),
        "open_set/auprc": float(auprc),
        "open_set/fpr95": float(fpr95),
        "open_set/unknown_detection_rate": float(unknown_recall),
        "open_set/unknown_f1": float(unknown_f1),
        "open_set/error_scale_factor": float(error_scale),
    }
    # Compatibility with old tests/artifacts. There is no global delta in phase 2.
    metrics["openset_global_delta"] = float(evt_meta.get("global_delta", 0.0))
    metrics["open_set/global_delta"] = float(evt_meta.get("global_delta", 0.0))

    (output_dir / "open_set_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics
