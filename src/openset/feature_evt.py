"""Feature-distance EVT open-set recognition utilities.

This module implements the replacement open-set path for DKD-FedOS:

1. The globally aggregated student produces a known-class feature vector.
2. For each known class, a compact feature boundary is estimated from correctly
   classified known validation samples.
3. EVT/GPD is fitted on the high tail of class-wise Mahalanobis distances.
4. A sample is rejected as Unknown when it falls outside the predicted class
   boundary.  Optionally, a support-gated local generator EVT boundary can add a
   second rejection test when local class support is reliable.

The implementation deliberately avoids weighted score fusion.  The decision is a
simple class-wise boundary rule, which is easier to debug and easier to justify
in a paper.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
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

logger = logging.getLogger("FeatureEVT")

UNKNOWN_LABEL_ID = -1
OPEN_SET_LABEL_ID = 99


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    return getattr(cfg, key, default) if cfg is not None else default


def _nested_cfg_value(cfg: Any, path: str, default: Any) -> Any:
    cur = cfg
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def mahalanobis_diag(
    features: np.ndarray,
    center: np.ndarray,
    variance: np.ndarray,
    *,
    eps: float = 1.0e-4,
) -> np.ndarray:
    """Diagonal Mahalanobis distance with shrinkage-safe variance."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mu = np.asarray(center, dtype=np.float64).reshape(1, -1)
    var = np.asarray(variance, dtype=np.float64).reshape(1, -1)
    var = np.maximum(var, float(eps))
    return np.sum(((x - mu) ** 2) / var, axis=1)


@dataclass
class FeatureEVTBoundary:
    class_id: int
    center: np.ndarray
    variance: np.ndarray
    evt_model: EVTModel
    threshold: float
    count: int
    distance_type: str = "mahalanobis_diag"
    covariance_eps: float = 1.0e-4

    def distance(self, features: np.ndarray) -> np.ndarray:
        return mahalanobis_diag(
            features,
            self.center,
            self.variance,
            eps=float(self.covariance_eps),
        )

    def is_unknown(self, features: np.ndarray) -> np.ndarray:
        distances = self.distance(features)
        return distances > float(self.threshold)

    def unknown_probability(self, distance: float) -> float:
        return self.evt_model.predict_probability_unknown(float(distance))

    def to_payload(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "center": np.asarray(self.center, dtype=float).tolist(),
            "variance": np.asarray(self.variance, dtype=float).tolist(),
            "evt_model": self.evt_model.to_payload(),
            "threshold": float(self.threshold),
            "count": int(self.count),
            "distance_type": self.distance_type,
            "covariance_eps": float(self.covariance_eps),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FeatureEVTBoundary":
        return cls(
            class_id=int(payload["class_id"]),
            center=np.asarray(payload["center"], dtype=np.float64),
            variance=np.asarray(payload["variance"], dtype=np.float64),
            evt_model=EVTModel.from_payload(payload["evt_model"]),
            threshold=float(payload["threshold"]),
            count=int(payload.get("count", 0)),
            distance_type=str(payload.get("distance_type", "mahalanobis_diag")),
            covariance_eps=float(payload.get("covariance_eps", 1.0e-4)),
        )


@dataclass
class LocalGeneratorBoundary:
    class_id: int
    evt_model: EVTModel
    threshold: float
    count: int
    valid: bool
    reason: str = "ok"
    known_reject_rate: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "evt_model": self.evt_model.to_payload(),
            "threshold": float(self.threshold),
            "count": int(self.count),
            "valid": bool(self.valid),
            "reason": str(self.reason),
            "known_reject_rate": float(self.known_reject_rate),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LocalGeneratorBoundary":
        return cls(
            class_id=int(payload["class_id"]),
            evt_model=EVTModel.from_payload(payload["evt_model"]),
            threshold=float(payload["threshold"]),
            count=int(payload.get("count", 0)),
            valid=bool(payload.get("valid", False)),
            reason=str(payload.get("reason", "loaded")),
            known_reject_rate=float(payload.get("known_reject_rate", 0.0)),
        )


def save_feature_evt_collection(
    boundaries: dict[int, FeatureEVTBoundary], filepath: Path | str, logger_: logging.Logger | None = None
) -> None:
    active_logger = logger_ or logger
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {int(k): v.to_payload() for k, v in boundaries.items()}
    joblib.dump(payload, path)
    active_logger.info("Saved feature EVT collection with %d classes to %s", len(payload), path)


def load_feature_evt_collection(filepath: Path | str) -> dict[int, FeatureEVTBoundary]:
    payload = joblib.load(filepath)
    if not isinstance(payload, dict):
        raise ValueError("Feature EVT collection file is corrupted or not a dict.")
    return {int(k): FeatureEVTBoundary.from_payload(v) for k, v in payload.items()}


def save_local_generator_evt_collection(
    boundaries: dict[int, LocalGeneratorBoundary], filepath: Path | str, logger_: logging.Logger | None = None
) -> None:
    active_logger = logger_ or logger
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {int(k): v.to_payload() for k, v in boundaries.items()}
    joblib.dump(payload, path)
    active_logger.info("Saved local generator EVT collection with %d classes to %s", len(payload), path)


def _student_feature_batches(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = TensorDataset(features.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feature_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    student_model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            h, logits = student_model(x)
            feature_chunks.append(h.detach().cpu().numpy())
            logit_chunks.append(logits.detach().cpu().numpy())
            label_chunks.append(y.cpu().numpy())
    if not feature_chunks:
        raise ValueError("No samples available for student feature extraction.")
    return (
        np.concatenate(feature_chunks, axis=0),
        np.concatenate(logit_chunks, axis=0),
        np.concatenate(label_chunks, axis=0).astype(int),
    )


def _evt_fit_kwargs(evt_cfg: Any) -> dict[str, Any]:
    return {
        "target_fpr": float(_cfg_value(evt_cfg, "target_known_fpr", 0.05)),
        "min_tail_size": int(_cfg_value(evt_cfg, "min_tail_size", 20)),
        "threshold_method": str(_cfg_value(evt_cfg, "threshold_method", "mef")),
        "mef_min_quantile": float(_cfg_value(evt_cfg, "mef_min_quantile", 0.70)),
        "mef_max_quantile": float(_cfg_value(evt_cfg, "mef_max_quantile", 0.98)),
        "mef_num_candidates": int(_cfg_value(evt_cfg, "mef_num_candidates", 40)),
    }


def fit_student_feature_evt_models(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    student_model: torch.nn.Module,
    evt_cfg: Any,
    device: torch.device,
    logger_: logging.Logger | None = None,
) -> tuple[dict[int, FeatureEVTBoundary], dict[str, Any], pd.DataFrame]:
    """Fit class-wise EVT boundaries on global-student feature distances."""
    active_logger = logger_ or logger
    h_all, logits_all, y_all = _student_feature_batches(
        features,
        labels,
        student_model=student_model,
        batch_size=batch_size,
        device=device,
    )
    preds = logits_all.argmax(axis=1).astype(int)
    num_classes = int(getattr(student_model, "num_classes", logits_all.shape[1]))
    fit_correct_only = bool(_cfg_value(evt_cfg, "fit_correct_only", True))
    min_errors = int(_cfg_value(evt_cfg, "min_errors_per_class", 50))
    tail_percent = float(_cfg_value(evt_cfg, "tail_size_percent", 0.10))
    target_fpr = float(_cfg_value(evt_cfg, "target_known_fpr", 0.05))
    covariance_eps = float(_cfg_value(evt_cfg, "covariance_eps", 1.0e-4))

    boundaries: dict[int, FeatureEVTBoundary] = {}
    rows: list[dict[str, Any]] = []
    for class_id in range(num_classes):
        mask = y_all == class_id
        if fit_correct_only:
            mask = mask & (preds == y_all)
        h_k = h_all[mask]
        if h_k.shape[0] < min_errors and fit_correct_only:
            active_logger.warning(
                "Feature EVT class %d has too few correctly classified samples; retrying with all known class samples | have=%d min=%d",
                class_id,
                int(h_k.shape[0]),
                int(min_errors),
            )
            mask = y_all == class_id
            h_k = h_all[mask]
        if h_k.shape[0] < min_errors:
            active_logger.warning(
                "Feature EVT skipped class %d | have=%d min=%d",
                class_id,
                int(h_k.shape[0]),
                int(min_errors),
            )
            continue
        center = h_k.mean(axis=0)
        variance = h_k.var(axis=0) + covariance_eps
        distances = mahalanobis_diag(h_k, center, variance, eps=covariance_eps)
        evt_model = EVTModel(
            tail_size_percent=tail_percent,
            threshold_method=str(_cfg_value(evt_cfg, "threshold_method", "mef")),
            target_fpr=target_fpr,
        )
        evt_model.fit(distances, logger=active_logger, **_evt_fit_kwargs(evt_cfg))
        threshold = float(evt_model.decision_threshold or evt_model.threshold_u or np.quantile(distances, 1.0 - target_fpr))
        boundaries[class_id] = FeatureEVTBoundary(
            class_id=class_id,
            center=center,
            variance=variance,
            evt_model=evt_model,
            threshold=threshold,
            count=int(h_k.shape[0]),
            covariance_eps=covariance_eps,
        )
        for d in distances:
            rows.append({"class_id": class_id, "distance": float(d), "threshold": threshold, "split": "calibration"})
        active_logger.info(
            "Fitted global student Feature-EVT | class=%d | count=%d | threshold=%.6g | u=%.6g | tail=%d | method=%s",
            class_id,
            int(h_k.shape[0]),
            threshold,
            float(evt_model.threshold_u or 0.0),
            int(evt_model.tail_size),
            str(evt_model.threshold_selection.get("method", _cfg_value(evt_cfg, "threshold_method", "mef"))),
        )

    if not boundaries:
        raise RuntimeError("Failed to fit any global student feature EVT boundary.")

    meta = {
        "backend": "student_feature_evt",
        "score": "mahalanobis_feature_distance",
        "decision_rule": "feature_distance_gt_class_evt_threshold",
        "target_known_fpr": target_fpr,
        "threshold_method": str(_cfg_value(evt_cfg, "threshold_method", "mef")),
        "unknown_label_id": int(_cfg_value(evt_cfg, "unknown_label_id", UNKNOWN_LABEL_ID)),
        "open_set_label_id": int(_cfg_value(evt_cfg, "open_set_label_id", OPEN_SET_LABEL_ID)),
        "covariance": str(_cfg_value(evt_cfg, "covariance", "diagonal_shrinkage")),
        "covariance_eps": covariance_eps,
        "class_thresholds": {str(k): v.to_payload() for k, v in sorted(boundaries.items())},
    }
    return boundaries, meta, pd.DataFrame(rows)


def _teacher_reconstruction_errors_for_classes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    value_net_main: torch.nn.Module,
    generation_net: torch.nn.Module,
    device: torch.device,
    error_scale_factor: float = 100000.0,
    fit_correct_only: bool = True,
) -> dict[int, np.ndarray]:
    loss_fn = nn.MSELoss(reduction="none")
    dataset = TensorDataset(features.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_class: dict[int, list[np.ndarray]] = {}
    prior_net.eval(); recognition_net.eval(); value_net_main.eval(); generation_net.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            known_mask = (y >= 0) & (y < int(value_net_main.num_actions))
            if not bool(known_mask.any().item()):
                continue
            x = x[known_mask]
            y = y[known_mask]
            mu_p, _ = prior_net(x)
            logits = value_net_main(mu_p, x)
            preds = logits.argmax(dim=1)
            onehot = to_one_hot(y, int(value_net_main.num_actions))
            mu_q, _ = recognition_net(x, onehot)
            recon = generation_net(mu_q, onehot)
            errs = loss_fn(recon, x).mean(dim=1) * float(error_scale_factor)
            for class_id in torch.unique(y).tolist():
                mask = y == int(class_id)
                if fit_correct_only:
                    mask = mask & (preds == y)
                if mask.any():
                    per_class.setdefault(int(class_id), []).append(errs[mask].cpu().numpy())
    return {k: np.concatenate(v) for k, v in per_class.items() if v}


def fit_local_generator_evt_models(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    value_net_main: torch.nn.Module,
    generation_net: torch.nn.Module,
    evt_cfg: Any,
    device: torch.device,
    logger_: logging.Logger | None = None,
) -> dict[int, LocalGeneratorBoundary]:
    """Fit gated local generator EVT models for dual-boundary evaluation."""
    active_logger = logger_ or logger
    dual_cfg = getattr(evt_cfg, "dual_boundary", None)
    min_support = int(_nested_cfg_value(evt_cfg, "dual_boundary.local_min_clean_count", 100))
    max_reject = float(_nested_cfg_value(evt_cfg, "dual_boundary.local_max_known_reject_rate", 0.10))
    error_scale = float(_cfg_value(evt_cfg, "error_scale_factor", 100000.0))
    fit_correct_only = bool(_cfg_value(evt_cfg, "fit_correct_only", True))
    per_class = _teacher_reconstruction_errors_for_classes(
        features,
        labels,
        batch_size=batch_size,
        prior_net=prior_net,
        recognition_net=recognition_net,
        value_net_main=value_net_main,
        generation_net=generation_net,
        device=device,
        error_scale_factor=error_scale,
        fit_correct_only=fit_correct_only,
    )
    boundaries: dict[int, LocalGeneratorBoundary] = {}
    for class_id in range(int(value_net_main.num_actions)):
        errs = np.asarray(per_class.get(class_id, np.asarray([], dtype=np.float64)), dtype=np.float64)
        if errs.size < min_support:
            active_logger.info(
                "LOCAL GENERATOR EVT | class=%d clean_count=%d valid=false reason=low_support min=%d",
                class_id,
                int(errs.size),
                int(min_support),
            )
            continue
        evt_model = EVTModel(
            tail_size_percent=float(_cfg_value(evt_cfg, "tail_size_percent", 0.10)),
            threshold_method=str(_cfg_value(evt_cfg, "threshold_method", "mef")),
            target_fpr=float(_cfg_value(evt_cfg, "target_known_fpr", 0.05)),
        )
        evt_model.fit(errs, logger=active_logger, **_evt_fit_kwargs(evt_cfg))
        threshold = float(evt_model.decision_threshold or evt_model.threshold_u or np.quantile(errs, 0.95))
        known_reject_rate = float(np.mean(errs > threshold)) if errs.size else 1.0
        valid = known_reject_rate <= max_reject
        reason = "ok" if valid else "high_known_reject_rate"
        boundaries[class_id] = LocalGeneratorBoundary(
            class_id=class_id,
            evt_model=evt_model,
            threshold=threshold,
            count=int(errs.size),
            valid=valid,
            reason=reason,
            known_reject_rate=known_reject_rate,
        )
        active_logger.info(
            "LOCAL GENERATOR EVT | class=%d clean_count=%d valid=%s threshold=%.6g known_reject_rate=%.4f reason=%s",
            class_id,
            int(errs.size),
            bool(valid),
            threshold,
            known_reject_rate,
            reason,
        )
    return boundaries


def _local_generator_batch_errors(
    states_s: torch.Tensor,
    pred_labels: torch.Tensor,
    *,
    prior_net: torch.nn.Module,
    recognition_net: torch.nn.Module,
    generation_net: torch.nn.Module,
    num_actions: int,
    error_scale_factor: float,
) -> torch.Tensor:
    loss_fn = nn.MSELoss(reduction="none")
    onehot = to_one_hot(pred_labels, int(num_actions))
    mu_q, _ = recognition_net(states_s, onehot)
    recon = generation_net(mu_q, onehot)
    return loss_fn(recon, states_s).mean(dim=1) * float(error_scale_factor)


def evaluate_feature_evt_open_set(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    student_model: torch.nn.Module,
    feature_boundaries: dict[int, FeatureEVTBoundary],
    evt_meta: dict[str, Any],
    class_names: dict[int, str],
    output_dir: Path,
    device: torch.device,
    evt_cfg: Any = None,
    report_to_stdout: bool = False,
    logger_: logging.Logger | None = None,
    local_generator_boundaries: dict[int, LocalGeneratorBoundary] | None = None,
    prior_net: torch.nn.Module | None = None,
    recognition_net: torch.nn.Module | None = None,
    value_net_main: torch.nn.Module | None = None,
    generation_net: torch.nn.Module | None = None,
) -> dict[str, float]:
    """Evaluate global feature EVT with optional gated local generator boundary."""
    active_logger = logger_ or logger
    output_dir = _ensure_dir(output_dir)
    backend = str(evt_meta.get("backend", _cfg_value(evt_cfg, "backend", "student_feature_evt"))).lower()
    unknown_label_id = int(evt_meta.get("unknown_label_id", _cfg_value(evt_cfg, "unknown_label_id", UNKNOWN_LABEL_ID)))
    open_set_label_id = int(evt_meta.get("open_set_label_id", _cfg_value(evt_cfg, "open_set_label_id", OPEN_SET_LABEL_ID)))
    error_scale = float(_cfg_value(evt_cfg, "error_scale_factor", 100000.0))
    use_local = (
        local_generator_boundaries is not None
        and prior_net is not None
        and recognition_net is not None
        and value_net_main is not None
        and generation_net is not None
    )

    dataset = TensorDataset(features.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true_list: list[int] = []
    raw_pred_list: list[int] = []
    y_pred_list: list[int] = []
    scores: list[float] = []
    feature_distances: list[float] = []
    feature_thresholds: list[float] = []
    local_errors: list[float] = []
    local_thresholds: list[float] = []
    global_reject_flags: list[int] = []
    local_reject_flags: list[int] = []
    local_valid_flags: list[int] = []
    missing_boundary_count = 0
    local_reject_count = 0
    global_reject_count = 0

    student_model.eval()
    if use_local:
        prior_net.eval(); recognition_net.eval(); value_net_main.eval(); generation_net.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            h, logits = student_model(x)
            preds = logits.argmax(dim=1)
            h_np = h.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy().astype(int)

            local_batch_errs = None
            if use_local:
                local_batch_errs = _local_generator_batch_errors(
                    x,
                    preds,
                    prior_net=prior_net,
                    recognition_net=recognition_net,
                    generation_net=generation_net,
                    num_actions=int(value_net_main.num_actions),
                    error_scale_factor=error_scale,
                ).detach().cpu().numpy()

            for idx in range(x.shape[0]):
                pred_label = int(preds_np[idx])
                true_label = int(y[idx].item())
                mapped_true = open_set_label_id if true_label == unknown_label_id else true_label
                boundary = feature_boundaries.get(pred_label)
                if boundary is None:
                    missing_boundary_count += 1
                    d_value = float("nan")
                    t_value = float("nan")
                    global_reject = True
                    score_value = 1.0
                else:
                    d_value = float(boundary.distance(h_np[idx])[0])
                    t_value = float(boundary.threshold)
                    global_reject = d_value > t_value
                    # class-normalized score is robust for ROC across classes.
                    score_value = float(d_value / max(t_value, 1e-12))
                final_pred = open_set_label_id if global_reject else pred_label
                if global_reject:
                    global_reject_count += 1

                local_error_value = float("nan")
                local_threshold_value = float("nan")
                local_valid = False
                local_reject = False
                if not global_reject and use_local and local_batch_errs is not None:
                    local_boundary = local_generator_boundaries.get(pred_label) if local_generator_boundaries else None
                    if local_boundary is not None and local_boundary.valid:
                        local_valid = True
                        local_error_value = float(local_batch_errs[idx])
                        local_threshold_value = float(local_boundary.threshold)
                        local_reject = local_error_value > local_threshold_value
                        if local_reject:
                            final_pred = open_set_label_id
                            local_reject_count += 1
                        score_value = max(
                            score_value,
                            float(local_error_value / max(local_threshold_value, 1e-12)),
                        )

                y_true_list.append(mapped_true)
                raw_pred_list.append(pred_label)
                y_pred_list.append(final_pred)
                scores.append(score_value)
                feature_distances.append(d_value)
                feature_thresholds.append(t_value)
                local_errors.append(local_error_value)
                local_thresholds.append(local_threshold_value)
                global_reject_flags.append(int(global_reject))
                local_reject_flags.append(int(local_reject))
                local_valid_flags.append(int(local_valid))

    y_true = np.asarray(y_true_list, dtype=int)
    y_raw_pred = np.asarray(raw_pred_list, dtype=int)
    y_pred = np.asarray(y_pred_list, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    y_binary = (y_true == open_set_label_id).astype(int)

    if np.unique(y_binary).size < 2:
        auroc = 0.0; auprc = 0.0; fpr95 = 1.0
    else:
        try:
            auroc = float(roc_auc_score(y_binary, score_arr))
        except ValueError:
            auroc = 0.0
        if not np.isfinite(auroc): auroc = 0.0
        try:
            auprc = float(average_precision_score(y_binary, score_arr))
        except ValueError:
            auprc = 0.0
        if not np.isfinite(auprc): auprc = 0.0
        try:
            fpr, tpr, _ = roc_curve(y_binary, score_arr)
            valid = np.where(tpr >= 0.95)[0]
            fpr95 = float(fpr[valid[0]]) if valid.size else 1.0
        except ValueError:
            fpr95 = 1.0

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    unknown_f1 = float(f1_score(y_binary, (y_pred == open_set_label_id).astype(int), zero_division=0))
    known_mask = y_true != open_set_label_id
    unknown_mask = ~known_mask
    known_acc = float(accuracy_score(y_true[known_mask], y_pred[known_mask])) if known_mask.any() else 0.0
    unknown_recall = float(accuracy_score(y_true[unknown_mask], y_pred[unknown_mask])) if unknown_mask.any() else 0.0
    overall_acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    known_false_unknown_rate = float(np.mean(y_pred[known_mask] == open_set_label_id)) if known_mask.any() else 0.0

    report_labels, class_name_list = _open_set_label_order(class_names, open_set_label_id)
    _save_labeled_confusion_matrix(output_dir / "before_osr_confusion_matrix.csv", y_true, y_raw_pred, label_ids=report_labels, label_names=class_name_list)
    _save_labeled_confusion_matrix(output_dir / "after_osr_confusion_matrix.csv", y_true, y_pred, label_ids=report_labels, label_names=class_name_list)
    report = classification_report(y_true, y_pred, labels=report_labels, target_names=class_name_list, digits=4, zero_division=0)
    (output_dir / "openset_report.txt").write_text(report, encoding="utf-8")

    scores_df = pd.DataFrame({
        "y_true": y_true,
        "raw_pred": y_raw_pred,
        "y_pred": y_pred,
        "unknown_score": score_arr,
        "feature_distance": np.asarray(feature_distances, dtype=float),
        "feature_threshold": np.asarray(feature_thresholds, dtype=float),
        "local_reconstruction_error": np.asarray(local_errors, dtype=float),
        "local_reconstruction_threshold": np.asarray(local_thresholds, dtype=float),
        "global_reject": np.asarray(global_reject_flags, dtype=int),
        "local_reject": np.asarray(local_reject_flags, dtype=int),
        "local_generator_valid": np.asarray(local_valid_flags, dtype=int),
        "is_unknown": y_binary,
        "backend": backend,
    })
    scores_df.to_csv(output_dir / "open_set_scores.csv", index=False)
    scores_df[scores_df["is_unknown"] == 0].to_csv(output_dir / "student_feature_distances_known.csv", index=False)
    scores_df[scores_df["is_unknown"] == 1].to_csv(output_dir / "student_feature_distances_unknown.csv", index=False)

    (output_dir / "feature_evt_thresholds.json").write_text(
        json.dumps({str(k): v.to_payload() for k, v in sorted(feature_boundaries.items())}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if local_generator_boundaries:
        (output_dir / "local_generator_evt_thresholds.json").write_text(
            json.dumps({str(k): v.to_payload() for k, v in sorted(local_generator_boundaries.items())}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if np.unique(y_binary).size >= 2:
        fpr, tpr, roc_thresholds = roc_curve(y_binary, score_arr)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(output_dir / "open_set_roc_curve.csv", index=False)
        precision, recall, pr_thresholds = precision_recall_curve(y_binary, score_arr)
        padded = np.concatenate([pr_thresholds, [np.nan]])
        pd.DataFrame({"precision": precision, "recall": recall, "threshold": padded}).to_csv(output_dir / "open_set_pr_curve.csv", index=False)

    active_logger.info(
        "Open-set metrics | backend=%s | AUROC=%.4f | AUPRC=%.4f | FPR95=%.4f | F1_macro=%.4f | Known_Acc=%.4f | Unknown_Recall=%.4f | Known_FU=%.4f | Overall_Acc=%.4f | global_rejects=%d | local_rejects=%d | missing_feature_evt=%d",
        backend, auroc, auprc, fpr95, f1_macro, known_acc, unknown_recall,
        known_false_unknown_rate, overall_acc, global_reject_count, local_reject_count, missing_boundary_count,
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
        "openset_known_false_unknown_rate": float(known_false_unknown_rate),
        "openset_overall_acc": float(overall_acc),
        "openset_missing_evt_model_count": float(missing_boundary_count),
        "openset_global_reject_count": float(global_reject_count),
        "openset_local_reject_count": float(local_reject_count),
        "openset_local_valid_used_count": float(np.sum(local_valid_flags)),
        "openset_evt_backend": 2.0 if backend == "dual_boundary_evt" else 1.0,
        "open_set/auroc": float(auroc),
        "open_set/auprc": float(auprc),
        "open_set/fpr95": float(fpr95),
        "open_set/unknown_detection_rate": float(unknown_recall),
        "open_set/unknown_f1": float(unknown_f1),
        "open_set/known_false_unknown_rate": float(known_false_unknown_rate),
    }
    (output_dir / "open_set_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics
