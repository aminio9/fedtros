"""Fed-DiGOS open-set calibration and evaluation.

Fed-DiGOS attaches a disentangled generator branch to the federated student.
Open-set rejection uses max-calibrated tail probability from:
  1. student OSR reconstruction/latent score,
  2. student logit energy,
  3. optional prototype distance in classifier feature space.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
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

logger = logging.getLogger("FedDiGOS")
UNKNOWN_LABEL_ID = -1
OPEN_SET_LABEL_ID = 99


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    return getattr(cfg, key, default) if cfg is not None else default


def _nested(cfg: Any, path: str, default: Any) -> Any:
    cur = cfg
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _evt_kwargs(cfg: Any) -> dict[str, Any]:
    return {
        "target_fpr": float(_nested(cfg, "evt.target_known_fpr", 0.05)),
        "min_tail_size": int(_nested(cfg, "evt.min_tail_size", 20)),
        "threshold_method": str(_nested(cfg, "evt.threshold_method", "mef")),
        "mef_min_quantile": float(_nested(cfg, "evt.mef_min_quantile", 0.70)),
        "mef_max_quantile": float(_nested(cfg, "evt.mef_max_quantile", 0.98)),
        "mef_num_candidates": int(_nested(cfg, "evt.mef_num_candidates", 40)),
    }


def _fit_evt(scores: np.ndarray, cfg: Any, log: logging.Logger) -> EVTModel:
    model = EVTModel(
        tail_size_percent=float(_nested(cfg, "evt.tail_size_percent", 0.10)),
        threshold_method=str(_nested(cfg, "evt.threshold_method", "mef")),
        target_fpr=float(_nested(cfg, "evt.target_known_fpr", 0.05)),
    )
    model.fit(np.asarray(scores, dtype=np.float64), logger=log, **_evt_kwargs(cfg))
    return model


@dataclass
class PrototypeBank:
    prototypes: dict[int, np.ndarray]
    eps: float = 1.0e-8

    def score(self, features: np.ndarray, class_id: int) -> np.ndarray:
        p = self.prototypes.get(int(class_id))
        if p is None or p.size == 0:
            return np.full((features.shape[0],), np.nan, dtype=np.float64)
        x = np.asarray(features, dtype=np.float64)
        # Euclidean distance to nearest activation prototype. Higher = more unknown.
        d = ((x[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
        return np.sqrt(np.min(d, axis=1) + self.eps)

    def to_payload(self) -> dict[str, Any]:
        return {str(k): v.tolist() for k, v in sorted(self.prototypes.items())}


def _class_labels(class_names: dict[int, str], open_set_label_id: int) -> tuple[list[int], list[str]]:
    ids = sorted(int(k) for k in class_names)
    return ids + [open_set_label_id], [class_names[k] for k in ids] + ["Unknown"]


def _collect_student_scores(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    cfg: Any,
    class_condition: str = "true",
) -> pd.DataFrame:
    nll_weight = float(_cfg_value(cfg, "latent_nll_weight", 0.10))
    temperature = float(_nested(cfg, "energy.temperature", 1.0))
    loader = DataLoader(TensorDataset(features.float(), labels.long()), batch_size=batch_size, shuffle=False)
    rows: list[dict[str, Any]] = []
    student_model.eval()
    with torch.no_grad():
        offset = 0
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long().view(-1)
            h, logits = student_model(x)
            pred = torch.argmax(logits, dim=1)
            if class_condition == "pred":
                cond = pred
            else:
                cond = y.clamp(0, int(student_model.num_classes) - 1)
            osr = student_model.osr_score(x, cond, nll_weight=nll_weight, detach_features=True)
            energy = student_model.energy_score(logits, temperature=temperature)
            h_np = h.detach().cpu().numpy()
            for i in range(x.shape[0]):
                rows.append({
                    "sample_id": int(offset + i),
                    "y_raw": int(y[i].item()),
                    "pred_before_osr": int(pred[i].item()),
                    "condition_class": int(cond[i].item()),
                    "gen_score": float(osr["score"][i].detach().cpu().item()),
                    "recon_error": float(osr["recon_error"][i].detach().cpu().item()),
                    "latent_nll": float(osr["latent_nll"][i].detach().cpu().item()),
                    "energy_score": float(energy[i].detach().cpu().item()),
                    "correct_known": int((y[i].item() >= 0) and (y[i].item() == pred[i].item())),
                    "feature": h_np[i],
                })
            offset += int(x.shape[0])
    return pd.DataFrame(rows)


def _fit_prototypes(calib_df: pd.DataFrame, num_classes: int, cfg: Any, log: logging.Logger) -> PrototypeBank:
    enabled = bool(_nested(cfg, "prototype.enabled", True))
    k_default = int(_nested(cfg, "prototype.num_prototypes_per_class", 4))
    min_per_proto = int(_nested(cfg, "prototype.min_samples_per_prototype", 25))
    prototypes: dict[int, np.ndarray] = {}
    if not enabled:
        return PrototypeBank(prototypes)
    for c in range(num_classes):
        cls = calib_df[(calib_df["y_raw"] == c) & (calib_df["correct_known"] == 1)]
        if cls.empty:
            continue
        feats = np.stack(cls["feature"].to_numpy()).astype(np.float64)
        k = max(1, min(k_default, feats.shape[0] // max(min_per_proto, 1)))
        if k <= 1:
            centers = feats.mean(axis=0, keepdims=True)
        else:
            try:
                km = KMeans(n_clusters=k, n_init=5, random_state=42)
                km.fit(feats)
                centers = km.cluster_centers_
            except Exception as exc:
                log.warning("Fed-DiGOS prototype KMeans failed for class=%d: %s; using mean", c, exc)
                centers = feats.mean(axis=0, keepdims=True)
        prototypes[c] = centers.astype(np.float64)
        log.info("Fed-DiGOS prototypes | class=%d | samples=%d | k=%d", c, feats.shape[0], centers.shape[0])
    return PrototypeBank(prototypes)


def calibrate_fed_digos(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    cfg: Any,
    logger_: logging.Logger | None = None,
) -> tuple[dict[str, dict[int, EVTModel]], PrototypeBank, pd.DataFrame, dict[str, Any]]:
    log = logger_ or logger
    if not bool(getattr(student_model, "osr_enabled", False)):
        raise RuntimeError("Fed-DiGOS requires student_model.osr_enabled=True.")
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    unknown_label_id = int(_nested(cfg, "unknown_label_id", UNKNOWN_LABEL_ID))
    if np.any(labels_np == unknown_label_id):
        raise ValueError("Fed-DiGOS calibration data must contain known classes only; found unknown labels.")
    df = _collect_student_scores(
        features,
        labels,
        student_model=student_model,
        batch_size=batch_size,
        device=device,
        cfg=cfg,
        class_condition="true",
    )
    num_classes = int(student_model.num_classes)
    fit_correct_only = bool(_nested(cfg, "evt.fit_correct_only", True))
    min_errors = int(_nested(cfg, "evt.min_errors_per_class", 50))
    prototype_bank = _fit_prototypes(df, num_classes, cfg, log)
    proto_scores = []
    for _, row in df.iterrows():
        proto_scores.append(float(prototype_bank.score(np.asarray(row["feature"]).reshape(1, -1), int(row["y_raw"]))[0]))
    df["prototype_score"] = proto_scores

    models: dict[str, dict[int, EVTModel]] = {"gen": {}, "energy": {}, "prototype": {}}
    for c in range(num_classes):
        cls = df[df["y_raw"] == c]
        if fit_correct_only:
            cls_fit = cls[cls["correct_known"] == 1]
            if len(cls_fit) < min_errors:
                log.warning("Fed-DiGOS class=%d too few correct calibration samples (%d); using all class samples.", c, len(cls_fit))
                cls_fit = cls
        else:
            cls_fit = cls
        if len(cls_fit) < min_errors:
            log.warning("Fed-DiGOS EVT skipped class=%d | samples=%d min=%d", c, len(cls_fit), min_errors)
            continue
        for name, col in [("gen", "gen_score"), ("energy", "energy_score"), ("prototype", "prototype_score")]:
            values = cls_fit[col].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size < min_errors:
                continue
            models[name][c] = _fit_evt(values, cfg, log)
        log.info(
            "Fed-DiGOS calibration | class=%d n=%d gen_q50=%.6g gen_q95=%.6g T_gen=%.6g "
            "energy_q95=%.6g T_energy=%.6g proto_q95=%.6g T_proto=%.6g",
            c,
            len(cls_fit),
            float(np.quantile(cls_fit["gen_score"], 0.50)),
            float(np.quantile(cls_fit["gen_score"], 0.95)),
            float(models["gen"].get(c).decision_threshold if c in models["gen"] else np.nan),
            float(np.quantile(cls_fit["energy_score"], 0.95)),
            float(models["energy"].get(c).decision_threshold if c in models["energy"] else np.nan),
            float(np.nanquantile(cls_fit["prototype_score"], 0.95)),
            float(models["prototype"].get(c).decision_threshold if c in models["prototype"] else np.nan),
        )
    meta = {
        "backend": "fed_digos",
        "decision_rule": "max_tail_probability_and_or_evt_rejection",
        "scores": ["gen", "energy", "prototype"],
        "num_classes": num_classes,
        "unknown_label_id": int(unknown_label_id),
        "open_set_label_id": int(_nested(cfg, "open_set_label_id", OPEN_SET_LABEL_ID)),
        "thresholds": {
            name: {str(k): v.to_payload() for k, v in sorted(score_models.items())}
            for name, score_models in models.items()
        },
        "prototypes": prototype_bank.to_payload(),
    }
    return models, prototype_bank, df, meta


def evaluate_fed_digos(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    cfg: Any,
    class_names: dict[int, str],
    output_dir: Path,
    evt_models: dict[str, dict[int, EVTModel]],
    prototype_bank: PrototypeBank,
    calibration_df: pd.DataFrame | None = None,
    logger_: logging.Logger | None = None,
    report_to_stdout: bool = False,
) -> dict[str, float]:
    log = logger_ or logger
    output_dir = _ensure_dir(output_dir)
    open_set_label_id = int(_nested(cfg, "open_set_label_id", OPEN_SET_LABEL_ID))
    unknown_label_id = int(_nested(cfg, "unknown_label_id", UNKNOWN_LABEL_ID))
    df = _collect_student_scores(
        features,
        labels,
        student_model=student_model,
        batch_size=batch_size,
        device=device,
        cfg=cfg,
        class_condition="pred",
    )
    proto_scores = []
    for _, row in df.iterrows():
        proto_scores.append(float(prototype_bank.score(np.asarray(row["feature"]).reshape(1, -1), int(row["pred_before_osr"]))[0]))
    df["prototype_score"] = proto_scores

    y_true = np.asarray([open_set_label_id if int(v) == unknown_label_id else int(v) for v in df["y_raw"]], dtype=int)
    y_before = df["pred_before_osr"].to_numpy(dtype=int)
    y_binary = (y_true == open_set_label_id).astype(int)
    final_preds = []
    unknown_probs = []
    gen_rejects = []
    energy_rejects = []
    proto_rejects = []
    gen_probs = []
    energy_probs = []
    proto_probs = []
    T_gen_used = []
    T_energy_used = []
    T_proto_used = []

    for _, row in df.iterrows():
        c = int(row["pred_before_osr"])
        score_items = [("gen", "gen_score"), ("energy", "energy_score"), ("prototype", "prototype_score")]
        rejects = {}
        probs = {}
        thresholds = {}
        for name, col in score_items:
            model = evt_models.get(name, {}).get(c)
            value = float(row[col])
            if model is None or not np.isfinite(value):
                rejects[name] = False
                probs[name] = 0.0
                thresholds[name] = np.nan
            else:
                rejects[name] = bool(model.is_unknown(value))
                probs[name] = float(model.predict_probability_unknown(value))
                thresholds[name] = float(model.decision_threshold if model.decision_threshold is not None else np.nan)
        final_reject = rejects["gen"] or rejects["energy"] or rejects["prototype"]
        final_preds.append(open_set_label_id if final_reject else c)
        gen_rejects.append(int(rejects["gen"])); energy_rejects.append(int(rejects["energy"])); proto_rejects.append(int(rejects["prototype"]))
        gen_probs.append(float(probs["gen"])); energy_probs.append(float(probs["energy"])); proto_probs.append(float(probs["prototype"]))
        unknown_probs.append(float(max(probs.values())))
        T_gen_used.append(thresholds["gen"]); T_energy_used.append(thresholds["energy"]); T_proto_used.append(thresholds["prototype"])

    y_pred = np.asarray(final_preds, dtype=int)
    score_arr = np.asarray(unknown_probs, dtype=float)
    if np.unique(y_binary).size < 2:
        auroc = 0.0; auprc = 0.0; fpr95 = 1.0
    else:
        auroc = float(roc_auc_score(y_binary, score_arr))
        auprc = float(average_precision_score(y_binary, score_arr))
        fpr, tpr, roc_thresholds = roc_curve(y_binary, score_arr)
        valid = np.where(tpr >= 0.95)[0]
        fpr95 = float(fpr[valid[0]]) if valid.size else 1.0
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(output_dir / "open_set_roc_curve.csv", index=False)
        precision, recall, pr_thresholds = precision_recall_curve(y_binary, score_arr)
        pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.concatenate([pr_thresholds, [np.nan]])}).to_csv(output_dir / "open_set_pr_curve.csv", index=False)

    known_mask = y_true != open_set_label_id
    unknown_mask = ~known_mask
    known_acc_before = float(accuracy_score(y_true[known_mask], y_before[known_mask])) if known_mask.any() else 0.0
    known_acc_after = float(accuracy_score(y_true[known_mask], y_pred[known_mask])) if known_mask.any() else 0.0
    unknown_recall = float(accuracy_score(y_true[unknown_mask], y_pred[unknown_mask])) if unknown_mask.any() else 0.0
    known_false_unknown_rate = float(np.mean(y_pred[known_mask] == open_set_label_id)) if known_mask.any() else 0.0
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    unknown_f1 = float(f1_score(y_binary, (y_pred == open_set_label_id).astype(int), zero_division=0))
    overall_acc = float(accuracy_score(y_true, y_pred))

    report_labels, report_names = _class_labels(class_names, open_set_label_id)
    before_cm = confusion_matrix(y_true, y_before, labels=report_labels)
    after_cm = confusion_matrix(y_true, y_pred, labels=report_labels)
    pd.DataFrame(before_cm, index=report_names, columns=report_names).to_csv(output_dir / "before_osr_confusion_matrix.csv")
    pd.DataFrame(after_cm, index=report_names, columns=report_names).to_csv(output_dir / "after_osr_confusion_matrix.csv")
    report = classification_report(y_true, y_pred, labels=report_labels, target_names=report_names, digits=4, zero_division=0)
    (output_dir / "openset_report.txt").write_text(report, encoding="utf-8")
    if report_to_stdout:
        print(report)

    df["y_true"] = y_true
    df["pred_after_osr"] = y_pred
    df["gen_unknown_prob"] = gen_probs
    df["energy_unknown_prob"] = energy_probs
    df["prototype_unknown_prob"] = proto_probs
    df["unknown_score"] = score_arr
    df["T_gen_used"] = T_gen_used
    df["T_energy_used"] = T_energy_used
    df["T_proto_used"] = T_proto_used
    df["gen_reject"] = gen_rejects
    df["energy_reject"] = energy_rejects
    df["prototype_reject"] = proto_rejects
    df["final_reject"] = (y_pred == open_set_label_id).astype(int)
    df["known_or_unknown"] = np.where(y_binary == 1, "unknown", "known")
    # Do not serialize high-dimensional feature arrays in the main CSV.  The CSV
    # is for debugging, not for making Excel beg for mercy.
    export_df = df.drop(columns=["feature"])
    export_df.to_csv(output_dir / "open_set_scores.csv", index=False)

    if calibration_df is not None:
        calibration_df.drop(columns=["feature"], errors="ignore").to_csv(output_dir / "fed_digos_calibration_scores.csv", index=False)
    thresholds_payload = {
        name: {str(k): v.to_payload() for k, v in sorted(models.items())}
        for name, models in evt_models.items()
    }
    (output_dir / "fed_digos_evt_thresholds.json").write_text(json.dumps(thresholds_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "fed_digos_prototypes.json").write_text(json.dumps(prototype_bank.to_payload(), indent=2, sort_keys=True), encoding="utf-8")

    quantiles: dict[str, dict[str, float]] = {}
    overlap: dict[str, Any] = {}
    for col in ["gen_score", "energy_score", "prototype_score", "unknown_score"]:
        known_vals = export_df.loc[export_df["known_or_unknown"] == "known", col].to_numpy(dtype=float)
        unk_vals = export_df.loc[export_df["known_or_unknown"] == "unknown", col].to_numpy(dtype=float)
        quantiles[col] = {}
        for prefix, vals in [("known", known_vals), ("unknown", unk_vals)]:
            vals = vals[np.isfinite(vals)]
            if vals.size:
                for q in [0.50, 0.90, 0.95, 0.99]:
                    quantiles[col][f"{prefix}_q{int(q*100)}"] = float(np.quantile(vals, q))
        if known_vals.size and unk_vals.size:
            known95 = float(np.nanquantile(known_vals, 0.95))
            overlap[col] = {
                "known_q95": known95,
                "unknown_le_known_q95_rate": float(np.mean(unk_vals <= known95)),
                "unknown_gt_known_q95_rate": float(np.mean(unk_vals > known95)),
            }
    (output_dir / "known_unknown_score_quantiles.json").write_text(json.dumps(quantiles, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "score_overlap_report.json").write_text(json.dumps(overlap, indent=2, sort_keys=True), encoding="utf-8")

    unknown_as_normal_before = 0.0
    if unknown_mask.any() and 0 in class_names:
        unknown_as_normal_before = float(np.mean(y_before[unknown_mask] == 0))
    metrics = {
        "openset_backend_fed_digos": 1.0,
        "openset_auroc": auroc,
        "openset_auprc": auprc,
        "openset_fpr95": fpr95,
        "openset_f1_macro": f1_macro,
        "openset_unknown_f1": unknown_f1,
        "openset_known_acc_before": known_acc_before,
        "openset_known_acc": known_acc_after,
        "openset_unknown_recall": unknown_recall,
        "openset_known_false_unknown_rate": known_false_unknown_rate,
        "openset_overall_acc": overall_acc,
        "openset_unknown_as_normal_before_rate": unknown_as_normal_before,
        "openset_rejected_by_gen": float(np.sum(gen_rejects)),
        "openset_rejected_by_energy": float(np.sum(energy_rejects)),
        "openset_rejected_by_prototype": float(np.sum(proto_rejects)),
        "openset_rejected_unknown_by_gen": float(np.sum(np.asarray(gen_rejects)[unknown_mask])) if unknown_mask.any() else 0.0,
        "openset_rejected_unknown_by_energy": float(np.sum(np.asarray(energy_rejects)[unknown_mask])) if unknown_mask.any() else 0.0,
        "openset_rejected_unknown_by_prototype": float(np.sum(np.asarray(proto_rejects)[unknown_mask])) if unknown_mask.any() else 0.0,
        "open_set/auroc": auroc,
        "open_set/auprc": auprc,
        "open_set/fpr95": fpr95,
        "open_set/unknown_f1": unknown_f1,
        "open_set/unknown_detection_rate": unknown_recall,
        "open_set/known_false_unknown_rate": known_false_unknown_rate,
    }
    (output_dir / "open_set_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    log.info(
        "Fed-DiGOS final open-set | AUROC=%.4f AUPRC=%.4f FPR95=%.4f KnownAcc %.4f->%.4f "
        "UnknownRecall=%.4f KnownFU=%.4f unknown_as_Normal_before=%.4f rejects(gen=%d energy=%d proto=%d)",
        auroc, auprc, fpr95, known_acc_before, known_acc_after, unknown_recall,
        known_false_unknown_rate, unknown_as_normal_before, int(np.sum(gen_rejects)),
        int(np.sum(energy_rejects)), int(np.sum(proto_rejects)),
    )
    return metrics
