"""Fed-DiGOS open-set calibration and evaluation.

Fed-DiGOS attaches a disentangled generator branch to the federated student.
The original hard EVT-tail fusion was too brittle for FoT-like unknowns that
sit close to Normal traffic.  This evaluator keeps EVT thresholds for rejection
and reporting, but uses continuous empirical-rank calibration for AUROC
and the default binary decision.  The default scope is global because the smoke
test showed class-wise Normal-tail calibration was still too broad for FoT:

  1. student OSR reconstruction/generative score, high tail,
  2. two-sided student energy deviation, both low and high energy are abnormal,
  3. prototype/OpenMax-style activation distance, high tail.

Final unknown score is the mean of the calibrated ranks.  This preserves useful
middle-range separation instead of zeroing everything below an EVT threshold.
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
EPS = 1.0e-12


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


def _finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _safe_quantile(values: Any, q: float, default: float = np.nan) -> float:
    arr = _finite_array(values)
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def _rank_high(value: float, sorted_values: np.ndarray) -> float:
    """Empirical CDF rank.  Higher score -> more unknown -> larger rank."""
    if not np.isfinite(value) or sorted_values.size == 0:
        return float("nan")
    return float(np.searchsorted(sorted_values, float(value), side="right") / max(sorted_values.size, 1))


def _energy_rank(
    value: float,
    *,
    median: float,
    iqr: float,
    sorted_energy: np.ndarray,
    sorted_neg_energy: np.ndarray,
    sorted_deviations: np.ndarray,
    direction: str,
) -> tuple[float, float]:
    """Return robust energy deviation and configured abnormality rank.

    ``direction=low`` is the default for the current IoTID/FoT split because
    smoke-test diagnostics showed unknown FoT samples have *lower* raw energy.
    ``high`` matches the usual energy-OSR convention; ``two_sided`` is available
    for datasets where both unusually high and low energy are suspicious.
    """
    if not np.isfinite(value):
        return float("nan"), float("nan")
    denom = max(float(iqr), EPS)
    dev = abs(float(value) - float(median)) / denom if np.isfinite(median) else float("nan")
    d = str(direction or "low").lower()
    if d in {"low", "lower", "reversed", "reverse"}:
        return float(dev), _rank_high(-float(value), sorted_neg_energy)
    if d in {"high", "higher"}:
        return float(dev), _rank_high(float(value), sorted_energy)
    return float(dev), _rank_high(float(dev), sorted_deviations)


def _mean_finite(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


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
    nll_weight = float(_cfg_value(cfg, "latent_nll_weight", 0.02))
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


def _prototype_k_for_class(cfg: Any, class_id: int) -> int:
    raw = _nested(cfg, "prototype.num_prototypes_per_class", 16)
    if isinstance(raw, dict):
        return int(raw.get(str(class_id), raw.get(class_id, raw.get("default", 16))))
    # OmegaConf DictConfig behaves like a mapping but may not subclass dict.
    if hasattr(raw, "get") and not isinstance(raw, (int, float, str)):
        try:
            return int(raw.get(str(class_id), raw.get(class_id, raw.get("default", 16))))
        except Exception:
            pass
    return int(raw)


def _fit_prototypes(calib_df: pd.DataFrame, num_classes: int, cfg: Any, log: logging.Logger) -> PrototypeBank:
    enabled = bool(_nested(cfg, "prototype.enabled", True))
    min_per_proto = int(_nested(cfg, "prototype.min_samples_per_prototype", 25))
    prototypes: dict[int, np.ndarray] = {}
    if not enabled:
        return PrototypeBank(prototypes)
    for c in range(num_classes):
        cls = calib_df[(calib_df["y_raw"] == c) & (calib_df["correct_known"] == 1)]
        if cls.empty:
            continue
        feats = np.stack(cls["feature"].to_numpy()).astype(np.float64)
        k_requested = max(1, _prototype_k_for_class(cfg, c))
        k = max(1, min(k_requested, feats.shape[0] // max(min_per_proto, 1)))
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
        log.info(
            "Fed-DiGOS prototypes | class=%d | samples=%d | k=%d requested=%d",
            c, feats.shape[0], centers.shape[0], k_requested,
        )
    return PrototypeBank(prototypes)


def _class_calibration_slice(calib_df: pd.DataFrame, class_id: int, cfg: Any, log: logging.Logger) -> pd.DataFrame:
    min_errors = int(_nested(cfg, "evt.min_errors_per_class", 50))
    fit_correct_only = bool(_nested(cfg, "evt.fit_correct_only", True))
    cls = calib_df[calib_df["y_raw"] == int(class_id)]
    if fit_correct_only:
        cls_fit = cls[cls["correct_known"] == 1]
        if len(cls_fit) < min_errors:
            log.warning(
                "Fed-DiGOS class=%d too few correct calibration samples (%d); using all class samples.",
                class_id, len(cls_fit),
            )
            cls_fit = cls
    else:
        cls_fit = cls
    return cls_fit


def _build_rank_calibrators(
    calib_df: pd.DataFrame,
    *,
    num_classes: int,
    cfg: Any,
    log: logging.Logger,
) -> dict[int, dict[str, Any]]:
    """Build class-wise empirical calibrators and fused-score thresholds."""
    target_fpr = float(_nested(cfg, "evt.target_known_fpr", 0.05))
    min_errors = int(_nested(cfg, "evt.min_errors_per_class", 50))
    generator_col = str(_nested(cfg, "score_fusion.generator_score_column", "recon_error"))
    energy_direction = str(_nested(cfg, "energy.rank_direction", "low"))
    calibration_scope = str(_nested(cfg, "score_fusion.calibration_scope", "global")).lower()
    calibrators: dict[int, dict[str, Any]] = {}

    def build_one(class_key: int, cls_fit: pd.DataFrame) -> None:
        if len(cls_fit) < min_errors:
            log.warning("Fed-DiGOS rank calibration skipped class=%s | samples=%d min=%d", class_key, len(cls_fit), min_errors)
            return
        if generator_col not in cls_fit.columns:
            log.warning("Fed-DiGOS rank calibration requested missing generator score column=%s; falling back to gen_score", generator_col)
            gen_col = "gen_score"
        else:
            gen_col = generator_col
        gen_values = np.sort(_finite_array(cls_fit[gen_col].to_numpy()))
        proto_values = np.sort(_finite_array(cls_fit["prototype_score"].to_numpy()))
        energy_values = _finite_array(cls_fit["energy_score"].to_numpy())
        energy_median = float(np.median(energy_values)) if energy_values.size else 0.0
        q75, q25 = (np.quantile(energy_values, [0.75, 0.25]) if energy_values.size else np.asarray([1.0, 0.0]))
        energy_iqr = float(max(q75 - q25, EPS))
        energy_devs = np.sort(np.abs(energy_values - energy_median) / energy_iqr) if energy_values.size else np.asarray([], dtype=np.float64)
        neg_energy_values = np.sort(-energy_values) if energy_values.size else np.asarray([], dtype=np.float64)
        sorted_energy_values = np.sort(energy_values) if energy_values.size else np.asarray([], dtype=np.float64)

        fused_known: list[float] = []
        for _, row in cls_fit.iterrows():
            _, energy_rank = _energy_rank(
                float(row["energy_score"]),
                median=energy_median,
                iqr=energy_iqr,
                sorted_energy=sorted_energy_values,
                sorted_neg_energy=neg_energy_values,
                sorted_deviations=energy_devs,
                direction=energy_direction,
            )
            ranks = [
                _rank_high(float(row[gen_col]), gen_values),
                _rank_high(float(row["prototype_score"]), proto_values),
                energy_rank,
            ]
            fused_known.append(_mean_finite(ranks))
        fused_values = _finite_array(fused_known)
        fusion_threshold = float(np.quantile(fused_values, 1.0 - target_fpr)) if fused_values.size else 1.0
        calibrators[int(class_key)] = {
            "generator_score_column": gen_col,
            "gen_values": gen_values,
            "prototype_values": proto_values,
            "energy_values": sorted_energy_values,
            "neg_energy_values": neg_energy_values,
            "energy_median": energy_median,
            "energy_iqr": energy_iqr,
            "energy_deviation_values": energy_devs,
            "energy_direction": energy_direction,
            "fusion_known_values": np.sort(fused_values),
            "fusion_threshold": fusion_threshold,
            "n": int(len(cls_fit)),
        }
        log.info(
            "Fed-DiGOS rank calibration | scope=%s class=%s n=%d gen_col=%s gen_q50=%.6g proto_q50=%.6g "
            "energy_dir=%s energy_median=%.6g energy_iqr=%.6g fusion_T=%.6g target_fpr=%.3f",
            calibration_scope,
            class_key,
            len(cls_fit),
            gen_col,
            _safe_quantile(cls_fit[gen_col], 0.50),
            _safe_quantile(cls_fit["prototype_score"], 0.50),
            energy_direction,
            energy_median,
            energy_iqr,
            fusion_threshold,
            target_fpr,
        )

    # Global rank calibration was empirically much stronger on the smoke test
    # because FoT is nearly always predicted as Normal and class-wise Normal
    # thresholds remain too broad.  Class-wise calibrators are still built for
    # diagnostics/fallback, but global is the default score scope.
    if calibration_scope == "global":
        global_fit = calib_df[calib_df["correct_known"] == 1]
        if len(global_fit) < min_errors:
            global_fit = calib_df
        build_one(-1, global_fit)

    for c in range(num_classes):
        cls_fit = _class_calibration_slice(calib_df, c, cfg, log)
        build_one(c, cls_fit)
    return calibrators


def _score_with_rank_calibrator(row: pd.Series, calibrator: dict[str, Any]) -> dict[str, float]:
    gen_col = str(calibrator.get("generator_score_column", "recon_error"))
    if gen_col not in row.index:
        gen_col = "gen_score"
    gen_rank = _rank_high(float(row[gen_col]), calibrator["gen_values"])
    proto_rank = _rank_high(float(row["prototype_score"]), calibrator["prototype_values"])
    energy_dev, energy_rank = _energy_rank(
        float(row["energy_score"]),
        median=float(calibrator["energy_median"]),
        iqr=float(calibrator["energy_iqr"]),
        sorted_energy=calibrator.get("energy_values", np.asarray([], dtype=np.float64)),
        sorted_neg_energy=calibrator.get("neg_energy_values", np.asarray([], dtype=np.float64)),
        sorted_deviations=calibrator.get("energy_deviation_values", np.asarray([], dtype=np.float64)),
        direction=str(calibrator.get("energy_direction", "low")),
    )
    fused = _mean_finite([gen_rank, proto_rank, energy_rank])
    return {
        "gen_rank_score": gen_rank,
        "prototype_rank_score": proto_rank,
        "energy_deviation_score": energy_dev,
        "energy_rank_score": energy_rank,
        "unknown_score": fused,
        "fusion_threshold": float(calibrator.get("fusion_threshold", 1.0)),
    }


def _safe_auc(y_binary: np.ndarray, values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(vals)
    if np.unique(y_binary[mask]).size < 2:
        return 0.0
    return float(roc_auc_score(y_binary[mask], vals[mask]))


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
    min_errors = int(_nested(cfg, "evt.min_errors_per_class", 50))
    prototype_bank = _fit_prototypes(df, num_classes, cfg, log)
    proto_scores = []
    for _, row in df.iterrows():
        proto_scores.append(float(prototype_bank.score(np.asarray(row["feature"]).reshape(1, -1), int(row["y_raw"]))[0]))
    df["prototype_score"] = proto_scores

    models: dict[str, dict[int, EVTModel]] = {"gen": {}, "energy": {}, "prototype": {}, "energy_deviation": {}}
    for c in range(num_classes):
        cls_fit = _class_calibration_slice(df, c, cfg, log)
        if len(cls_fit) < min_errors:
            log.warning("Fed-DiGOS EVT skipped class=%d | samples=%d min=%d", c, len(cls_fit), min_errors)
            continue
        energy_values = _finite_array(cls_fit["energy_score"].to_numpy())
        energy_median = float(np.median(energy_values)) if energy_values.size else 0.0
        q75, q25 = (np.quantile(energy_values, [0.75, 0.25]) if energy_values.size else np.asarray([1.0, 0.0]))
        energy_iqr = float(max(q75 - q25, EPS))
        energy_deviation = np.abs(energy_values - energy_median) / energy_iqr if energy_values.size else np.asarray([], dtype=np.float64)
        score_specs = [
            ("gen", "gen_score"),
            ("energy", "energy_score"),
            ("prototype", "prototype_score"),
        ]
        for name, col in score_specs:
            values = _finite_array(cls_fit[col].to_numpy())
            if values.size >= min_errors:
                models[name][c] = _fit_evt(values, cfg, log)
        if energy_deviation.size >= min_errors:
            models["energy_deviation"][c] = _fit_evt(energy_deviation, cfg, log)
        log.info(
            "Fed-DiGOS EVT calibration | class=%d n=%d gen_q50=%.6g gen_q95=%.6g T_gen=%.6g "
            "energy_q05=%.6g energy_q95=%.6g T_energy_high=%.6g energy_dev_q95=%.6g T_energy_dev=%.6g "
            "proto_q95=%.6g T_proto=%.6g",
            c,
            len(cls_fit),
            _safe_quantile(cls_fit["gen_score"], 0.50),
            _safe_quantile(cls_fit["gen_score"], 0.95),
            float(models["gen"].get(c).decision_threshold if c in models["gen"] else np.nan),
            _safe_quantile(cls_fit["energy_score"], 0.05),
            _safe_quantile(cls_fit["energy_score"], 0.95),
            float(models["energy"].get(c).decision_threshold if c in models["energy"] else np.nan),
            _safe_quantile(energy_deviation, 0.95),
            float(models["energy_deviation"].get(c).decision_threshold if c in models["energy_deviation"] else np.nan),
            _safe_quantile(cls_fit["prototype_score"], 0.95),
            float(models["prototype"].get(c).decision_threshold if c in models["prototype"] else np.nan),
        )

    rank_calibrators = _build_rank_calibrators(df, num_classes=num_classes, cfg=cfg, log=log)
    for c, cal in rank_calibrators.items():
        cls_idx = df.index[df["y_raw"] == c]
        for idx in cls_idx:
            scores = _score_with_rank_calibrator(df.loc[idx], cal)
            for key, val in scores.items():
                df.loc[idx, key] = val

    meta = {
        "backend": "fed_digos",
        "decision_rule": "mean_rank_fusion_with_class_threshold",
        "scores": ["gen_rank", "two_sided_energy_rank", "prototype_rank"],
        "num_classes": num_classes,
        "unknown_label_id": int(unknown_label_id),
        "open_set_label_id": int(_nested(cfg, "open_set_label_id", OPEN_SET_LABEL_ID)),
        "rank_calibration": {
            str(k): {
                "n": int(v.get("n", 0)),
                "energy_median": float(v.get("energy_median", np.nan)),
                "energy_iqr": float(v.get("energy_iqr", np.nan)),
                "fusion_threshold": float(v.get("fusion_threshold", np.nan)),
            }
            for k, v in rank_calibrators.items()
        },
        "thresholds": {
            name: {str(k): v.to_payload() for k, v in sorted(score_models.items())}
            for name, score_models in models.items()
        },
        "prototypes": prototype_bank.to_payload(),
    }
    # Attach non-serializable calibrators via DataFrame attrs for immediate evaluate.
    df.attrs["rank_calibrators"] = rank_calibrators
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

    if calibration_df is not None and "rank_calibrators" in calibration_df.attrs:
        rank_calibrators = calibration_df.attrs["rank_calibrators"]
    elif calibration_df is not None:
        rank_calibrators = _build_rank_calibrators(
            calibration_df,
            num_classes=int(student_model.num_classes),
            cfg=cfg,
            log=log,
        )
    else:
        rank_calibrators = {}
        log.warning("Fed-DiGOS evaluate has no calibration_df; rank-fusion scores will fall back to EVT probabilities.")

    final_preds = []
    unknown_probs = []
    gen_rejects: list[int] = []
    energy_rejects: list[int] = []
    energy_dev_rejects: list[int] = []
    proto_rejects: list[int] = []
    fusion_rejects: list[int] = []
    gen_probs: list[float] = []
    energy_probs: list[float] = []
    energy_dev_probs: list[float] = []
    proto_probs: list[float] = []
    T_gen_used: list[float] = []
    T_energy_used: list[float] = []
    T_energy_dev_used: list[float] = []
    T_proto_used: list[float] = []
    T_fusion_used: list[float] = []
    gen_rank_scores: list[float] = []
    energy_rank_scores: list[float] = []
    energy_dev_scores: list[float] = []
    prototype_rank_scores: list[float] = []

    for _, row in df.iterrows():
        c = int(row["pred_before_osr"])
        evt_rejects: dict[str, bool] = {}
        evt_probs: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        # EVT diagnostics.  Energy high-tail is kept, but final scoring uses two-sided energy rank.
        for name, col in [("gen", "gen_score"), ("energy", "energy_score"), ("prototype", "prototype_score")]:
            model = evt_models.get(name, {}).get(c)
            value = float(row[col])
            if model is None or not np.isfinite(value):
                evt_rejects[name] = False
                evt_probs[name] = 0.0
                thresholds[name] = np.nan
            else:
                evt_rejects[name] = bool(model.is_unknown(value))
                evt_probs[name] = float(model.predict_probability_unknown(value))
                thresholds[name] = float(model.decision_threshold if model.decision_threshold is not None else np.nan)

        calibration_scope = str(_nested(cfg, "score_fusion.calibration_scope", "global")).lower()
        cal = rank_calibrators.get(-1) if calibration_scope == "global" else rank_calibrators.get(c)
        if cal is None:
            cal = rank_calibrators.get(c)
        if cal is not None:
            rank_scores = _score_with_rank_calibrator(row, cal)
            unknown_score = float(rank_scores["unknown_score"])
            fusion_threshold = float(rank_scores["fusion_threshold"])
            gen_rank = float(rank_scores["gen_rank_score"])
            proto_rank = float(rank_scores["prototype_rank_score"])
            energy_dev = float(rank_scores["energy_deviation_score"])
            energy_rank = float(rank_scores["energy_rank_score"])
        else:
            # Last-resort fallback; keep the run alive but make the log loud.
            unknown_score = float(max(evt_probs.values())) if evt_probs else 0.0
            fusion_threshold = 1.0 - float(_nested(cfg, "evt.target_known_fpr", 0.05))
            gen_rank = evt_probs.get("gen", 0.0)
            proto_rank = evt_probs.get("prototype", 0.0)
            energy_dev = np.nan
            energy_rank = evt_probs.get("energy", 0.0)

        energy_dev_model = evt_models.get("energy_deviation", {}).get(c)
        if energy_dev_model is not None and np.isfinite(energy_dev):
            energy_dev_reject = bool(energy_dev_model.is_unknown(energy_dev))
            energy_dev_prob = float(energy_dev_model.predict_probability_unknown(energy_dev))
            T_energy_dev = float(energy_dev_model.decision_threshold if energy_dev_model.decision_threshold is not None else np.nan)
        else:
            energy_dev_reject = False
            energy_dev_prob = float(energy_rank) if np.isfinite(energy_rank) else 0.0
            T_energy_dev = np.nan

        fusion_reject = bool(unknown_score > fusion_threshold)
        final_reject = fusion_reject
        final_preds.append(open_set_label_id if final_reject else c)
        unknown_probs.append(float(unknown_score))
        gen_rejects.append(int(evt_rejects.get("gen", False)))
        energy_rejects.append(int(evt_rejects.get("energy", False)))
        energy_dev_rejects.append(int(energy_dev_reject))
        proto_rejects.append(int(evt_rejects.get("prototype", False)))
        fusion_rejects.append(int(fusion_reject))
        gen_probs.append(float(evt_probs.get("gen", 0.0)))
        energy_probs.append(float(evt_probs.get("energy", 0.0)))
        energy_dev_probs.append(float(energy_dev_prob))
        proto_probs.append(float(evt_probs.get("prototype", 0.0)))
        T_gen_used.append(thresholds.get("gen", np.nan))
        T_energy_used.append(thresholds.get("energy", np.nan))
        T_energy_dev_used.append(T_energy_dev)
        T_proto_used.append(thresholds.get("prototype", np.nan))
        T_fusion_used.append(fusion_threshold)
        gen_rank_scores.append(gen_rank)
        energy_rank_scores.append(energy_rank)
        energy_dev_scores.append(energy_dev)
        prototype_rank_scores.append(proto_rank)

    y_pred = np.asarray(final_preds, dtype=int)
    score_arr = np.asarray(unknown_probs, dtype=float)
    if np.unique(y_binary).size < 2:
        auroc = 0.0
        auprc = 0.0
        fpr95 = 1.0
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
    df["gen_evt_prob"] = gen_probs
    df["energy_evt_prob"] = energy_probs
    df["energy_deviation_evt_prob"] = energy_dev_probs
    df["prototype_evt_prob"] = proto_probs
    df["gen_rank_score"] = gen_rank_scores
    df["energy_deviation_score"] = energy_dev_scores
    df["energy_rank_score"] = energy_rank_scores
    df["prototype_rank_score"] = prototype_rank_scores
    df["unknown_score"] = score_arr
    df["T_gen_used"] = T_gen_used
    df["T_energy_used"] = T_energy_used
    df["T_energy_deviation_used"] = T_energy_dev_used
    df["T_proto_used"] = T_proto_used
    df["T_fusion_used"] = T_fusion_used
    df["gen_reject"] = gen_rejects
    df["energy_reject"] = energy_rejects
    df["energy_deviation_reject"] = energy_dev_rejects
    df["prototype_reject"] = proto_rejects
    df["fusion_reject"] = fusion_rejects
    df["final_reject"] = (y_pred == open_set_label_id).astype(int)
    df["known_or_unknown"] = np.where(y_binary == 1, "unknown", "known")
    export_df = df.drop(columns=["feature"])
    export_df.to_csv(output_dir / "open_set_scores.csv", index=False)

    if calibration_df is not None:
        calibration_df.drop(columns=["feature"], errors="ignore").to_csv(output_dir / "fed_digos_calibration_scores.csv", index=False)
    thresholds_payload = {
        name: {str(k): v.to_payload() for k, v in sorted(models.items())}
        for name, models in evt_models.items()
    }
    rank_thresholds_payload = {
        str(k): {
            "fusion_threshold": float(v.get("fusion_threshold", np.nan)),
            "energy_median": float(v.get("energy_median", np.nan)),
            "energy_iqr": float(v.get("energy_iqr", np.nan)),
            "n": int(v.get("n", 0)),
        }
        for k, v in rank_calibrators.items()
    }
    (output_dir / "fed_digos_evt_thresholds.json").write_text(json.dumps(thresholds_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "fed_digos_rank_calibration.json").write_text(json.dumps(rank_thresholds_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "fed_digos_prototypes.json").write_text(json.dumps(prototype_bank.to_payload(), indent=2, sort_keys=True), encoding="utf-8")

    quantiles: dict[str, dict[str, float]] = {}
    overlap: dict[str, Any] = {}
    quantile_cols = [
        "gen_score", "recon_error", "latent_nll", "energy_score", "energy_deviation_score",
        "prototype_score", "gen_rank_score", "energy_rank_score", "prototype_rank_score", "unknown_score",
    ]
    for col in quantile_cols:
        if col not in export_df.columns:
            continue
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

    raw_aurocs = {
        "auroc_unknown_score_rank_fusion": auroc,
        "auroc_gen_score": _safe_auc(y_binary, export_df["gen_score"].to_numpy(dtype=float)),
        "auroc_recon_error": _safe_auc(y_binary, export_df["recon_error"].to_numpy(dtype=float)),
        "auroc_prototype_score": _safe_auc(y_binary, export_df["prototype_score"].to_numpy(dtype=float)),
        "auroc_energy_score_high": _safe_auc(y_binary, export_df["energy_score"].to_numpy(dtype=float)),
        "auroc_energy_score_reversed": _safe_auc(y_binary, -export_df["energy_score"].to_numpy(dtype=float)),
        "auroc_energy_deviation": _safe_auc(y_binary, export_df["energy_deviation_score"].to_numpy(dtype=float)),
        "auroc_gen_rank": _safe_auc(y_binary, export_df["gen_rank_score"].to_numpy(dtype=float)),
        "auroc_energy_rank": _safe_auc(y_binary, export_df["energy_rank_score"].to_numpy(dtype=float)),
        "auroc_prototype_rank": _safe_auc(y_binary, export_df["prototype_rank_score"].to_numpy(dtype=float)),
    }
    (output_dir / "fed_digos_component_aurocs.json").write_text(json.dumps(raw_aurocs, indent=2, sort_keys=True), encoding="utf-8")

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
        "openset_rejected_by_gen_evt": float(np.sum(gen_rejects)),
        "openset_rejected_by_energy_evt": float(np.sum(energy_rejects)),
        "openset_rejected_by_energy_deviation_evt": float(np.sum(energy_dev_rejects)),
        "openset_rejected_by_prototype_evt": float(np.sum(proto_rejects)),
        "openset_rejected_by_fusion": float(np.sum(fusion_rejects)),
        "openset_rejected_unknown_by_fusion": float(np.sum(np.asarray(fusion_rejects)[unknown_mask])) if unknown_mask.any() else 0.0,
        "open_set/auroc": auroc,
        "open_set/auprc": auprc,
        "open_set/fpr95": fpr95,
        "open_set/unknown_f1": unknown_f1,
        "open_set/unknown_detection_rate": unknown_recall,
        "open_set/known_false_unknown_rate": known_false_unknown_rate,
        **raw_aurocs,
    }
    # Backward-compatible aliases for old dashboards.
    metrics["openset_rejected_by_gen"] = metrics["openset_rejected_by_gen_evt"]
    metrics["openset_rejected_by_energy"] = metrics["openset_rejected_by_energy_evt"]
    metrics["openset_rejected_by_prototype"] = metrics["openset_rejected_by_prototype_evt"]
    (output_dir / "open_set_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    log.info(
        "Fed-DiGOS final open-set | AUROC=%.4f AUPRC=%.4f FPR95=%.4f KnownAcc %.4f->%.4f "
        "UnknownRecall=%.4f KnownFU=%.4f unknown_as_Normal_before=%.4f "
        "rawAUROC(gen=%.4f recon=%.4f energy_high=%.4f energy_rev=%.4f energy_dev=%.4f proto=%.4f) "
        "rejects(fusion=%d gen_evt=%d energy_evt=%d energy_dev_evt=%d proto_evt=%d)",
        auroc, auprc, fpr95, known_acc_before, known_acc_after, unknown_recall,
        known_false_unknown_rate, unknown_as_normal_before,
        raw_aurocs["auroc_gen_score"], raw_aurocs["auroc_recon_error"],
        raw_aurocs["auroc_energy_score_high"], raw_aurocs["auroc_energy_score_reversed"],
        raw_aurocs["auroc_energy_deviation"], raw_aurocs["auroc_prototype_score"],
        int(np.sum(fusion_rejects)), int(np.sum(gen_rejects)), int(np.sum(energy_rejects)),
        int(np.sum(energy_dev_rejects)), int(np.sum(proto_rejects)),
    )
    return metrics
