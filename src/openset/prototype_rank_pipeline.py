"""Known-only Prototype-Rank calibration and evaluation for FedTROS-PR.

The canonical detector operates on a frozen federated-student representation. The
default feature source is the normalized deterministic penultimate student embedding;
an optional dedicated student OSR representation can be selected only for the A5
feature-source gate. Positive prototypes and known-derived boundary prototypes are fit
from a prototype-fit subset of known data, while empirical-rank/threshold calibration
uses a disjoint known-only calibration subset. No final unknown sample is used to fit
preprocessing, prototypes, rank distributions, or the operating threshold.

The module also exposes matched post-hoc detector variants (MSP, Energy, positive-only
prototype, raw positive+boundary score, and full empirical-rank Prototype-Rank) so A4
can isolate which rejection components actually earn their complexity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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


logger = logging.getLogger("PrototypeRank")
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


def _l2_normalize_np(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        denom = max(float(np.linalg.norm(arr)), eps)
        return arr / denom
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(denom, eps)


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


def _mean_finite(values: list[float], weights: list[float] | None = None) -> float:
    finite_vals: list[float] = []
    finite_weights: list[float] = []
    for idx, value in enumerate(values):
        if np.isfinite(value):
            finite_vals.append(float(value))
            if weights is not None and idx < len(weights) and np.isfinite(weights[idx]):
                finite_weights.append(max(float(weights[idx]), 0.0))
            else:
                finite_weights.append(1.0)
    if not finite_vals:
        return 0.0
    vals = np.asarray(finite_vals, dtype=np.float64)
    w = np.asarray(finite_weights, dtype=np.float64)
    if float(w.sum()) <= EPS:
        return float(np.mean(vals))
    return float(np.average(vals, weights=w))


def _rank_threshold(values: list[float], target_fpr: float) -> float:
    arr = _finite_array(values)
    if arr.size == 0:
        return 1.0
    return float(np.quantile(arr, 1.0 - float(target_fpr)))


def _score_weights(cfg: Any) -> list[float]:
    weights = _nested(cfg, "score_fusion.weights", None)
    if weights is None:
        return [1.0, 1.0, 1.0, 1.0]
    try:
        return [
            float(getattr(weights, "proser", 1.0)),
            float(getattr(weights, "generator", 1.0)),
            float(getattr(weights, "energy", 1.0)),
            float(getattr(weights, "prototype", 1.0)),
        ]
    except Exception:
        return [1.0, 1.0, 1.0, 1.0]


def _fuse_rank_scores(proser_rank: float, gen_rank: float, energy_rank: float, proto_rank: float, cfg: Any, *, msp_rank: float = float("nan")) -> float:
    method = str(_nested(cfg, "score_fusion.method", "prototype_rank")).lower()
    if method in {"msp", "msp_rank", "maxsoftmax", "max_softmax"}:
        return float(msp_rank) if np.isfinite(msp_rank) else 0.0
    if method in {"proser_rank", "placeholder_rank", "proser", "placeholder", "proser_only"}:
        return float(proser_rank) if np.isfinite(proser_rank) else 0.0
    if method in {"generator_rank", "gen_rank", "generator", "gen_only", "generator_only"}:
        return float(gen_rank) if np.isfinite(gen_rank) else 0.0
    if method in {"energy_rank", "energy", "energy_only"}:
        return float(energy_rank) if np.isfinite(energy_rank) else 0.0
    if method in {"prototype_rank", "proto_rank", "prototype", "proto_only", "prototype_only"}:
        return float(proto_rank) if np.isfinite(proto_rank) else 0.0
    if method in {"max_rank", "max"}:
        return float(np.nanmax([proser_rank, gen_rank, energy_rank, proto_rank]))
    if method in {"weighted_rank", "weighted_mean_rank"}:
        return _mean_finite([proser_rank, gen_rank, energy_rank, proto_rank], weights=_score_weights(cfg))
    return _mean_finite([proser_rank, gen_rank, energy_rank, proto_rank])


def _component_threshold_key(cfg: Any) -> str:
    method = str(_nested(cfg, "score_fusion.method", "prototype_rank")).lower()
    if method in {"msp", "msp_rank", "maxsoftmax", "max_softmax"}:
        return "msp_threshold"
    if method in {"proser_rank", "placeholder_rank", "proser", "placeholder", "proser_only"}:
        return "proser_threshold"
    if method in {"generator_rank", "gen_rank", "generator", "gen_only", "generator_only"}:
        return "gen_threshold"
    if method in {"energy_rank", "energy", "energy_only"}:
        return "energy_threshold"
    if method in {"prototype_rank", "proto_rank", "prototype", "proto_only", "prototype_only"}:
        return "prototype_threshold"
    return "fusion_threshold"



@dataclass
class PrototypeBank:
    """Positive/negative prototype bank for Prototype-Rank diagnostics.

    Earlier prototypes used positive-only KMeans over the closed-set backbone,
    which drifted as the classifier became more confident.  This bank defaults
    to OSR latent features, L2 normalization, class radii, and optional negative
    boundary prototypes generated from known-only calibration features.
    """

    prototypes: dict[int, np.ndarray]
    radii: dict[int, float] | None = None
    negative_prototypes: np.ndarray | None = None
    negative_radius: float = 1.0
    normalize: bool = True
    negative_weight: float = 0.0
    eps: float = 1.0e-8

    def _prep(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return _l2_normalize_np(x, self.eps) if self.normalize else x

    def score(self, features: np.ndarray, class_id: int) -> np.ndarray:
        p = self.prototypes.get(int(class_id))
        if p is None or p.size == 0:
            return np.full((features.shape[0],), np.nan, dtype=np.float64)
        x = self._prep(features)
        centers = self._prep(p) if self.normalize else np.asarray(p, dtype=np.float64)
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        pos_dist = np.sqrt(np.min(d, axis=1) + self.eps)
        radius = float((self.radii or {}).get(int(class_id), 1.0))
        pos_score = pos_dist / max(radius, self.eps)

        if self.negative_prototypes is None or self.negative_prototypes.size == 0 or self.negative_weight <= 0.0:
            return pos_score.astype(np.float64)
        neg_centers = self._prep(self.negative_prototypes) if self.normalize else np.asarray(self.negative_prototypes, dtype=np.float64)
        nd = ((x[:, None, :] - neg_centers[None, :, :]) ** 2).sum(axis=2)
        neg_dist = np.sqrt(np.min(nd, axis=1) + self.eps)
        # High when a sample is inside/near the known-only synthetic boundary region.
        neg_close = np.maximum(0.0, (float(self.negative_radius) - neg_dist) / max(float(self.negative_radius), self.eps))
        return (pos_score + (float(self.negative_weight) * neg_close)).astype(np.float64)

    def to_payload(self) -> dict[str, Any]:
        return {
            "positive_prototypes": {str(k): v.tolist() for k, v in sorted(self.prototypes.items())},
            "positive_radii": {str(k): float(v) for k, v in sorted((self.radii or {}).items())},
            "negative_prototypes": self.negative_prototypes.tolist() if self.negative_prototypes is not None else [],
            "negative_radius": float(self.negative_radius),
            "negative_weight": float(self.negative_weight),
            "normalize": bool(self.normalize),
        }


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
    """Collect detector inputs without assuming the optional OSR branch exists.

    The canonical Prototype-Rank feature source is ``student_embedding`` unless the
    A5 feature-source gate explicitly selects the dedicated OSR representation.  MSP
    and Energy therefore remain valid on a plain architecture-matched student.
    """
    nll_weight = float(_cfg_value(cfg, "latent_nll_weight", 0.02))
    temperature = float(_nested(cfg, "energy.temperature", 1.0))
    proto_source_cfg = str(_nested(cfg, "prototype.feature_source", "student_embedding")).lower()
    use_osr_feature = proto_source_cfg in {"osr_mu", "osr_embedding", "student_osr", "osr"}
    osr_available = bool(getattr(student_model, "osr_enabled", False)) and hasattr(student_model, "osr_score")
    if use_osr_feature and not osr_available:
        raise RuntimeError(
            "Prototype-Rank feature_source requires the optional student OSR branch, "
            "but that branch is disabled or unavailable. Use feature_source=student_embedding."
        )

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
            probs = torch.softmax(logits, dim=1)
            msp_score = 1.0 - torch.max(probs, dim=1).values  # larger = more suspicious
            cond = pred if class_condition == "pred" else y.clamp(0, int(student_model.num_classes) - 1)

            energy = student_model.energy_score(logits, temperature=temperature)

            osr = None
            if osr_available:
                osr = student_model.osr_score(x, cond, nll_weight=nll_weight, detach_features=True)

            if bool(getattr(student_model, "open_set_enabled", False)) and getattr(student_model, "open_set_head", None) is not None:
                proser = student_model.open_set_score(
                    x,
                    detach_features=True,
                    score_type=str(_nested(cfg, "proser.score_type", "margin")),
                )
                proser_score = proser["score"]
                proser_prob = proser["unknown_prob"]
                proser_margin = proser["unknown_margin"]
            else:
                proser_score = torch.zeros_like(energy)
                proser_prob = torch.zeros_like(energy)
                proser_margin = torch.zeros_like(energy)

            h_np = h.detach().cpu().numpy()
            if osr is not None:
                mu_np = osr["mu"].detach().cpu().numpy()
                gen_score = osr["score"].detach().cpu().numpy()
                recon_error = osr["recon_error"].detach().cpu().numpy()
                latent_nll = osr["latent_nll"].detach().cpu().numpy()
            else:
                mu_np = None
                gen_score = np.full((x.shape[0],), np.nan, dtype=np.float64)
                recon_error = np.full((x.shape[0],), np.nan, dtype=np.float64)
                latent_nll = np.full((x.shape[0],), np.nan, dtype=np.float64)

            if use_osr_feature:
                assert mu_np is not None
                proto_np = mu_np
                proto_source = "osr_mu"
            else:
                proto_np = h_np
                proto_source = "student_embedding"

            msp_np = msp_score.detach().cpu().numpy()
            energy_np = energy.detach().cpu().numpy()
            proser_score_np = proser_score.detach().cpu().numpy()
            proser_prob_np = proser_prob.detach().cpu().numpy()
            proser_margin_np = proser_margin.detach().cpu().numpy()

            for i in range(x.shape[0]):
                rows.append({
                    "sample_id": int(offset + i),
                    "y_raw": int(y[i].item()),
                    "pred_before_osr": int(pred[i].item()),
                    "condition_class": int(cond[i].item()),
                    "msp_score": float(msp_np[i]),
                    "gen_score": float(gen_score[i]),
                    "recon_error": float(recon_error[i]),
                    "latent_nll": float(latent_nll[i]),
                    "energy_score": float(energy_np[i]),
                    "proser_score": float(proser_score_np[i]),
                    "proser_unknown_prob": float(proser_prob_np[i]),
                    "proser_unknown_margin": float(proser_margin_np[i]),
                    "correct_known": int((y[i].item() >= 0) and (y[i].item() == pred[i].item())),
                    "prototype_feature_source": proto_source,
                    "feature": proto_np[i],
                    "student_embedding_norm": float(np.linalg.norm(h_np[i])),
                    "osr_mu_norm": float(np.linalg.norm(mu_np[i])) if mu_np is not None else float("nan"),
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


def _make_negative_boundary_features(
    features_by_class: dict[int, np.ndarray],
    *,
    max_samples: int,
    mixup_alpha: float,
    noise_std: float,
    normalize: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    classes = [c for c, feats in features_by_class.items() if feats.size > 0]
    if len(classes) < 2 or max_samples <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    dim = next(iter(features_by_class.values())).shape[1]
    out = np.zeros((max_samples, dim), dtype=np.float64)
    alpha = max(float(mixup_alpha), 1.0e-3)
    for idx in range(max_samples):
        c1, c2 = rng.choice(classes, size=2, replace=False)
        a = features_by_class[int(c1)][rng.integers(0, features_by_class[int(c1)].shape[0])]
        b = features_by_class[int(c2)][rng.integers(0, features_by_class[int(c2)].shape[0])]
        lam = rng.beta(alpha, alpha)
        out[idx] = (lam * a) + ((1.0 - lam) * b)
    if float(noise_std) > 0.0:
        out += rng.normal(0.0, float(noise_std), size=out.shape)
    return _l2_normalize_np(out) if normalize else out


def _fit_prototypes(calib_df: pd.DataFrame, num_classes: int, cfg: Any, log: logging.Logger) -> PrototypeBank:
    enabled = bool(_nested(cfg, "prototype.enabled", True))
    min_per_proto = int(_nested(cfg, "prototype.min_samples_per_prototype", 25))
    normalize = bool(_nested(cfg, "prototype.normalize", True))
    radius_q = float(_nested(cfg, "prototype.radius_quantile", 0.95))
    prototypes: dict[int, np.ndarray] = {}
    radii: dict[int, float] = {}
    features_by_class: dict[int, np.ndarray] = {}
    if not enabled:
        return PrototypeBank(prototypes, radii={}, normalize=normalize)
    feature_source = str(_nested(cfg, "prototype.feature_source", "osr_mu"))
    for c in range(num_classes):
        cls = calib_df[(calib_df["y_raw"] == c) & (calib_df["correct_known"] == 1)]
        if cls.empty:
            cls = calib_df[calib_df["y_raw"] == c]
        if cls.empty:
            continue
        feats = np.stack(cls["feature"].to_numpy()).astype(np.float64)
        feats = _l2_normalize_np(feats) if normalize else feats
        features_by_class[c] = feats
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
                log.warning("Prototype-Rank prototype KMeans failed for class=%d: %s; using mean", c, exc)
                centers = feats.mean(axis=0, keepdims=True)
        centers = _l2_normalize_np(centers) if normalize else centers.astype(np.float64)
        d = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        nearest = np.sqrt(np.min(d, axis=1) + EPS)
        radius = float(max(np.quantile(nearest, min(max(radius_q, 0.50), 0.999)), EPS))
        prototypes[c] = centers.astype(np.float64)
        radii[c] = radius
        log.info(
            "Prototype-Rank positive prototypes | feature_source=%s normalize=%s class=%d samples=%d "
            "k=%d requested=%d radius_q=%.2f radius=%.6g feat_norm_q50=%.6g",
            feature_source,
            normalize,
            c,
            feats.shape[0],
            centers.shape[0],
            k_requested,
            radius_q,
            radius,
            _safe_quantile(np.linalg.norm(feats, axis=1), 0.50),
        )

    negative_enabled = bool(_nested(cfg, "prototype.negative.enabled", True))
    negative_prototypes = None
    negative_radius = 1.0
    negative_weight = float(_nested(cfg, "prototype.negative.weight", 0.35)) if negative_enabled else 0.0
    if negative_enabled and len(features_by_class) >= 2:
        rng = np.random.default_rng(int(_nested(cfg, "prototype.negative.random_seed", 42)))
        max_negative = int(_nested(cfg, "prototype.negative.max_samples", 5000))
        mixup_alpha = float(_nested(cfg, "prototype.negative.mixup_alpha", 1.0))
        noise_std = float(_nested(cfg, "prototype.negative.noise_std", 0.005))
        boundary = _make_negative_boundary_features(
            features_by_class,
            max_samples=max_negative,
            mixup_alpha=mixup_alpha,
            noise_std=noise_std,
            normalize=normalize,
            rng=rng,
        )
        if boundary.size > 0:
            k_neg_req = int(_nested(cfg, "prototype.negative.num_prototypes", 32))
            k_neg = max(1, min(k_neg_req, boundary.shape[0] // max(min_per_proto, 1)))
            if k_neg <= 1:
                neg_centers = boundary.mean(axis=0, keepdims=True)
            else:
                try:
                    km = KMeans(n_clusters=k_neg, n_init=5, random_state=43)
                    km.fit(boundary)
                    neg_centers = km.cluster_centers_
                except Exception as exc:
                    log.warning("Prototype-Rank negative prototype KMeans failed: %s; using mean", exc)
                    neg_centers = boundary.mean(axis=0, keepdims=True)
            neg_centers = _l2_normalize_np(neg_centers) if normalize else neg_centers.astype(np.float64)
            nd = ((boundary[:, None, :] - neg_centers[None, :, :]) ** 2).sum(axis=2)
            nearest_neg = np.sqrt(np.min(nd, axis=1) + EPS)
            neg_q = float(_nested(cfg, "prototype.negative.radius_quantile", 0.75))
            negative_radius = float(max(np.quantile(nearest_neg, min(max(neg_q, 0.50), 0.999)), EPS))
            negative_prototypes = neg_centers.astype(np.float64)
            log.info(
                "Prototype-Rank negative prototypes | enabled=true samples=%d k=%d requested=%d radius=%.6g "
                "weight=%.3f mixup_alpha=%.3f noise_std=%.4f",
                boundary.shape[0],
                negative_prototypes.shape[0],
                k_neg_req,
                negative_radius,
                negative_weight,
                mixup_alpha,
                noise_std,
            )
    return PrototypeBank(
        prototypes,
        radii=radii,
        negative_prototypes=negative_prototypes,
        negative_radius=negative_radius,
        normalize=normalize,
        negative_weight=negative_weight,
    )


def _class_calibration_slice(calib_df: pd.DataFrame, class_id: int, cfg: Any, log: logging.Logger) -> pd.DataFrame:
    min_errors = int(_nested(cfg, "calibration.min_samples_per_class", 50))
    fit_correct_only = bool(_nested(cfg, "calibration.fit_correct_only", True))
    cls = calib_df[calib_df["y_raw"] == int(class_id)]
    if fit_correct_only:
        cls_fit = cls[cls["correct_known"] == 1]
        if len(cls_fit) < min_errors:
            log.warning(
                "Prototype-Rank class=%d too few correct calibration samples (%d); using all class samples.",
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
    target_fpr = float(_nested(cfg, "calibration.target_known_fpr", 0.05))
    min_errors = int(_nested(cfg, "calibration.min_samples_per_class", 50))
    generator_col = str(_nested(cfg, "score_fusion.generator_score_column", "recon_error"))
    energy_direction = str(_nested(cfg, "energy.rank_direction", "low"))
    calibration_scope = str(_nested(cfg, "score_fusion.calibration_scope", "global")).lower()
    calibrators: dict[int, dict[str, Any]] = {}

    def build_one(class_key: int, cls_fit: pd.DataFrame) -> None:
        if len(cls_fit) < min_errors:
            log.warning("Prototype-Rank rank calibration skipped class=%s | samples=%d min=%d", class_key, len(cls_fit), min_errors)
            return
        if generator_col not in cls_fit.columns:
            log.warning("Prototype-Rank rank calibration requested missing generator score column=%s; falling back to gen_score", generator_col)
            gen_col = "gen_score"
        else:
            gen_col = generator_col
        gen_values = np.sort(_finite_array(cls_fit[gen_col].to_numpy()))
        msp_values = np.sort(_finite_array(cls_fit["msp_score"].to_numpy())) if "msp_score" in cls_fit.columns else np.asarray([], dtype=np.float64)
        proser_values = np.sort(_finite_array(cls_fit["proser_score"].to_numpy())) if "proser_score" in cls_fit.columns else np.asarray([], dtype=np.float64)
        proto_values = np.sort(_finite_array(cls_fit["prototype_score"].to_numpy()))
        energy_values = _finite_array(cls_fit["energy_score"].to_numpy())
        energy_median = float(np.median(energy_values)) if energy_values.size else 0.0
        q75, q25 = (np.quantile(energy_values, [0.75, 0.25]) if energy_values.size else np.asarray([1.0, 0.0]))
        energy_iqr = float(max(q75 - q25, EPS))
        energy_devs = np.sort(np.abs(energy_values - energy_median) / energy_iqr) if energy_values.size else np.asarray([], dtype=np.float64)
        neg_energy_values = np.sort(-energy_values) if energy_values.size else np.asarray([], dtype=np.float64)
        sorted_energy_values = np.sort(energy_values) if energy_values.size else np.asarray([], dtype=np.float64)

        fused_known: list[float] = []
        mean_rank_known: list[float] = []
        weighted_rank_known: list[float] = []
        max_rank_known: list[float] = []
        msp_rank_known: list[float] = []
        proser_rank_known: list[float] = []
        gen_rank_known: list[float] = []
        energy_rank_known: list[float] = []
        proto_rank_known: list[float] = []
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
            msp_rank = _rank_high(float(row.get("msp_score", np.nan)), msp_values)
            proser_rank = _rank_high(float(row.get("proser_score", 0.0)), proser_values)
            gen_rank = _rank_high(float(row[gen_col]), gen_values)
            proto_rank = _rank_high(float(row["prototype_score"]), proto_values)
            msp_rank_known.append(msp_rank)
            proser_rank_known.append(proser_rank)
            gen_rank_known.append(gen_rank)
            energy_rank_known.append(energy_rank)
            proto_rank_known.append(proto_rank)
            ranks4 = [proser_rank, gen_rank, energy_rank, proto_rank]
            fused_known.append(_fuse_rank_scores(proser_rank, gen_rank, energy_rank, proto_rank, cfg, msp_rank=msp_rank))
            mean_rank_known.append(_mean_finite(ranks4))
            weighted_rank_known.append(_mean_finite(ranks4, weights=_score_weights(cfg)))
            max_rank_known.append(float(np.nanmax(ranks4)))
        fused_values = _finite_array(fused_known)
        fusion_threshold = _rank_threshold(fused_known, target_fpr)
        mean_threshold = _rank_threshold(mean_rank_known, target_fpr)
        weighted_threshold = _rank_threshold(weighted_rank_known, target_fpr)
        max_threshold = _rank_threshold(max_rank_known, target_fpr)
        msp_threshold = _rank_threshold(msp_rank_known, target_fpr)
        proser_threshold = _rank_threshold(proser_rank_known, target_fpr)
        gen_threshold = _rank_threshold(gen_rank_known, target_fpr)
        energy_threshold = _rank_threshold(energy_rank_known, target_fpr)
        proto_threshold = _rank_threshold(proto_rank_known, target_fpr)
        prototype_raw_threshold = _safe_quantile(proto_values, 1.0 - target_fpr, default=float("inf"))
        calibrators[int(class_key)] = {
            "generator_score_column": gen_col,
            "gen_values": gen_values,
            "msp_values": msp_values,
            "proser_values": proser_values,
            "prototype_values": proto_values,
            "energy_values": sorted_energy_values,
            "neg_energy_values": neg_energy_values,
            "energy_median": energy_median,
            "energy_iqr": energy_iqr,
            "energy_deviation_values": energy_devs,
            "energy_direction": energy_direction,
            "fusion_known_values": np.sort(fused_values),
            "msp_threshold": msp_threshold,
            "proser_threshold": proser_threshold,
            "gen_threshold": gen_threshold,
            "energy_threshold": energy_threshold,
            "prototype_threshold": proto_threshold,
            "prototype_raw_threshold": prototype_raw_threshold,
            "fusion_threshold": fusion_threshold,
            "mean_threshold": mean_threshold,
            "weighted_threshold": weighted_threshold,
            "max_threshold": max_threshold,
            "n": int(len(cls_fit)),
        }
        log.info(
            "Prototype-Rank rank calibration | scope=%s class=%s n=%d method=%s gen_col=%s "
            "proser_q50=%.6g gen_q50=%.6g proto_q50=%.6g energy_dir=%s energy_median=%.6g energy_iqr=%.6g "
            "T_proser_rank=%.6g T_gen_rank=%.6g T_energy_rank=%.6g T_proto_rank=%.6g T_fusion=%.6g target_fpr=%.3f",
            calibration_scope,
            class_key,
            len(cls_fit),
            str(_nested(cfg, "score_fusion.method", "proser_rank")),
            gen_col,
            _safe_quantile(cls_fit["proser_score"], 0.50) if "proser_score" in cls_fit.columns else float("nan"),
            _safe_quantile(cls_fit[gen_col], 0.50),
            _safe_quantile(cls_fit["prototype_score"], 0.50),
            energy_direction,
            energy_median,
            energy_iqr,
            proser_threshold,
            gen_threshold,
            energy_threshold,
            proto_threshold,
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
    msp_rank = _rank_high(float(row.get("msp_score", np.nan)), calibrator.get("msp_values", np.asarray([], dtype=np.float64)))
    proser_rank = _rank_high(float(row.get("proser_score", 0.0)), calibrator.get("proser_values", np.asarray([], dtype=np.float64)))
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
    # ``unknown_score`` is set later by _fuse_rank_scores because the configured
    # method lives in cfg.  Keep this helper focused on component scores.
    return {
        "msp_rank_score": msp_rank,
        "proser_rank_score": proser_rank,
        "gen_rank_score": gen_rank,
        "prototype_rank_score": proto_rank,
        "energy_deviation_score": energy_dev,
        "energy_rank_score": energy_rank,
        "msp_threshold": float(calibrator.get("msp_threshold", 1.0)),
        "proser_threshold": float(calibrator.get("proser_threshold", 1.0)),
        "gen_threshold": float(calibrator.get("gen_threshold", 1.0)),
        "energy_threshold": float(calibrator.get("energy_threshold", 1.0)),
        "prototype_threshold": float(calibrator.get("prototype_threshold", 1.0)),
        "prototype_raw_threshold": float(calibrator.get("prototype_raw_threshold", float("inf"))),
        "fusion_threshold": float(calibrator.get("fusion_threshold", 1.0)),
        "mean_threshold": float(calibrator.get("mean_threshold", calibrator.get("fusion_threshold", 1.0))),
        "weighted_threshold": float(calibrator.get("weighted_threshold", calibrator.get("fusion_threshold", 1.0))),
        "max_threshold": float(calibrator.get("max_threshold", calibrator.get("fusion_threshold", 1.0))),
    }


def _safe_auc(y_binary: np.ndarray, values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(vals)
    if np.unique(y_binary[mask]).size < 2:
        return 0.0
    return float(roc_auc_score(y_binary[mask], vals[mask]))


def _select_rank_score_and_threshold(
    row: pd.Series,
    calibrator: dict[str, Any],
    cfg: Any,
) -> tuple[float, float, dict[str, float]]:
    """Return the configured detector score and known-only operating threshold.

    All thresholds are derived from the disjoint known-only calibration subset.
    No EVT/Weibull fallback is used by the canonical FedTROS-PR implementation.
    """
    scores = _score_with_rank_calibrator(row, calibrator)
    msp_rank = float(scores["msp_rank_score"])
    proser_rank = float(scores["proser_rank_score"])
    gen_rank = float(scores["gen_rank_score"])
    energy_rank = float(scores["energy_rank_score"])
    proto_rank = float(scores["prototype_rank_score"])
    selected = str(_nested(cfg, "score_fusion.method", "prototype_rank")).lower()
    if selected in {"prototype_raw", "raw_prototype", "positive_boundary_raw"}:
        return float(row["prototype_score"]), float(scores["prototype_raw_threshold"]), scores
    return (
        float(_fuse_rank_scores(proser_rank, gen_rank, energy_rank, proto_rank, cfg, msp_rank=msp_rank)),
        float(scores[_component_threshold_key(cfg)]),
        scores,
    )


def calibrate_prototype_rank(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    cfg: Any,
    logger_: logging.Logger | None = None,
) -> tuple[PrototypeBank, pd.DataFrame, dict[str, Any]]:
    """Fit known-only prototypes and empirical-rank calibration.

    The input pool must contain known classes only.  A deterministic stratified
    70/30 (configurable) split separates prototype fitting from threshold/rank
    calibration.  The returned DataFrame carries rank calibrators in ``attrs``
    for immediate evaluation and can also be serialized for provenance.
    """
    log = logger_ or logger
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    unknown_label_id = int(_nested(cfg, "unknown_label_id", UNKNOWN_LABEL_ID))
    if np.any(labels_np == unknown_label_id):
        raise ValueError("Prototype-Rank calibration data must contain known classes only; found unknown labels.")

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

    proto_fit_fraction = float(_nested(cfg, "calibration.prototype_fit_fraction", 0.70))
    threshold_calib_fraction = float(_nested(cfg, "calibration.threshold_calibration_fraction", 0.30))
    split_seed = int(_nested(cfg, "prototype.seed", 42))

    if not np.isclose(proto_fit_fraction + threshold_calib_fraction, 1.0, atol=1e-6):
        raise ValueError(
            "Prototype-Rank prototype_fit_fraction + threshold_calibration_fraction must equal 1.0."
        )

    if 0.0 < proto_fit_fraction < 1.0 and len(df) >= max(2 * num_classes, 20):
        import hashlib
        from sklearn.model_selection import train_test_split

        try:
            proto_indices, calib_indices = train_test_split(
                np.arange(len(df)),
                test_size=threshold_calib_fraction,
                stratify=df["y_raw"].to_numpy(),
                random_state=split_seed,
            )
        except Exception as exc:
            log.warning("Stratified calibration split failed (%s); falling back to seeded random split.", exc)
            proto_indices, calib_indices = train_test_split(
                np.arange(len(df)),
                test_size=threshold_calib_fraction,
                random_state=split_seed,
            )
        if set(proto_indices.tolist()).intersection(calib_indices.tolist()):
            raise AssertionError("Prototype-fit and threshold-calibration indices overlap.")
        df_proto = df.iloc[proto_indices].copy()
        df_calib = df.iloc[calib_indices].copy()
        split_provenance = {
            "disjoint_split": True,
            "prototype_fit_samples": int(len(df_proto)),
            "threshold_calibration_samples": int(len(df_calib)),
            "prototype_fit_fraction": float(proto_fit_fraction),
            "threshold_calibration_fraction": float(threshold_calib_fraction),
            "split_seed": split_seed,
            "proto_indices_hash": hashlib.sha256(str(sorted(proto_indices.tolist())).encode()).hexdigest(),
            "calib_indices_hash": hashlib.sha256(str(sorted(calib_indices.tolist())).encode()).hexdigest(),
        }
    else:
        if bool(_nested(cfg, "calibration.strict_disjoint", True)):
            raise ValueError(
                "Known-only calibration pool is too small for the required disjoint prototype-fit / "
                "threshold-calibration split. Increase the reference pool or set "
                "calibration.strict_disjoint=false only for an explicit smoke/development run."
            )
        df_proto = df.copy()
        df_calib = df.copy()
        split_provenance = {
            "disjoint_split": False,
            "total_samples": int(len(df)),
            "note": "Smoke/development fallback only: prototype fitting and calibration reuse the same pool.",
        }

    prototype_bank = _fit_prototypes(df_proto, num_classes, cfg, log)
    df["prototype_score"] = [
        float(prototype_bank.score(np.asarray(row["feature"]).reshape(1, -1), int(row["y_raw"]))[0])
        for _, row in df.iterrows()
    ]
    df_calib["prototype_score"] = [float(df.loc[idx, "prototype_score"]) for idx in df_calib.index]

    rank_calibrators = _build_rank_calibrators(df_calib, num_classes=num_classes, cfg=cfg, log=log)
    calibration_scope = str(_nested(cfg, "score_fusion.calibration_scope", "global")).lower()
    selected_scores: list[float] = []
    selected_thresholds: list[float] = []
    selected_rejects: list[int] = []
    for idx, row in df_calib.iterrows():
        c = int(row["y_raw"])
        cal = rank_calibrators.get(-1) if calibration_scope == "global" else rank_calibrators.get(c)
        if cal is None:
            cal = rank_calibrators.get(c)
        if cal is None:
            raise ValueError(
                f"No empirical-rank calibrator available for class={c}. "
                "Increase known calibration support or lower calibration.min_samples_per_class for a development run."
            )
        score, threshold, component_scores = _select_rank_score_and_threshold(row, cal, cfg)
        selected_scores.append(score)
        selected_thresholds.append(threshold)
        selected_rejects.append(int(score > threshold))
        for key, value in component_scores.items():
            df_calib.loc[idx, key] = value

    df_calib["selected_score"] = selected_scores
    df_calib["selected_threshold"] = selected_thresholds
    df_calib["selected_reject"] = selected_rejects
    calibration_known_fpr = float(np.mean(df_calib["selected_reject"].to_numpy(dtype=float))) if len(df_calib) else 0.0

    meta = {
        "backend": "prototype_rank",
        "method": "FedTROS-PR",
        "decision_rule": str(_nested(cfg, "score_fusion.method", "prototype_rank")),
        "calibration_type": "known_only_empirical_rank",
        "num_classes": num_classes,
        "unknown_label_id": int(unknown_label_id),
        "open_set_label_id": int(_nested(cfg, "open_set_label_id", OPEN_SET_LABEL_ID)),
        "target_known_fpr": float(_nested(cfg, "calibration.target_known_fpr", 0.05)),
        "realized_calibration_known_fpr": calibration_known_fpr,
        "split_provenance": split_provenance,
        "rank_calibration": {
            str(k): {
                "n": int(v.get("n", 0)),
                "energy_median": float(v.get("energy_median", np.nan)),
                "energy_iqr": float(v.get("energy_iqr", np.nan)),
                "msp_threshold": float(v.get("msp_threshold", np.nan)),
                "proser_threshold": float(v.get("proser_threshold", np.nan)),
                "gen_threshold": float(v.get("gen_threshold", np.nan)),
                "energy_threshold": float(v.get("energy_threshold", np.nan)),
                "prototype_threshold": float(v.get("prototype_threshold", np.nan)),
                "prototype_raw_threshold": float(v.get("prototype_raw_threshold", np.nan)),
                "fusion_threshold": float(v.get("fusion_threshold", np.nan)),
            }
            for k, v in rank_calibrators.items()
        },
        "prototypes": prototype_bank.to_payload(),
    }
    df_calib.attrs["rank_calibrators"] = rank_calibrators
    df_calib.attrs["split_provenance"] = split_provenance
    df_calib.attrs["calibration_known_fpr"] = calibration_known_fpr
    return prototype_bank, df_calib, meta


def evaluate_prototype_rank(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    cfg: Any,
    class_names: dict[int, str],
    output_dir: Path,
    prototype_bank: PrototypeBank,
    calibration_df: pd.DataFrame,
    logger_: logging.Logger | None = None,
    report_to_stdout: bool = False,
) -> dict[str, float]:
    """Evaluate the configured known-only Prototype-Rank detector.

    The function requires a calibration DataFrame.  There is deliberately no EVT,
    Weibull, or uncalibrated fallback: a publication run must fail loudly if the
    known-only calibration contract is unavailable.
    """
    log = logger_ or logger
    output_dir = _ensure_dir(output_dir)
    metrics_dir = _ensure_dir(output_dir / "metrics")
    predictions_dir = _ensure_dir(output_dir / "predictions")
    artifacts_dir = _ensure_dir(output_dir / "artifacts")
    open_set_label_id = int(_nested(cfg, "open_set_label_id", OPEN_SET_LABEL_ID))
    unknown_label_id = int(_nested(cfg, "unknown_label_id", UNKNOWN_LABEL_ID))

    if calibration_df is None:
        raise ValueError("Prototype-Rank evaluation requires the disjoint known-only calibration DataFrame.")
    rank_calibrators = calibration_df.attrs.get("rank_calibrators")
    if not rank_calibrators:
        rank_calibrators = _build_rank_calibrators(
            calibration_df,
            num_classes=int(student_model.num_classes),
            cfg=cfg,
            log=log,
        )
    if not rank_calibrators:
        raise ValueError("No empirical-rank calibrators were fitted; refusing uncalibrated open-set evaluation.")

    df = _collect_student_scores(
        features,
        labels,
        student_model=student_model,
        batch_size=batch_size,
        device=device,
        cfg=cfg,
        class_condition="pred",
    )
    df["prototype_score"] = [
        float(prototype_bank.score(np.asarray(row["feature"]).reshape(1, -1), int(row["pred_before_osr"]))[0])
        for _, row in df.iterrows()
    ]

    y_true = np.asarray(
        [open_set_label_id if int(v) == unknown_label_id else int(v) for v in df["y_raw"]],
        dtype=int,
    )
    y_before = df["pred_before_osr"].to_numpy(dtype=int)
    y_binary = (y_true == open_set_label_id).astype(int)
    calibration_scope = str(_nested(cfg, "score_fusion.calibration_scope", "global")).lower()
    selected_score_name = str(_nested(cfg, "score_fusion.method", "prototype_rank"))

    final_preds: list[int] = []
    unknown_scores: list[float] = []
    selected_thresholds: list[float] = []
    component_rows: list[dict[str, float]] = []

    for _, row in df.iterrows():
        c = int(row["pred_before_osr"])
        cal = rank_calibrators.get(-1) if calibration_scope == "global" else rank_calibrators.get(c)
        if cal is None:
            cal = rank_calibrators.get(c)
        if cal is None:
            raise ValueError(f"Missing empirical-rank calibrator for predicted class={c}.")

        unknown_score, selected_threshold, scores = _select_rank_score_and_threshold(row, cal, cfg)
        msp_rank = float(scores["msp_rank_score"])
        proser_rank = float(scores["proser_rank_score"])
        gen_rank = float(scores["gen_rank_score"])
        energy_rank = float(scores["energy_rank_score"])
        proto_rank = float(scores["prototype_rank_score"])
        mean_rank = _mean_finite([proser_rank, gen_rank, energy_rank, proto_rank])
        weighted_rank = _mean_finite(
            [proser_rank, gen_rank, energy_rank, proto_rank], weights=_score_weights(cfg)
        )
        max_rank = float(np.nanmax([proser_rank, gen_rank, energy_rank, proto_rank]))
        reject = bool(unknown_score > selected_threshold)
        final_preds.append(open_set_label_id if reject else c)
        unknown_scores.append(float(unknown_score))
        selected_thresholds.append(float(selected_threshold))
        component_rows.append(
            {
                **scores,
                "mean_rank_score": float(mean_rank),
                "weighted_rank_score": float(weighted_rank),
                "max_rank_score": float(max_rank),
                "msp_rank_reject": int(msp_rank > float(scores["msp_threshold"])),
                "proser_rank_reject": int(proser_rank > float(scores["proser_threshold"])),
                "gen_rank_reject": int(gen_rank > float(scores["gen_threshold"])),
                "energy_rank_reject": int(energy_rank > float(scores["energy_threshold"])),
                "prototype_rank_reject": int(proto_rank > float(scores["prototype_threshold"])),
                "prototype_raw_reject": int(float(row["prototype_score"]) > float(scores["prototype_raw_threshold"])),
                "mean_rank_reject": int(mean_rank > float(scores["mean_threshold"])),
                "weighted_rank_reject": int(weighted_rank > float(scores["weighted_threshold"])),
                "max_rank_reject": int(max_rank > float(scores["max_threshold"])),
            }
        )

    component_df = pd.DataFrame(component_rows, index=df.index)
    for column in component_df.columns:
        df[column] = component_df[column]

    y_pred = np.asarray(final_preds, dtype=int)
    score_arr = np.asarray(unknown_scores, dtype=float)
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
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
            artifacts_dir / "open_set_roc_curve.csv", index=False
        )
        np.savez_compressed(artifacts_dir / "roc_data.npz", fpr=fpr, tpr=tpr, thresholds=roc_thresholds)
        precision, recall, pr_thresholds = precision_recall_curve(y_binary, score_arr)
        pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": np.concatenate([pr_thresholds, [np.nan]]),
            }
        ).to_csv(artifacts_dir / "open_set_pr_curve.csv", index=False)
        np.savez_compressed(
            artifacts_dir / "pr_data.npz", precision=precision, recall=recall, thresholds=pr_thresholds
        )

    known_mask = y_true != open_set_label_id
    unknown_mask = ~known_mask
    known_acc_before = float(accuracy_score(y_true[known_mask], y_before[known_mask])) if known_mask.any() else 0.0
    known_acc_after = float(accuracy_score(y_true[known_mask], y_pred[known_mask])) if known_mask.any() else 0.0
    unknown_recall = float(np.mean(y_pred[unknown_mask] == open_set_label_id)) if unknown_mask.any() else 0.0
    known_false_unknown_rate = float(np.mean(y_pred[known_mask] == open_set_label_id)) if known_mask.any() else 0.0
    known_acceptance_rate = 1.0 - known_false_unknown_rate
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    pred_unknown_binary = (y_pred == open_set_label_id).astype(int)
    unknown_f1 = float(f1_score(y_binary, pred_unknown_binary, zero_division=0))
    tp_unknown = float(np.sum((y_binary == 1) & (pred_unknown_binary == 1)))
    pred_unknown_total = float(np.sum(pred_unknown_binary == 1))
    unknown_precision = tp_unknown / pred_unknown_total if pred_unknown_total > 0 else 0.0
    overall_acc = float(accuracy_score(y_true, y_pred))

    report_labels, report_names = _class_labels(class_names, open_set_label_id)
    before_cm = confusion_matrix(y_true, y_before, labels=report_labels)
    after_cm = confusion_matrix(y_true, y_pred, labels=report_labels)
    pd.DataFrame(before_cm, index=report_names, columns=report_names).to_csv(
        artifacts_dir / "before_osr_confusion_matrix.csv"
    )
    pd.DataFrame(after_cm, index=report_names, columns=report_names).to_csv(
        artifacts_dir / "after_osr_confusion_matrix.csv"
    )
    np.save(artifacts_dir / "confusion_closed.npy", before_cm)
    np.save(artifacts_dir / "confusion_open.npy", after_cm)
    report = classification_report(
        y_true, y_pred, labels=report_labels, target_names=report_names, digits=4, zero_division=0
    )
    (artifacts_dir / "openset_report.txt").write_text(report, encoding="utf-8")
    if report_to_stdout:
        print(report)

    df["y_true"] = y_true
    df["pred_after_osr"] = y_pred
    df["unknown_score"] = score_arr
    df["selected_score_name"] = selected_score_name
    df["selected_threshold_used"] = selected_thresholds
    df["final_reject"] = pred_unknown_binary
    df["known_or_unknown"] = np.where(y_binary == 1, "unknown", "known")
    export_df = df.drop(columns=["feature"])
    export_df.to_csv(predictions_dir / "prototype_rank_scores.csv", index=False)
    export_df.to_csv(predictions_dir / "open_set_scores.csv", index=False)
    calibration_df.drop(columns=["feature"], errors="ignore").to_csv(
        artifacts_dir / "prototype_rank_calibration_scores.csv", index=False
    )

    rank_thresholds_payload = {
        str(k): {
            "msp_threshold": float(v.get("msp_threshold", np.nan)),
            "proser_threshold": float(v.get("proser_threshold", np.nan)),
            "gen_threshold": float(v.get("gen_threshold", np.nan)),
            "energy_threshold": float(v.get("energy_threshold", np.nan)),
            "prototype_threshold": float(v.get("prototype_threshold", np.nan)),
            "prototype_raw_threshold": float(v.get("prototype_raw_threshold", np.nan)),
            "fusion_threshold": float(v.get("fusion_threshold", np.nan)),
            "mean_threshold": float(v.get("mean_threshold", np.nan)),
            "weighted_threshold": float(v.get("weighted_threshold", np.nan)),
            "max_threshold": float(v.get("max_threshold", np.nan)),
            "energy_median": float(v.get("energy_median", np.nan)),
            "energy_iqr": float(v.get("energy_iqr", np.nan)),
            "n": int(v.get("n", 0)),
        }
        for k, v in rank_calibrators.items()
    }
    (artifacts_dir / "prototype_rank_calibration.json").write_text(
        json.dumps(rank_thresholds_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifacts_dir / "fedtros_pr_rank_calibration.json").write_text(
        json.dumps(rank_thresholds_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifacts_dir / "fedtros_pr_prototype_bank.json").write_text(
        json.dumps(prototype_bank.to_payload(), indent=2, sort_keys=True), encoding="utf-8"
    )

    quantiles: dict[str, dict[str, float]] = {}
    overlap: dict[str, Any] = {}
    quantile_cols = [
        "msp_score",
        "msp_rank_score",
        "energy_score",
        "energy_rank_score",
        "prototype_score",
        "prototype_rank_score",
        "unknown_score",
    ]
    for col in quantile_cols:
        if col not in export_df.columns:
            continue
        known_vals = export_df.loc[export_df["known_or_unknown"] == "known", col].to_numpy(dtype=float)
        unknown_vals = export_df.loc[export_df["known_or_unknown"] == "unknown", col].to_numpy(dtype=float)
        quantiles[col] = {}
        for prefix, vals in [("known", known_vals), ("unknown", unknown_vals)]:
            vals = vals[np.isfinite(vals)]
            if vals.size:
                for q in [0.50, 0.90, 0.95, 0.99]:
                    quantiles[col][f"{prefix}_q{int(q * 100)}"] = float(np.quantile(vals, q))
        known_finite = known_vals[np.isfinite(known_vals)]
        unknown_finite = unknown_vals[np.isfinite(unknown_vals)]
        if known_finite.size and unknown_finite.size:
            known95 = float(np.quantile(known_finite, 0.95))
            overlap[col] = {
                "known_q95": known95,
                "unknown_le_known_q95_rate": float(np.mean(unknown_finite <= known95)),
                "unknown_gt_known_q95_rate": float(np.mean(unknown_finite > known95)),
            }
    (artifacts_dir / "known_unknown_score_quantiles.json").write_text(
        json.dumps(quantiles, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifacts_dir / "score_overlap_report.json").write_text(
        json.dumps(overlap, indent=2, sort_keys=True), encoding="utf-8"
    )

    raw_aurocs = {
        "auroc_selected_unknown_score": auroc,
        "auroc_msp_score": _safe_auc(y_binary, export_df["msp_score"].to_numpy(dtype=float)),
        "auroc_msp_rank": _safe_auc(y_binary, export_df["msp_rank_score"].to_numpy(dtype=float)),
        "auroc_energy_score_high": _safe_auc(y_binary, export_df["energy_score"].to_numpy(dtype=float)),
        "auroc_energy_score_reversed": _safe_auc(y_binary, -export_df["energy_score"].to_numpy(dtype=float)),
        "auroc_energy_rank": _safe_auc(y_binary, export_df["energy_rank_score"].to_numpy(dtype=float)),
        "auroc_prototype_score": _safe_auc(y_binary, export_df["prototype_score"].to_numpy(dtype=float)),
        "auroc_prototype_rank": _safe_auc(y_binary, export_df["prototype_rank_score"].to_numpy(dtype=float)),
    }

    def _component_stats(name: str, reject_col: str) -> dict[str, float]:
        reject = export_df[reject_col].to_numpy(dtype=int) == 1
        pred_component = np.where(reject, open_set_label_id, y_before)
        return {
            f"{name}_unknown_recall": float(np.mean(reject[unknown_mask])) if unknown_mask.any() else 0.0,
            f"{name}_known_false_unknown_rate": float(np.mean(reject[known_mask])) if known_mask.any() else 0.0,
            f"{name}_overall_acc": float(accuracy_score(y_true, pred_component)),
            f"{name}_rejected_total": float(np.sum(reject)),
        }

    component_decisions: dict[str, float] = {}
    for name, column in [
        ("msp_rank", "msp_rank_reject"),
        ("energy_rank", "energy_rank_reject"),
        ("prototype_raw", "prototype_raw_reject"),
        ("prototype_rank", "prototype_rank_reject"),
        ("mean_rank_fusion", "mean_rank_reject"),
        ("weighted_rank_fusion", "weighted_rank_reject"),
        ("max_rank_fusion", "max_rank_reject"),
    ]:
        component_decisions.update(_component_stats(name, column))

    unknown_as_normal_before = 0.0
    if unknown_mask.any() and 0 in class_names:
        unknown_as_normal_before = float(np.mean(y_before[unknown_mask] == 0))
    calibration_known_fpr = float(calibration_df.attrs.get("calibration_known_fpr", np.nan))
    global_cal = rank_calibrators.get(-1)
    reported_threshold = float("nan")
    if global_cal is not None:
        dummy = calibration_df.iloc[0] if len(calibration_df) else None
        if dummy is not None:
            _, reported_threshold, _ = _select_rank_score_and_threshold(dummy, global_cal, cfg)

    metrics = {
        "openset_backend_prototype_rank": 1.0,
        "openset_auroc": auroc,
        "openset_auprc": auprc,
        "openset_fpr95": fpr95,
        "openset_f1_macro": f1_macro,
        "openset_unknown_f1": unknown_f1,
        "openset_unknown_precision": unknown_precision,
        "openset_known_acc_before": known_acc_before,
        "openset_known_acc": known_acc_after,
        "openset_unknown_recall": unknown_recall,
        "openset_known_false_unknown_rate": known_false_unknown_rate,
        "openset_overall_acc": overall_acc,
        "openset_unknown_as_normal_before_rate": unknown_as_normal_before,
        "openset_rejected_by_proser": float(np.sum(export_df.get("proser_rank_reject", 0.0))),
        "openset_rejected_by_gen": float(np.sum(export_df.get("gen_rank_reject", 0.0))),
        "openset_rejected_by_energy": float(np.sum(export_df.get("energy_rank_reject", 0.0))),
        "openset_rejected_by_prototype": float(np.sum(export_df.get("prototype_rank_reject", 0.0))),
        "open_set/auroc": auroc,
        "open_set/auprc": auprc,
        "open_set/fpr95": fpr95,
        "open_set/fpr_at_95_tpr": fpr95,
        "open_set/unknown_precision": unknown_precision,
        "open_set/unknown_recall": unknown_recall,
        "open_set/unknown_f1": unknown_f1,
        "open_set/known_accuracy_before": known_acc_before,
        "open_set/known_accuracy_after": known_acc_after,
        "open_set/known_acceptance_rate": known_acceptance_rate,
        "open_set/known_false_unknown_rate": known_false_unknown_rate,
        "open_set/macro_f1": f1_macro,
        "open_set/overall_accuracy": overall_acc,
        "prototype_rank/threshold": reported_threshold,
        "prototype_rank/calibration_known_fpr": calibration_known_fpr,
        "prototype_rank/test_known_fur": known_false_unknown_rate,
        "prototype_rank/num_positive_prototypes": float(sum(len(v) for v in prototype_bank.prototypes.values())),
        "prototype_rank/num_boundary_prototypes": float(
            len(prototype_bank.negative_prototypes) if prototype_bank.negative_prototypes is not None else 0
        ),
        **raw_aurocs,
        **component_decisions,
    }
    (metrics_dir / "open_set_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info(
        "FedTROS-PR open-set | selected=%s AUROC=%.4f AUPRC=%.4f FPR95=%.4f KnownAcc %.4f->%.4f "
        "UnknownRecall=%.4f KnownFU=%.4f calibration_KFU=%.4f",
        selected_score_name,
        auroc,
        auprc,
        fpr95,
        known_acc_before,
        known_acc_after,
        unknown_recall,
        known_false_unknown_rate,
        calibration_known_fpr,
    )
    return metrics


# Backward-compatibility aliases
