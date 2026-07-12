"""Robust post-federation Positive-Negative Prototypes Fusion Framework.

The positive/negative prototype objectives implement Zhong and Cui (2025),
Equations 1-12 and 27-28.  The detector adapts a private copy of the aggregated
student backbone and uses known-derived manifold mixup only for open-space
regularisation and score calibration.  Real unknown samples are never accepted
by the fitting API.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


EPS = 1.0e-8


@dataclass
class PNPFFConfig:
    feature_dim: int = 128
    num_positive_prototypes: int = 7
    gamma1: float = 1.0
    gamma2: float = 1.0
    lambda0: float = 0.1
    lambda1: float = 0.1
    lambda2: float = 0.01
    lambda_unknown: float = 0.1
    diversity_temperature: float = 10.0
    eta: float = 0.5
    omega: float = 0.5
    fit_fraction: float = 0.70
    seed: int = 42
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.01
    paper_learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_milestones: tuple[int, ...] = (30, 60, 90)
    lr_gamma: float = 0.1
    grad_clip_norm: float = 5.0
    adapt_backbone: bool = True
    normalize_embedding: bool = True
    balanced_batches: bool = True
    known_accuracy_tolerance: float = 0.02
    threshold_mode: str = "pseudo_unknown_constrained"
    tau: float = 0.5
    target_known_fpr: float = 0.05
    health_min_auroc: float = 0.55
    health_fpr_tolerance: float = 0.01
    health_action: str = "error"
    fit_each_round: bool = False
    pseudo_unknown_enabled: bool = True
    pseudo_unknown_ratio: float = 0.5
    pseudo_unknown_mixup_alpha: float = 0.4
    pseudo_unknown_mask_probability: float = 0.15
    pseudo_unknown_noise_std: float = 0.05

    @classmethod
    def from_cfg(cls, cfg: Any, *, feature_dim: int, batch_size: int) -> "PNPFFConfig":
        p = getattr(cfg, "pnpff", None)

        def value(name: str, default: Any) -> Any:
            return getattr(p, name, default) if p is not None else default

        pseudo = getattr(p, "pseudo_unknown", None) if p is not None else None

        def pseudo_value(name: str, default: Any) -> Any:
            return getattr(pseudo, name, default) if pseudo is not None else default

        return cls(
            feature_dim=int(value("feature_dim", feature_dim)),
            num_positive_prototypes=int(value("num_positive_prototypes", 7)),
            gamma1=float(value("gamma1", 1.0)),
            gamma2=float(value("gamma2", 1.0)),
            lambda0=float(value("lambda0", 0.1)),
            lambda1=float(value("lambda1", 0.1)),
            lambda2=float(value("lambda2", 0.01)),
            lambda_unknown=float(value("lambda_unknown", 0.1)),
            diversity_temperature=float(value("diversity_temperature", 10.0)),
            eta=float(value("eta", 0.5)),
            omega=float(value("omega", 0.5)),
            fit_fraction=float(value("fit_fraction", 0.70)),
            seed=int(value("seed", 42)),
            epochs=int(value("epochs", 100)),
            batch_size=int(value("batch_size", batch_size)),
            learning_rate=float(value("learning_rate", 0.01)),
            paper_learning_rate=float(value("paper_learning_rate", 0.1)),
            momentum=float(value("momentum", 0.9)),
            weight_decay=float(value("weight_decay", 0.0)),
            lr_milestones=tuple(int(v) for v in value("lr_milestones", [30, 60, 90])),
            lr_gamma=float(value("lr_gamma", 0.1)),
            grad_clip_norm=float(value("grad_clip_norm", 5.0)),
            adapt_backbone=bool(value("adapt_backbone", True)),
            normalize_embedding=bool(value("normalize_embedding", True)),
            balanced_batches=bool(value("balanced_batches", True)),
            known_accuracy_tolerance=float(value("known_accuracy_tolerance", 0.02)),
            threshold_mode=str(value("threshold_mode", "pseudo_unknown_constrained")),
            tau=float(value("tau", 0.5)),
            target_known_fpr=float(value("target_known_fpr", 0.05)),
            health_min_auroc=float(value("health_min_auroc", 0.55)),
            health_fpr_tolerance=float(value("health_fpr_tolerance", 0.01)),
            health_action=str(value("health_action", "error")),
            fit_each_round=bool(value("fit_each_round", False)),
            pseudo_unknown_enabled=bool(pseudo_value("enabled", True)),
            pseudo_unknown_ratio=float(pseudo_value("ratio", 0.5)),
            pseudo_unknown_mixup_alpha=float(pseudo_value("mixup_alpha", 0.4)),
            pseudo_unknown_mask_probability=float(pseudo_value("mask_probability", 0.15)),
            pseudo_unknown_noise_std=float(pseudo_value("noise_std", 0.05)),
        )


def stratified_fit_calibration_split(
    labels: torch.Tensor,
    *,
    fit_fraction: float = 0.70,
    seed: int = 42,
    unknown_label_id: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic known-only fit/calibration indices."""
    y = labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
    if np.any(y == int(unknown_label_id)) or np.any(y < 0):
        raise ValueError("PNPFF fitting/calibration data must contain known classes only.")
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("PNPFF fit_fraction must be strictly between 0 and 1.")
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2 or np.any(counts < 2):
        raise ValueError(
            "PNPFF stratified split requires at least two classes and two samples per class."
        )
    indices = np.arange(y.size)
    fit_idx, calibration_idx = train_test_split(
        indices,
        train_size=float(fit_fraction),
        random_state=int(seed),
        shuffle=True,
        stratify=y,
    )
    return np.sort(fit_idx), np.sort(calibration_idx)


def make_pseudo_unknowns(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    ratio: float,
    mixup_alpha: float,
    mask_probability: float,
    noise_std: float,
    seed: int,
) -> torch.Tensor:
    """Create deterministic cross-class manifold-mixup samples from known data."""
    x = inputs.detach().cpu().float()
    y = labels.detach().cpu().long().reshape(-1)
    if x.shape[0] < 2 or float(ratio) <= 0.0:
        return x.new_zeros((0, *x.shape[1:]))
    classes = torch.unique(y)
    if classes.numel() < 2:
        raise ValueError("PNPFF pseudo-unknown generation requires at least two known classes.")
    rng = np.random.default_rng(int(seed))
    n = max(1, int(round(float(ratio) * x.shape[0])))
    first = rng.integers(0, x.shape[0], size=n)
    second = np.empty(n, dtype=np.int64)
    y_np = y.numpy()
    for i, idx in enumerate(first):
        candidates = np.flatnonzero(y_np != y_np[idx])
        second[i] = int(candidates[rng.integers(0, candidates.size)])
    alpha = max(float(mixup_alpha), 1.0e-3)
    lam_shape = (n,) + ((1,) * (x.ndim - 1))
    lam = torch.from_numpy(rng.beta(alpha, alpha, size=n).astype(np.float32)).reshape(lam_shape)
    pseudo = lam * x[torch.from_numpy(first)] + (1.0 - lam) * x[torch.from_numpy(second)]
    if float(mask_probability) > 0.0:
        mask = torch.from_numpy(rng.random(pseudo.shape) < float(mask_probability))
        pseudo = pseudo.masked_fill(mask, 0.0)
    if float(noise_std) > 0.0:
        noise = torch.from_numpy(
            rng.normal(0.0, float(noise_std), size=pseudo.shape).astype(np.float32)
        )
        pseudo = pseudo + noise
    return pseudo.float()


class PNPFFModel(nn.Module):
    """Adapted embedding plus class-specific positive and negative prototypes."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cfg: PNPFFConfig,
        *,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.cfg = cfg
        self.backbone = copy.deepcopy(backbone) if backbone is not None else nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(bool(cfg.adapt_backbone))
        self.projection = nn.Linear(self.input_dim, int(cfg.feature_dim), bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            diagonal = min(self.input_dim, int(cfg.feature_dim))
            self.projection.weight[:diagonal, :diagonal] = torch.eye(diagonal)
        self.positive_prototypes = nn.Parameter(
            torch.empty(self.num_classes, int(cfg.num_positive_prototypes), int(cfg.feature_dim))
        )
        self.negative_prototypes = nn.Parameter(torch.empty(self.num_classes, int(cfg.feature_dim)))
        # softplus(-6.9) is close to zero but retains a useful gradient.
        self.raw_radius = nn.Parameter(torch.tensor(-6.9))
        self.register_buffer("feature_mean", torch.zeros(self.input_dim))
        self.register_buffer("feature_std", torch.ones(self.input_dim))
        nn.init.normal_(self.positive_prototypes, std=0.02)
        nn.init.normal_(self.negative_prototypes, std=0.02)
        self.retract_prototypes()

    @property
    def radius(self) -> torch.Tensor:
        return F.softplus(self.raw_radius)

    @staticmethod
    def distance(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Equation 1: squared Euclidean / m minus dot product."""
        m = max(int(features.shape[-1]), 1)
        return (features - prototypes).pow(2).sum(dim=-1) / float(m) - (features * prototypes).sum(
            dim=-1
        )

    def set_feature_normalization(self, inputs: torch.Tensor) -> None:
        self.eval()
        with torch.no_grad():
            features = self.backbone(inputs)
            self.feature_mean.copy_(features.mean(dim=0))
            self.feature_std.copy_(features.std(dim=0, unbiased=False).clamp_min(1.0e-4))

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        features = (features - self.feature_mean) / self.feature_std
        z = self.projection(features)
        return F.normalize(z, dim=-1, eps=EPS) if self.cfg.normalize_embedding else z

    def normalized_prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.normalize_embedding:
            return self.positive_prototypes, self.negative_prototypes
        return (
            F.normalize(self.positive_prototypes, dim=-1, eps=EPS),
            F.normalize(self.negative_prototypes, dim=-1, eps=EPS),
        )

    @torch.no_grad()
    def retract_prototypes(self) -> None:
        if self.cfg.normalize_embedding:
            self.positive_prototypes.copy_(F.normalize(self.positive_prototypes, dim=-1, eps=EPS))
            self.negative_prototypes.copy_(F.normalize(self.negative_prototypes, dim=-1, eps=EPS))

    def distances(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.embed(inputs)
        positive, negative = self.normalized_prototypes()
        pos_all = self.distance(z[:, None, None, :], positive[None, :, :, :])
        pos = pos_all.min(dim=-1).values  # Equation 2.
        neg = self.distance(z[:, None, :], negative[None, :, :])
        return z, pos, neg

    def probabilities(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        z, pos_dist, neg_dist = self.distances(inputs)
        positive = F.softmax(-float(self.cfg.gamma1) * pos_dist, dim=1)  # Equation 3.
        negative = F.softmax(float(self.cfg.gamma2) * neg_dist, dim=1)  # Equation 10.
        weight_sum = max(float(self.cfg.eta) + float(self.cfg.omega), EPS)
        fused = ((float(self.cfg.eta) * positive) + (float(self.cfg.omega) * negative)) / weight_sum
        known_confidence, predicted_class = fused.max(dim=1)
        raw_unknown_score = (1.0 - known_confidence).clamp(0.0, 1.0)
        return {
            "features": z,
            "embedding_norm": z.norm(dim=1),
            "positive_distances": pos_dist,
            "negative_distances": neg_dist,
            "positive_probabilities": positive,
            "negative_probabilities": negative,
            "fused_scores": fused,
            "known_confidence": known_confidence,
            "raw_unknown_score": raw_unknown_score,
            "unknown_score": raw_unknown_score,
            "predicted_class": predicted_class,
        }

    def positive_loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        z, pos_dist, _ = self.distances(inputs)
        classification = F.cross_entropy(-float(self.cfg.gamma1) * pos_dist, labels)
        positive, _ = self.normalized_prototypes()
        class_prototypes = positive[labels]
        class_average = class_prototypes.mean(dim=1)
        moving = F.relu(
            (z - class_average).pow(2).sum(dim=1) / float(self.cfg.feature_dim) - self.radius
        ).mean()
        pairwise = self.distance(positive.unsqueeze(2), positive.unsqueeze(1))
        mask = ~torch.eye(positive.shape[1], dtype=torch.bool, device=positive.device).unsqueeze(0)
        local_diversity = (
            -pairwise.masked_select(mask).mean() if positive.shape[1] > 1 else positive.sum() * 0.0
        )
        assignments = self.assignment_probabilities_from_embeddings(z, class_prototypes)
        assignment_by_class = []
        uniform = torch.full((positive.shape[1],), 1.0 / float(positive.shape[1]), device=z.device)
        for class_id in torch.unique(labels):
            mean_assignment = assignments[labels == class_id].mean(dim=0).clamp_min(EPS)
            assignment_by_class.append(
                (mean_assignment * (mean_assignment.log() - uniform.log())).sum()
            )
        assignment_diversity = torch.stack(assignment_by_class).mean()
        diversity = local_diversity + assignment_diversity
        total = (
            classification + float(self.cfg.lambda1) * moving + float(self.cfg.lambda2) * diversity
        )
        return {
            "total": total,
            "classification": classification,
            "moving": moving,
            "diversity": diversity,
        }

    def assignment_probabilities_from_embeddings(
        self,
        embeddings: torch.Tensor,
        class_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        assigned_dist = self.distance(embeddings.unsqueeze(1), class_prototypes)
        # Eq. 8 prints +T*d, but Eq. 1 defines lower d as more similar and the
        # accompanying prose says assignment must favour the most similar slot.
        return F.softmax(-float(self.cfg.diversity_temperature) * assigned_dist, dim=1)

    def negative_loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, _, neg_dist = self.distances(inputs)
        return float(self.cfg.lambda0) * F.cross_entropy(float(self.cfg.gamma2) * neg_dist, labels)

    def open_space_loss(self, pseudo_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.probabilities(pseudo_inputs)
        log_uniform = -float(np.log(max(self.num_classes, 1)))
        positive = probs["positive_probabilities"].clamp_min(EPS)
        negative = probs["negative_probabilities"].clamp_min(EPS)
        positive_kl = (positive * (positive.log() - log_uniform)).sum(dim=1).mean()
        negative_kl = (negative * (negative.log() - log_uniform)).sum(dim=1).mean()
        return positive_kl, negative_kl


@dataclass
class MonotonicCalibrator:
    x_thresholds: list[float] = field(default_factory=lambda: [0.0, 1.0])
    y_thresholds: list[float] = field(default_factory=lambda: [0.0, 1.0])

    @classmethod
    def fit(cls, known_scores: np.ndarray, pseudo_scores: np.ndarray) -> "MonotonicCalibrator":
        x = np.concatenate([known_scores, pseudo_scores]).astype(np.float64)
        y = np.concatenate([np.zeros(known_scores.size), np.ones(pseudo_scores.size)])
        if x.size == 0 or not np.isfinite(x).all():
            raise ValueError("PNPFF calibration scores must be finite and non-empty.")
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(x, y)
        return cls(
            iso.X_thresholds_.astype(float).tolist(), iso.y_thresholds_.astype(float).tolist()
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(scores, dtype=np.float64),
            np.asarray(self.x_thresholds, dtype=np.float64),
            np.asarray(self.y_thresholds, dtype=np.float64),
        ).clip(0.0, 1.0)


@dataclass
class PNPFFDetector:
    model: PNPFFModel
    threshold: float
    config: PNPFFConfig
    history: list[dict[str, float]]
    fit_size: int
    calibration_size: int
    pseudo_fit_size: int = 0
    pseudo_calibration_size: int = 0
    calibrator: MonotonicCalibrator = field(default_factory=MonotonicCalibrator)
    health: dict[str, Any] = field(default_factory=dict)
    best_epoch: int = 0

    @property
    def is_healthy(self) -> bool:
        return bool(self.health.get("healthy", False))

    def predict_inputs(
        self, inputs: torch.Tensor, *, device: torch.device | str = "cpu"
    ) -> dict[str, np.ndarray]:
        target = torch.device(device)
        self.model.to(target).eval()
        with torch.no_grad():
            values = self.model.probabilities(inputs.to(target).float())
        result = {
            key: value.detach().cpu().numpy() for key, value in values.items() if key != "features"
        }
        result["unknown_score"] = self.calibrator.predict(result["raw_unknown_score"]).astype(
            np.float32
        )
        return result

    # Backward compatibility for callers/tests that used an identity backbone.
    def predict_features(
        self, features: torch.Tensor, *, device: torch.device | str = "cpu"
    ) -> dict[str, np.ndarray]:
        return self.predict_inputs(features, device=device)

    def to_payload(self) -> dict[str, Any]:
        positive, negative = self.model.normalized_prototypes()
        return {
            "method": "robust_pnpff_fed",
            "threshold": float(self.threshold),
            "config": asdict(self.config),
            "fit_size": int(self.fit_size),
            "calibration_size": int(self.calibration_size),
            "pseudo_fit_size": int(self.pseudo_fit_size),
            "pseudo_calibration_size": int(self.pseudo_calibration_size),
            "best_epoch": int(self.best_epoch),
            "history": self.history,
            "health": self.health,
            "calibrator": asdict(self.calibrator),
            "positive_prototypes": positive.detach().cpu().tolist(),
            "negative_prototypes": negative.detach().cpu().tolist(),
            "radius": float(self.model.radius.detach().cpu().item()),
            "feature_mean": self.model.feature_mean.detach().cpu().tolist(),
            "feature_std": self.model.feature_std.detach().cpu().tolist(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_dim": self.model.input_dim,
                "num_classes": self.model.num_classes,
                "threshold": float(self.threshold),
                "config": asdict(self.config),
                "history": self.history,
                "fit_size": int(self.fit_size),
                "calibration_size": int(self.calibration_size),
                "pseudo_fit_size": int(self.pseudo_fit_size),
                "pseudo_calibration_size": int(self.pseudo_calibration_size),
                "best_epoch": int(self.best_epoch),
                "health": self.health,
                "calibrator": asdict(self.calibrator),
            },
            path,
        )


def _initialise_prototypes(
    model: PNPFFModel, inputs: torch.Tensor, labels: torch.Tensor, seed: int
) -> None:
    generator = torch.Generator(device=inputs.device).manual_seed(int(seed))
    with torch.no_grad():
        projected = model.embed(inputs)
        global_mean = projected.mean(dim=0)
        for class_id in range(model.num_classes):
            class_features = projected[labels == class_id]
            if class_features.numel() == 0:
                raise ValueError(f"PNPFF fit split is missing class {class_id}.")
            order = torch.randperm(
                class_features.shape[0], generator=generator, device=inputs.device
            )
            selected = class_features[
                order[: min(order.numel(), model.cfg.num_positive_prototypes)]
            ]
            if selected.shape[0] < model.cfg.num_positive_prototypes:
                repeats = int(np.ceil(model.cfg.num_positive_prototypes / selected.shape[0]))
                selected = selected.repeat((repeats, 1))
            model.positive_prototypes[class_id].copy_(selected[: model.cfg.num_positive_prototypes])
            other = projected[labels != class_id]
            model.negative_prototypes[class_id].copy_(
                other.mean(dim=0) if other.numel() else global_mean
            )
        model.retract_prototypes()


def _balanced_loader(inputs: torch.Tensor, labels: torch.Tensor, cfg: PNPFFConfig) -> DataLoader:
    dataset = TensorDataset(inputs.detach().cpu(), labels.detach().cpu())
    generator = torch.Generator().manual_seed(int(cfg.seed))
    if not cfg.balanced_batches:
        return DataLoader(
            dataset, batch_size=max(1, cfg.batch_size), shuffle=True, generator=generator
        )
    counts = (
        torch.bincount(labels.detach().cpu(), minlength=int(labels.max().item()) + 1)
        .float()
        .clamp_min(1.0)
    )
    weights = 1.0 / counts[labels.detach().cpu()]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True, generator=generator
    )
    return DataLoader(dataset, batch_size=max(1, cfg.batch_size), sampler=sampler)


def _safe_auc(known: np.ndarray, pseudo: np.ndarray) -> float:
    y = np.concatenate([np.zeros(known.size), np.ones(pseudo.size)])
    score = np.concatenate([known, pseudo])
    return float(roc_auc_score(y, score)) if known.size and pseudo.size else 0.0


def _choose_threshold(
    known_scores: np.ndarray,
    pseudo_scores: np.ndarray,
    *,
    target_known_fpr: float,
) -> tuple[float, float, float]:
    candidates = np.unique(np.concatenate([known_scores, pseudo_scores]))
    best: tuple[float, float, float] | None = None
    y = np.concatenate(
        [np.zeros(known_scores.size, dtype=int), np.ones(pseudo_scores.size, dtype=int)]
    )
    all_scores = np.concatenate([known_scores, pseudo_scores])
    for threshold in candidates:
        known_fpr = float(np.mean(known_scores > threshold))
        if known_fpr > float(target_known_fpr) + 1.0e-12:
            continue
        f1 = float(f1_score(y, (all_scores > threshold).astype(int), zero_division=0))
        candidate = (f1, -known_fpr, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None or best[0] <= 0.0:
        threshold = float(np.quantile(known_scores, 1.0 - float(target_known_fpr)))
        return threshold, float(np.mean(known_scores > threshold)), 0.0
    return best[2], -best[1], best[0]


def fit_pnpff_detector(
    fit_inputs: torch.Tensor,
    fit_labels: torch.Tensor,
    calibration_inputs: torch.Tensor,
    calibration_labels: torch.Tensor,
    *,
    num_classes: int,
    cfg: PNPFFConfig,
    device: torch.device,
    backbone: nn.Module | None = None,
    backbone_feature_dim: int | None = None,
) -> PNPFFDetector:
    """Fit robust PNPFF using only known inputs and known-derived pseudo unknowns."""
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    fit_inputs = fit_inputs.float().to(device)
    fit_labels = fit_labels.long().to(device)
    calibration_inputs = calibration_inputs.float().to(device)
    calibration_labels = calibration_labels.long().to(device)
    if (fit_labels < 0).any() or (calibration_labels < 0).any():
        raise ValueError("PNPFF fitting/calibration data must contain known classes only.")
    model_dim = int(
        backbone_feature_dim if backbone_feature_dim is not None else fit_inputs.shape[-1]
    )
    model = PNPFFModel(model_dim, int(num_classes), cfg, backbone=backbone).to(device)
    model.set_feature_normalization(fit_inputs)
    _initialise_prototypes(model, fit_inputs, fit_labels, cfg.seed)

    pseudo_fit = make_pseudo_unknowns(
        fit_inputs,
        fit_labels,
        ratio=cfg.pseudo_unknown_ratio if cfg.pseudo_unknown_enabled else 0.0,
        mixup_alpha=cfg.pseudo_unknown_mixup_alpha,
        mask_probability=cfg.pseudo_unknown_mask_probability,
        noise_std=cfg.pseudo_unknown_noise_std,
        seed=cfg.seed,
    ).to(device)
    pseudo_calibration = make_pseudo_unknowns(
        calibration_inputs,
        calibration_labels,
        ratio=cfg.pseudo_unknown_ratio if cfg.pseudo_unknown_enabled else 0.0,
        mixup_alpha=cfg.pseudo_unknown_mixup_alpha,
        mask_probability=cfg.pseudo_unknown_mask_probability,
        noise_std=cfg.pseudo_unknown_noise_std,
        seed=cfg.seed + 1,
    ).to(device)
    if pseudo_fit.numel() == 0 or pseudo_calibration.numel() == 0:
        raise ValueError(
            "Robust PNPFF requires pseudo_unknown.enabled=true and non-empty pseudo samples."
        )

    shared = [
        p
        for p in list(model.backbone.parameters()) + list(model.projection.parameters())
        if p.requires_grad
    ]
    positive_params = shared + [model.positive_prototypes, model.raw_radius]
    negative_params = shared + [model.negative_prototypes]
    positive_optimizer = torch.optim.SGD(
        positive_params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    negative_optimizer = torch.optim.SGD(
        negative_params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    positive_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        positive_optimizer, list(cfg.lr_milestones), gamma=cfg.lr_gamma
    )
    negative_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        negative_optimizer, list(cfg.lr_milestones), gamma=cfg.lr_gamma
    )
    loader = _balanced_loader(fit_inputs, fit_labels, cfg)
    pseudo_loader = DataLoader(
        TensorDataset(pseudo_fit.detach().cpu()),
        batch_size=max(1, cfg.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed + 2),
    )

    best_known = -1.0
    best_objective = -1.0
    best_selected_balanced = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(max(1, cfg.epochs)):
        model.train()
        totals = {
            "positive": 0.0,
            "negative": 0.0,
            "open_positive": 0.0,
            "open_negative": 0.0,
            "batches": 0.0,
        }
        pseudo_iter = iter(pseudo_loader)
        for batch_inputs, batch_labels in loader:
            try:
                (pseudo_batch,) = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                (pseudo_batch,) = next(pseudo_iter)
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            pseudo_batch = pseudo_batch.to(device)

            positive_optimizer.zero_grad(set_to_none=True)
            positive = model.positive_loss(batch_inputs, batch_labels)
            open_positive, _ = model.open_space_loss(pseudo_batch)
            positive_total = positive["total"] + float(cfg.lambda_unknown) * open_positive
            positive_total.backward()
            torch.nn.utils.clip_grad_norm_(positive_params, max_norm=float(cfg.grad_clip_norm))
            positive_optimizer.step()
            model.retract_prototypes()

            negative_optimizer.zero_grad(set_to_none=True)
            negative = model.negative_loss(batch_inputs, batch_labels)
            _, open_negative = model.open_space_loss(pseudo_batch)
            negative_total = negative + float(cfg.lambda_unknown) * open_negative
            negative_total.backward()
            torch.nn.utils.clip_grad_norm_(negative_params, max_norm=float(cfg.grad_clip_norm))
            negative_optimizer.step()
            model.retract_prototypes()

            totals["positive"] += float(positive_total.detach())
            totals["negative"] += float(negative_total.detach())
            totals["open_positive"] += float(open_positive.detach())
            totals["open_negative"] += float(open_negative.detach())
            totals["batches"] += 1.0
        positive_scheduler.step()
        negative_scheduler.step()

        model.eval()
        with torch.no_grad():
            known_out = model.probabilities(calibration_inputs)
            pseudo_out = model.probabilities(pseudo_calibration)
        known_pred = known_out["predicted_class"].cpu().numpy()
        known_labels_np = calibration_labels.cpu().numpy()
        balanced_acc = float(balanced_accuracy_score(known_labels_np, known_pred))
        ordinary_acc = float(np.mean(known_pred == known_labels_np))
        known_raw = known_out["raw_unknown_score"].cpu().numpy()
        pseudo_raw = pseudo_out["raw_unknown_score"].cpu().numpy()
        pseudo_auc = _safe_auc(known_raw, pseudo_raw)
        harmonic = 2.0 * balanced_acc * pseudo_auc / max(balanced_acc + pseudo_auc, EPS)
        best_known = max(best_known, balanced_acc)
        if best_selected_balanced < best_known - float(cfg.known_accuracy_tolerance):
            best_objective = -1.0
        eligible = balanced_acc >= best_known - float(cfg.known_accuracy_tolerance)
        if eligible and harmonic > best_objective:
            best_objective = harmonic
            best_selected_balanced = balanced_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        history.append(
            {
                "epoch": float(epoch + 1),
                "positive_loss": totals["positive"] / max(totals["batches"], 1.0),
                "negative_loss": totals["negative"] / max(totals["batches"], 1.0),
                "open_positive_loss": totals["open_positive"] / max(totals["batches"], 1.0),
                "open_negative_loss": totals["open_negative"] / max(totals["batches"], 1.0),
                "calibration_accuracy": ordinary_acc,
                "calibration_balanced_accuracy": balanced_acc,
                "pseudo_unknown_auroc": pseudo_auc,
                "selection_objective": harmonic,
                "learning_rate": float(positive_optimizer.param_groups[0]["lr"]),
                "radius": float(model.radius.detach()),
            }
        )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        known_raw = model.probabilities(calibration_inputs)["raw_unknown_score"].cpu().numpy()
        pseudo_raw = model.probabilities(pseudo_calibration)["raw_unknown_score"].cpu().numpy()
    calibrator = MonotonicCalibrator.fit(known_raw, pseudo_raw)
    known_scores = calibrator.predict(known_raw)
    pseudo_scores = calibrator.predict(pseudo_raw)
    mode = str(cfg.threshold_mode).lower()
    if mode == "fixed":
        threshold, threshold_fpr, pseudo_f1 = (
            float(cfg.tau),
            float(np.mean(known_scores > cfg.tau)),
            0.0,
        )
    elif mode in {"known_fpr", "quantile", "validation_quantile"}:
        threshold = float(np.quantile(known_scores, 1.0 - float(cfg.target_known_fpr)))
        threshold_fpr, pseudo_f1 = float(np.mean(known_scores > threshold)), 0.0
    else:
        threshold, threshold_fpr, pseudo_f1 = _choose_threshold(
            known_scores, pseudo_scores, target_known_fpr=cfg.target_known_fpr
        )
    calibrated_auc = _safe_auc(known_scores, pseudo_scores)
    finite = bool(
        np.isfinite(known_scores).all()
        and np.isfinite(pseudo_scores).all()
        and all(torch.isfinite(v).all().item() for v in model.state_dict().values())
    )
    reasons: list[str] = []
    if not finite:
        reasons.append("non_finite_scores_or_parameters")
    if calibrated_auc < float(cfg.health_min_auroc):
        reasons.append("pseudo_unknown_auroc_below_minimum")
    if float(np.median(pseudo_scores)) <= float(np.median(known_scores)):
        reasons.append("pseudo_unknown_median_not_above_known")
    if threshold_fpr > float(cfg.target_known_fpr) + float(cfg.health_fpr_tolerance):
        reasons.append("known_false_unknown_rate_above_tolerance")
    health = {
        "healthy": not reasons,
        "reasons": reasons,
        "calibration_pseudo_unknown_auroc": calibrated_auc,
        "known_score_median": float(np.median(known_scores)),
        "pseudo_unknown_score_median": float(np.median(pseudo_scores)),
        "known_false_unknown_rate": threshold_fpr,
        "pseudo_unknown_f1": pseudo_f1,
        "positive_probability_saturation_rate": float(
            np.mean(
                np.max(
                    model.probabilities(calibration_inputs)["positive_probabilities"]
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=1,
                )
                > 0.999
            )
        ),
        "negative_probability_saturation_rate": float(
            np.mean(
                np.max(
                    model.probabilities(calibration_inputs)["negative_probabilities"]
                    .detach()
                    .cpu()
                    .numpy(),
                    axis=1,
                )
                > 0.999
            )
        ),
        "positive_prototype_norm_mean": float(
            model.positive_prototypes.detach().norm(dim=-1).mean()
        ),
        "negative_prototype_norm_mean": float(
            model.negative_prototypes.detach().norm(dim=-1).mean()
        ),
    }
    return PNPFFDetector(
        model=model.cpu(),
        threshold=threshold,
        config=cfg,
        history=history,
        fit_size=int(fit_labels.numel()),
        calibration_size=int(calibration_labels.numel()),
        pseudo_fit_size=int(pseudo_fit.shape[0]),
        pseudo_calibration_size=int(pseudo_calibration.shape[0]),
        calibrator=calibrator,
        health=health,
        best_epoch=best_epoch,
    )
