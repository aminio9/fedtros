"""Paper-style Positive-Negative Prototypes Fusion Framework (PNPFF).

This module implements the base PNPFF equations from Zhong and Cui (2025) as
a post-federation detector over frozen student-backbone features.  It does not
implement the APNPFF/APNPFF++ adversarial extensions.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PNPFFConfig:
    feature_dim: int = 128
    num_positive_prototypes: int = 7
    gamma1: float = 1.0
    gamma2: float = 1.0
    lambda0: float = 0.1
    lambda1: float = 0.1
    lambda2: float = 0.01
    diversity_temperature: float = 10.0
    eta: float = 0.5
    omega: float = 0.5
    fit_fraction: float = 0.70
    seed: int = 42
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_milestones: tuple[int, ...] = (30, 60, 90)
    lr_gamma: float = 0.1
    threshold_mode: str = "fixed"
    tau: float = 0.5
    target_known_fpr: float = 0.05

    @classmethod
    def from_cfg(cls, cfg: Any, *, feature_dim: int, batch_size: int) -> "PNPFFConfig":
        p = getattr(cfg, "pnpff", None)

        def value(name: str, default: Any) -> Any:
            return getattr(p, name, default) if p is not None else default

        milestones = value("lr_milestones", [30, 60, 90])
        return cls(
            feature_dim=int(value("feature_dim", feature_dim)),
            num_positive_prototypes=int(value("num_positive_prototypes", 7)),
            gamma1=float(value("gamma1", 1.0)),
            gamma2=float(value("gamma2", 1.0)),
            lambda0=float(value("lambda0", 0.1)),
            lambda1=float(value("lambda1", 0.1)),
            lambda2=float(value("lambda2", 0.01)),
            diversity_temperature=float(value("diversity_temperature", 10.0)),
            eta=float(value("eta", 0.5)),
            omega=float(value("omega", 0.5)),
            fit_fraction=float(value("fit_fraction", 0.70)),
            seed=int(value("seed", 42)),
            epochs=int(value("epochs", 100)),
            batch_size=int(value("batch_size", batch_size)),
            learning_rate=float(value("learning_rate", 0.1)),
            momentum=float(value("momentum", 0.9)),
            weight_decay=float(value("weight_decay", 0.0)),
            lr_milestones=tuple(int(v) for v in milestones),
            lr_gamma=float(value("lr_gamma", 0.1)),
            threshold_mode=str(value("threshold_mode", "fixed")),
            tau=float(value("tau", 0.5)),
            target_known_fpr=float(value("target_known_fpr", 0.05)),
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
    if np.any(y == int(unknown_label_id)):
        raise ValueError("PNPFF fitting/calibration data must contain known classes only.")
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("PNPFF fit_fraction must be strictly between 0 and 1.")
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2 or np.any(counts < 2):
        raise ValueError("PNPFF stratified split requires at least two classes and two samples per class.")
    indices = np.arange(y.size)
    fit_idx, calibration_idx = train_test_split(
        indices,
        train_size=float(fit_fraction),
        random_state=int(seed),
        shuffle=True,
        stratify=y,
    )
    return np.sort(fit_idx), np.sort(calibration_idx)


class PNPFFModel(nn.Module):
    """Learnable projection and class-specific positive/negative prototypes."""

    def __init__(self, input_dim: int, num_classes: int, cfg: PNPFFConfig):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.cfg = cfg
        self.projection = nn.Linear(self.input_dim, int(cfg.feature_dim), bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            diagonal = min(self.input_dim, int(cfg.feature_dim))
            self.projection.weight[:diagonal, :diagonal] = torch.eye(diagonal)
        self.positive_prototypes = nn.Parameter(
            torch.empty(self.num_classes, int(cfg.num_positive_prototypes), int(cfg.feature_dim))
        )
        self.negative_prototypes = nn.Parameter(torch.empty(self.num_classes, int(cfg.feature_dim)))
        self.radius = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.positive_prototypes, std=0.02)
        nn.init.normal_(self.negative_prototypes, std=0.02)

    @staticmethod
    def distance(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Equation 1: squared Euclidean / m minus dot product."""
        m = max(int(features.shape[-1]), 1)
        diff = features - prototypes
        euclidean = diff.pow(2).sum(dim=-1) / float(m)
        dot = (features * prototypes).sum(dim=-1)
        return euclidean - dot

    def distances(self, backbone_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.projection(backbone_features)
        # [B, K, V] -> minimum over the V positive prototypes (Equation 2).
        pos_all = self.distance(z[:, None, None, :], self.positive_prototypes[None, :, :, :])
        pos = pos_all.min(dim=-1).values
        neg = self.distance(z[:, None, :], self.negative_prototypes[None, :, :])
        return z, pos, neg

    def probabilities(self, backbone_features: torch.Tensor) -> dict[str, torch.Tensor]:
        z, pos_dist, neg_dist = self.distances(backbone_features)
        positive = F.softmax(-float(self.cfg.gamma1) * pos_dist, dim=1)
        negative = F.softmax(float(self.cfg.gamma2) * neg_dist, dim=1)
        fused = (float(self.cfg.eta) * positive) + (float(self.cfg.omega) * negative)
        known_confidence, predicted_class = fused.max(dim=1)
        unknown_score = (1.0 - known_confidence).clamp(0.0, 1.0)
        return {
            "features": z,
            "positive_distances": pos_dist,
            "negative_distances": neg_dist,
            "positive_probabilities": positive,
            "negative_probabilities": negative,
            "fused_scores": fused,
            "known_confidence": known_confidence,
            "unknown_score": unknown_score,
            "predicted_class": predicted_class,
        }

    def positive_loss(self, backbone_features: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        z, pos_dist, _ = self.distances(backbone_features)
        classification = F.cross_entropy(-float(self.cfg.gamma1) * pos_dist, labels)
        class_prototypes = self.positive_prototypes[labels]
        class_average = class_prototypes.mean(dim=1)
        moving = F.relu(((z - class_average).pow(2).sum(dim=1) / float(self.cfg.feature_dim)) - self.radius.clamp_min(0.0)).mean()

        p = self.positive_prototypes
        pairwise = self.distance(p.unsqueeze(2), p.unsqueeze(1))
        mask = ~torch.eye(p.shape[1], dtype=torch.bool, device=p.device).unsqueeze(0)
        local_diversity = -pairwise.masked_select(mask).mean() if p.shape[1] > 1 else p.sum() * 0.0

        assigned_dist = self.distance(z.unsqueeze(1), class_prototypes)
        assignments = F.softmax(float(self.cfg.diversity_temperature) * assigned_dist, dim=1)
        mean_assignment = assignments.mean(dim=0).clamp_min(1.0e-8)
        uniform = torch.full_like(mean_assignment, 1.0 / float(mean_assignment.numel()))
        assignment_diversity = (mean_assignment * (mean_assignment.log() - uniform.log())).sum()
        diversity = local_diversity + assignment_diversity
        total = classification + (float(self.cfg.lambda1) * moving) + (float(self.cfg.lambda2) * diversity)
        return {"total": total, "classification": classification, "moving": moving, "diversity": diversity}

    def negative_loss(self, backbone_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, _, neg_dist = self.distances(backbone_features)
        return float(self.cfg.lambda0) * F.cross_entropy(float(self.cfg.gamma2) * neg_dist, labels)


@dataclass
class PNPFFDetector:
    model: PNPFFModel
    threshold: float
    config: PNPFFConfig
    history: list[dict[str, float]]
    fit_size: int
    calibration_size: int

    def predict_features(self, features: torch.Tensor, *, device: torch.device | str = "cpu") -> dict[str, np.ndarray]:
        target = torch.device(device)
        self.model.to(target).eval()
        with torch.no_grad():
            values = self.model.probabilities(features.to(target).float())
        return {key: value.detach().cpu().numpy() for key, value in values.items() if key != "features"}

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": "pnpff",
            "threshold": float(self.threshold),
            "config": asdict(self.config),
            "fit_size": int(self.fit_size),
            "calibration_size": int(self.calibration_size),
            "history": self.history,
            "positive_prototypes": self.model.positive_prototypes.detach().cpu().tolist(),
            "negative_prototypes": self.model.negative_prototypes.detach().cpu().tolist(),
            "radius": float(self.model.radius.detach().clamp_min(0.0).cpu().item()),
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
            },
            path,
        )


def _initialise_prototypes(model: PNPFFModel, features: torch.Tensor, labels: torch.Tensor, seed: int) -> None:
    generator = torch.Generator(device=features.device).manual_seed(int(seed))
    with torch.no_grad():
        projected = model.projection(features)
        global_mean = projected.mean(dim=0)
        for class_id in range(model.num_classes):
            class_features = projected[labels == class_id]
            if class_features.numel() == 0:
                raise ValueError(f"PNPFF fit split is missing class {class_id}.")
            order = torch.randperm(class_features.shape[0], generator=generator, device=features.device)
            selected = class_features[order[: min(order.numel(), model.cfg.num_positive_prototypes)]]
            if selected.shape[0] < model.cfg.num_positive_prototypes:
                repeats = int(np.ceil(model.cfg.num_positive_prototypes / selected.shape[0]))
                selected = selected.repeat((repeats, 1))
            model.positive_prototypes[class_id].copy_(selected[: model.cfg.num_positive_prototypes])
            other = projected[labels != class_id]
            model.negative_prototypes[class_id].copy_(other.mean(dim=0) if other.numel() else global_mean)


def fit_pnpff_detector(
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    calibration_features: torch.Tensor,
    calibration_labels: torch.Tensor,
    *,
    num_classes: int,
    cfg: PNPFFConfig,
    device: torch.device,
) -> PNPFFDetector:
    """Fit PNPFF with alternating positive and negative optimization steps."""
    torch.manual_seed(int(cfg.seed))
    fit_features = fit_features.float().to(device)
    fit_labels = fit_labels.long().to(device)
    calibration_features = calibration_features.float().to(device)
    calibration_labels = calibration_labels.long().to(device)
    model = PNPFFModel(int(fit_features.shape[1]), int(num_classes), cfg).to(device)
    _initialise_prototypes(model, fit_features, fit_labels, cfg.seed)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.learning_rate),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(cfg.lr_milestones), gamma=float(cfg.lr_gamma))
    loader = DataLoader(
        TensorDataset(fit_features.detach().cpu(), fit_labels.detach().cpu()),
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg.seed)),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(max(1, int(cfg.epochs))):
        model.train()
        totals = {"positive": 0.0, "negative": 0.0, "batches": 0.0}
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            positive = model.positive_loss(batch_features, batch_labels)
            positive["total"].backward()
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            negative = model.negative_loss(batch_features, batch_labels)
            negative.backward()
            optimizer.step()
            totals["positive"] += float(positive["total"].detach().item())
            totals["negative"] += float(negative.detach().item())
            totals["batches"] += 1.0
        scheduler.step()
        model.eval()
        with torch.no_grad():
            calibration = model.probabilities(calibration_features)
            accuracy = float((calibration["predicted_class"] == calibration_labels).float().mean().item())
        row = {
            "epoch": float(epoch + 1),
            "positive_loss": totals["positive"] / max(totals["batches"], 1.0),
            "negative_loss": totals["negative"] / max(totals["batches"], 1.0),
            "calibration_accuracy": accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "radius": float(model.radius.detach().clamp_min(0.0).item()),
        }
        history.append(row)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        known_unknown_scores = model.probabilities(calibration_features)["unknown_score"].detach().cpu().numpy()
    mode = str(cfg.threshold_mode).lower()
    if mode in {"known_fpr", "quantile", "validation_quantile"}:
        threshold = float(np.quantile(known_unknown_scores, 1.0 - float(cfg.target_known_fpr)))
    else:
        threshold = float(cfg.tau)
    return PNPFFDetector(
        model=model.cpu(),
        threshold=threshold,
        config=cfg,
        history=history,
        fit_size=int(fit_labels.numel()),
        calibration_size=int(calibration_labels.numel()),
    )
