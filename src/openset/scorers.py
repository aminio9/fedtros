from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from src.openset.thresholding import predict_known_unknown, select_validation_threshold
from src.training.losses import energy_score


class OpenSetScorer(Protocol):
    """Protocol for validation-fitted open-set scoring baselines.

    Convention: all implemented scorers return larger scores for samples that
    look more likely to be unknown.
    """

    name: str
    higher_is_unknown: bool
    requires_fit: bool

    def fit(
        self,
        validation_features: np.ndarray | torch.Tensor,
        validation_labels: np.ndarray | torch.Tensor,
        known_labels: list[int] | tuple[int, ...] | np.ndarray,
    ) -> "OpenSetScorer":
        ...

    def score(
        self,
        features: np.ndarray | torch.Tensor | None = None,
        *,
        logits: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PrototypeSet:
    """Class prototypes fitted from validation/train embeddings."""

    prototypes: torch.Tensor
    counts: torch.Tensor


@dataclass(frozen=True)
class DiagonalMahalanobisModel:
    """Per-class diagonal Gaussian parameters for cheap distance scoring."""

    means: torch.Tensor
    variances: torch.Tensor
    counts: torch.Tensor


def _to_tensor(values: np.ndarray | torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.detach().to(dtype=dtype)
    return torch.as_tensor(values, dtype=dtype)


def _to_labels(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.detach().long().view(-1)
    return torch.as_tensor(values, dtype=torch.long).view(-1)


def _cfg_get(config: object, key: str, default: object = None) -> object:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def build_open_set_scorer_from_config(open_set_cfg: object) -> OpenSetScorer:
    """Build a standalone open-set scorer utility from an open_set config node.

    EVT reconstruction is intentionally excluded here because the implemented
    EVT path is coupled to the CVAE-DQN agent in ``src.evaluation.open_set``.
    """
    scorer_cfg = _cfg_get(open_set_cfg, "scorer", open_set_cfg)
    scorer_name = str(_cfg_get(scorer_cfg, "name", _cfg_get(open_set_cfg, "name", ""))).lower()
    if scorer_name == "msp":
        return MSPScorer(temperature=float(_cfg_get(scorer_cfg, "temperature", 1.0)))
    if scorer_name == "energy":
        return EnergyScorer(temperature=float(_cfg_get(scorer_cfg, "temperature", 1.0)))
    if scorer_name in {"prototype", "prototype_distance"}:
        return PrototypeDistanceScorer()
    if scorer_name in {"mahalanobis", "mahalanobis_distance"}:
        regularization = float(_cfg_get(scorer_cfg, "regularization", 1e-4))
        return MahalanobisDistanceScorer(regularization=regularization)
    if scorer_name == "no_rejection":
        return NoRejectionScorer()
    if scorer_name in {"evt_reconstruction", "openmax_evt_reconstruction"}:
        raise ValueError(
            f"open-set scorer {scorer_name!r} is handled by the CVAE-DQN EVT "
            "evaluation path, not by the standalone scorer factory."
        )
    raise ValueError(f"Unsupported open-set scorer {scorer_name!r}.")


def msp_unknown_score(logits: torch.Tensor) -> torch.Tensor:
    """Maximum-softmax-probability unknown score; larger means more unknown-like."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [B, C], got {tuple(logits.shape)}")
    max_prob = F.softmax(logits, dim=1).max(dim=1).values
    return 1.0 - max_prob


def energy_unknown_score(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    """Energy unknown score; larger values indicate weaker known-class evidence."""
    return energy_score(logits, temperature=temperature)


def fit_class_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int | None = None,
    normalize: bool = False,
) -> PrototypeSet:
    """Fit one mean feature prototype per known class."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [B, D], got {tuple(features.shape)}")
    targets = labels.view(-1).long()
    if targets.numel() != features.size(0):
        raise ValueError("labels and features batch size must match.")
    if num_classes is None:
        valid = targets >= 0
        if not bool(valid.any()):
            raise ValueError("Cannot infer num_classes without non-negative labels.")
        num_classes = int(targets[valid].max().item()) + 1

    work_features = F.normalize(features.float(), dim=1) if normalize else features.float()
    prototypes = torch.zeros(int(num_classes), features.size(1), device=features.device)
    counts = torch.zeros(int(num_classes), device=features.device)
    for class_id in range(int(num_classes)):
        mask = targets.eq(class_id)
        if bool(mask.any()):
            prototypes[class_id] = work_features[mask].mean(dim=0)
            counts[class_id] = float(mask.sum().item())
    return PrototypeSet(prototypes=prototypes, counts=counts)


def prototype_distance_unknown_score(
    features: torch.Tensor,
    prototypes: torch.Tensor | PrototypeSet,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Minimum squared distance to known-class prototypes; larger means unknown-like."""
    proto_tensor = prototypes.prototypes if isinstance(prototypes, PrototypeSet) else prototypes
    if features.ndim != 2 or proto_tensor.ndim != 2:
        raise ValueError("features and prototypes must both be 2D tensors.")
    if features.size(1) != proto_tensor.size(1):
        raise ValueError("features and prototypes must have the same feature dimension.")
    work_features = F.normalize(features.float(), dim=1) if normalize else features.float()
    work_prototypes = F.normalize(proto_tensor.float(), dim=1) if normalize else proto_tensor.float()
    distances = torch.cdist(work_features, work_prototypes, p=2).pow(2)
    return distances.min(dim=1).values


def fit_diagonal_mahalanobis(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int | None = None,
    eps: float = 1e-4,
) -> DiagonalMahalanobisModel:
    """Fit a diagonal Gaussian per class for distance-based unknown scoring."""
    proto = fit_class_prototypes(features, labels, num_classes=num_classes)
    targets = labels.view(-1).long()
    variances = torch.ones_like(proto.prototypes)
    for class_id in range(proto.prototypes.size(0)):
        mask = targets.eq(class_id)
        if bool(mask.any()):
            centered = features.float()[mask] - proto.prototypes[class_id]
            variances[class_id] = centered.pow(2).mean(dim=0).clamp_min(float(eps))
    return DiagonalMahalanobisModel(
        means=proto.prototypes,
        variances=variances,
        counts=proto.counts,
    )


def mahalanobis_unknown_score(
    features: torch.Tensor,
    model: DiagonalMahalanobisModel,
) -> torch.Tensor:
    """Minimum diagonal Mahalanobis distance to known classes."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [B, D], got {tuple(features.shape)}")
    if features.size(1) != model.means.size(1):
        raise ValueError("features and Mahalanobis means must have the same feature dimension.")
    diff = features.float().unsqueeze(1) - model.means.float().unsqueeze(0)
    distances = diff.pow(2).div(model.variances.float().unsqueeze(0).clamp_min(1e-12)).sum(dim=2)
    return distances.min(dim=1).values


def select_threshold_from_validation(
    scores: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    unknown_label_id: int = -1,
    target_known_fpr: float = 0.05,
    known_mask: torch.Tensor | None = None,
    mode: str = "validation_known_fpr",
    fixed_threshold: float | None = None,
    score_direction: str = "higher_unknown",
) -> float:
    """Choose an unknown threshold from validation known scores."""
    return select_validation_threshold(
        scores,
        labels,
        unknown_label_id=unknown_label_id,
        known_mask=known_mask,
        target_known_fpr=target_known_fpr,
        mode=mode,
        fixed_threshold=fixed_threshold,
        score_direction=score_direction,
    )


def predict_unknown_from_scores(
    scores: torch.Tensor,
    threshold: float,
    *,
    score_direction: str = "higher_unknown",
) -> torch.Tensor:
    """Return a boolean unknown mask from scores and threshold."""
    predictions = predict_known_unknown(
        scores,
        threshold,
        score_direction=score_direction,
    )
    return torch.as_tensor(predictions, dtype=torch.bool, device=scores.device)


@dataclass
class MSPScorer:
    """Maximum softmax probability baseline; score is 1 - max softmax probability."""

    temperature: float = 1.0
    name: str = "msp"
    higher_is_unknown: bool = True
    requires_fit: bool = False

    def fit(self, validation_features, validation_labels, known_labels) -> "MSPScorer":
        _ = validation_features, validation_labels, known_labels
        return self

    def score(self, features=None, *, logits=None) -> np.ndarray:
        _ = features
        if logits is None:
            raise ValueError("MSPScorer requires logits.")
        logits_tensor = _to_tensor(logits)
        probs = F.softmax(logits_tensor / max(float(self.temperature), 1e-12), dim=1)
        return (1.0 - probs.max(dim=1).values).cpu().numpy()


@dataclass
class EnergyScorer:
    """Energy baseline; score is -T logsumexp(logits/T), higher means more unknown."""

    temperature: float = 1.0
    name: str = "energy"
    higher_is_unknown: bool = True
    requires_fit: bool = False

    def fit(self, validation_features, validation_labels, known_labels) -> "EnergyScorer":
        _ = validation_features, validation_labels, known_labels
        return self

    def score(self, features=None, *, logits=None) -> np.ndarray:
        _ = features
        if logits is None:
            raise ValueError("EnergyScorer requires logits.")
        logits_tensor = _to_tensor(logits)
        temperature = max(float(self.temperature), 1e-12)
        energy = -temperature * torch.logsumexp(logits_tensor / temperature, dim=1)
        return energy.cpu().numpy()


@dataclass
class PrototypeDistanceScorer:
    """Nearest class-prototype distance baseline over embeddings or raw features."""

    name: str = "prototype_distance"
    higher_is_unknown: bool = True
    requires_fit: bool = True
    prototypes: torch.Tensor | None = None

    def fit(self, validation_features, validation_labels, known_labels) -> "PrototypeDistanceScorer":
        features = _to_tensor(validation_features)
        labels = _to_labels(validation_labels)
        known = torch.as_tensor(list(known_labels), dtype=torch.long)
        prototypes: list[torch.Tensor] = []
        for label in known:
            mask = labels.eq(label)
            if bool(mask.any()):
                prototypes.append(features[mask].mean(dim=0))
        if not prototypes:
            raise ValueError("PrototypeDistanceScorer requires at least one known class.")
        self.prototypes = torch.stack(prototypes, dim=0)
        return self

    def score(self, features=None, *, logits=None) -> np.ndarray:
        _ = logits
        if features is None:
            raise ValueError("PrototypeDistanceScorer requires features or embeddings.")
        if self.prototypes is None:
            raise RuntimeError("PrototypeDistanceScorer must be fit before score().")
        feature_tensor = _to_tensor(features)
        distances = torch.cdist(feature_tensor, self.prototypes.to(feature_tensor.device), p=2)
        return distances.min(dim=1).values.cpu().numpy()


@dataclass
class MahalanobisDistanceScorer:
    """Shared-covariance Mahalanobis distance baseline over embeddings or features."""

    regularization: float = 1e-4
    name: str = "mahalanobis_distance"
    higher_is_unknown: bool = True
    requires_fit: bool = True
    means: torch.Tensor | None = None
    precision: torch.Tensor | None = None

    def fit(self, validation_features, validation_labels, known_labels) -> "MahalanobisDistanceScorer":
        features = _to_tensor(validation_features)
        labels = _to_labels(validation_labels)
        known = torch.as_tensor(list(known_labels), dtype=torch.long)
        means: list[torch.Tensor] = []
        centered: list[torch.Tensor] = []
        for label in known:
            mask = labels.eq(label)
            if not bool(mask.any()):
                continue
            class_features = features[mask]
            mean = class_features.mean(dim=0)
            means.append(mean)
            centered.append(class_features - mean)
        if not means:
            raise ValueError("MahalanobisDistanceScorer requires at least one known class.")
        residuals = torch.cat(centered, dim=0)
        dim = features.size(1)
        denominator = max(int(residuals.size(0)) - len(means), 1)
        covariance = residuals.T.matmul(residuals) / float(denominator)
        covariance = covariance + float(self.regularization) * torch.eye(
            dim,
            dtype=features.dtype,
            device=features.device,
        )
        self.means = torch.stack(means, dim=0)
        self.precision = torch.linalg.pinv(covariance)
        return self

    def score(self, features=None, *, logits=None) -> np.ndarray:
        _ = logits
        if features is None:
            raise ValueError("MahalanobisDistanceScorer requires features or embeddings.")
        if self.means is None or self.precision is None:
            raise RuntimeError("MahalanobisDistanceScorer must be fit before score().")
        feature_tensor = _to_tensor(features)
        means = self.means.to(feature_tensor.device)
        precision = self.precision.to(feature_tensor.device)
        diffs = feature_tensor.unsqueeze(1) - means.unsqueeze(0)
        distances = torch.einsum("nkd,df,nkf->nk", diffs, precision, diffs).clamp_min(0.0)
        return distances.min(dim=1).values.cpu().numpy()


@dataclass
class NoRejectionScorer:
    """Closed-set baseline: never rejects as unknown."""

    name: str = "no_rejection"
    higher_is_unknown: bool = True
    requires_fit: bool = False

    def fit(self, validation_features, validation_labels, known_labels) -> "NoRejectionScorer":
        _ = validation_features, validation_labels, known_labels
        return self

    def score(self, features=None, *, logits=None) -> np.ndarray:
        values = logits if logits is not None else features
        if values is None:
            raise ValueError("NoRejectionScorer needs features or logits to infer sample count.")
        count = int(_to_tensor(values).shape[0])
        return np.zeros(count, dtype=np.float32)

    def select_threshold(self) -> float:
        return float("inf")
