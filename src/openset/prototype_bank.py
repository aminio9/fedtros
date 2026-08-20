"""Canonical Prototype Bank implementation for FedTROS-PR Open-Set Recognition.

Provides positive class KMeans prototypes, class-adaptive radii, and
known-derived synthetic boundary prototypes for geometric rejection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger("PrototypeBank")
EPS = 1.0e-12


def l2_normalize_np(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """L2 normalize numpy vectors."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        denom = max(float(np.linalg.norm(arr)), eps)
        return arr / denom
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(denom, eps)


@dataclass
class PrototypeBank:
    """Positive/boundary prototype bank for geometric open-set rejection."""

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
        return l2_normalize_np(x, self.eps) if self.normalize else x

    def score(self, features: np.ndarray, class_id: int) -> np.ndarray:
        """Compute prototype-based distance score (higher = further/more suspicious)."""
        p = self.prototypes.get(int(class_id))
        if p is None or p.size == 0:
            return np.full((features.shape[0],), np.nan, dtype=np.float64)
        x = self._prep(features)
        centers = self._prep(p) if self.normalize else np.asarray(p, dtype=np.float64)
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        pos_dist = np.sqrt(np.min(d, axis=1) + self.eps)
        radius = float((self.radii or {}).get(int(class_id), 1.0))
        pos_score = pos_dist / max(radius, self.eps)

        if (
            self.negative_prototypes is None
            or self.negative_prototypes.size == 0
            or self.negative_weight <= 0.0
        ):
            return pos_score.astype(np.float64)
        neg_centers = (
            self._prep(self.negative_prototypes)
            if self.normalize
            else np.asarray(self.negative_prototypes, dtype=np.float64)
        )
        nd = ((x[:, None, :] - neg_centers[None, :, :]) ** 2).sum(axis=2)
        neg_dist = np.sqrt(np.min(nd, axis=1) + self.eps)
        neg_close = np.maximum(
            0.0,
            (float(self.negative_radius) - neg_dist)
            / max(float(self.negative_radius), self.eps),
        )
        return (pos_score + (float(self.negative_weight) * neg_close)).astype(np.float64)

    def to_payload(self) -> dict[str, Any]:
        return {
            "positive_prototypes": {
                str(k): v.tolist() for k, v in sorted(self.prototypes.items())
            },
            "positive_radii": {
                str(k): float(v) for k, v in sorted((self.radii or {}).items())
            },
            "negative_prototypes": (
                self.negative_prototypes.tolist()
                if self.negative_prototypes is not None
                else []
            ),
            "negative_radius": float(self.negative_radius),
            "negative_weight": float(self.negative_weight),
            "normalize": bool(self.normalize),
        }


def make_negative_boundary_features(
    features_by_class: dict[int, np.ndarray],
    *,
    max_samples: int,
    mixup_alpha: float = 1.0,
    noise_std: float = 0.005,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesize boundary samples between known classes using manifold mixup."""
    if rng is None:
        rng = np.random.default_rng(42)
    classes = [c for c, feats in features_by_class.items() if feats.size > 0]
    if len(classes) < 2 or max_samples <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    dim = next(iter(features_by_class.values())).shape[1]
    out = np.zeros((max_samples, dim), dtype=np.float64)
    alpha = max(float(mixup_alpha), 1.0e-3)
    for idx in range(max_samples):
        c1, c2 = rng.choice(classes, size=2, replace=False)
        x1 = features_by_class[c1][rng.integers(0, features_by_class[c1].shape[0])]
        x2 = features_by_class[c2][rng.integers(0, features_by_class[c2].shape[0])]
        lam = float(rng.beta(alpha, alpha))
        lam = 0.5 + (0.5 * (lam - 0.5))  # Keep near midpoints
        mix = (lam * x1) + ((1.0 - lam) * x2)
        if noise_std > 0.0:
            mix = mix + rng.normal(0.0, noise_std, size=dim)
        out[idx] = l2_normalize_np(mix) if normalize else mix
    return out


def fit_prototype_bank(
    features_by_class: dict[int, np.ndarray],
    *,
    num_prototypes_per_class: int = 16,
    num_negative_prototypes: int = 32,
    radius_quantile: float = 0.95,
    negative_weight: float = 0.35,
    normalize: bool = True,
    mixup_alpha: float = 1.0,
    noise_std: float = 0.005,
    seed: int = 42,
) -> PrototypeBank:
    """Fit positive class prototypes and synthetic boundary negative prototypes."""
    rng = np.random.default_rng(seed)
    prototypes: dict[int, np.ndarray] = {}
    radii: dict[int, float] = {}

    for c, feats in sorted(features_by_class.items()):
        if feats.shape[0] == 0:
            continue
        x = l2_normalize_np(feats) if normalize else feats
        k = min(int(num_prototypes_per_class), x.shape[0])
        if k <= 1 or x.shape[0] <= 2:
            centers = np.mean(x, axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, random_state=seed, n_init=5)
            km.fit(x)
            centers = km.cluster_centers_
            if normalize:
                centers = l2_normalize_np(centers)
        prototypes[c] = centers
        d = np.sqrt(((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        min_d = np.min(d, axis=1)
        radii[c] = float(np.quantile(min_d, radius_quantile)) if min_d.size > 0 else 1.0

    neg_protos = None
    neg_radius = 1.0
    if num_negative_prototypes > 0 and len(prototypes) >= 2:
        neg_feats = make_negative_boundary_features(
            features_by_class,
            max_samples=max(num_negative_prototypes * 10, 500),
            mixup_alpha=mixup_alpha,
            noise_std=noise_std,
            normalize=normalize,
            rng=rng,
        )
        if neg_feats.shape[0] >= num_negative_prototypes:
            km_neg = KMeans(n_clusters=num_negative_prototypes, random_state=seed, n_init=5)
            km_neg.fit(neg_feats)
            neg_protos = km_neg.cluster_centers_
            if normalize:
                neg_protos = l2_normalize_np(neg_protos)
            nd = np.sqrt(((neg_feats[:, None, :] - neg_protos[None, :, :]) ** 2).sum(axis=2))
            neg_radius = float(np.quantile(np.min(nd, axis=1), radius_quantile))

    return PrototypeBank(
        prototypes=prototypes,
        radii=radii,
        negative_prototypes=neg_protos,
        negative_radius=neg_radius,
        normalize=normalize,
        negative_weight=negative_weight,
    )
