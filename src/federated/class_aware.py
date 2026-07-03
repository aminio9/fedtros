from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

EPS = 1e-8


def parse_class_vector(value: Any, num_classes: int) -> np.ndarray:
    """Parse a class-indexed JSON dict/list into a fixed-length float vector."""
    vector = np.zeros(int(num_classes), dtype=np.float64)
    if value is None:
        return vector

    raw = value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return vector

    if isinstance(raw, Mapping):
        for key, item in raw.items():
            try:
                idx = int(key)
                if 0 <= idx < num_classes:
                    vector[idx] = float(item)
            except (TypeError, ValueError):
                continue
        return vector

    if isinstance(raw, (list, tuple)):
        for idx, item in enumerate(raw[:num_classes]):
            try:
                vector[idx] = float(item)
            except (TypeError, ValueError):
                vector[idx] = 0.0
        return vector

    return vector


def class_rarity_vector(global_counts: np.ndarray, *, smoothing: float) -> np.ndarray:
    """Return inverse-frequency class weights normalized to mean one."""
    counts = np.asarray(global_counts, dtype=np.float64)
    if counts.size == 0:
        return counts
    smoothed = counts + max(float(smoothing), 0.0)
    total = float(smoothed.sum())
    if total <= EPS:
        return np.ones_like(counts, dtype=np.float64)

    probabilities = smoothed / total
    rarity = 1.0 / np.maximum(probabilities, EPS)
    return rarity / max(float(np.mean(rarity)), EPS)


def profile_cluster_id(label_counts: np.ndarray) -> str:
    """Deterministic traffic-profile cluster from the dominant local class."""
    counts = np.asarray(label_counts, dtype=np.float64)
    if counts.size == 0 or float(counts.sum()) <= EPS:
        return "empty"
    dominant = int(np.argmax(counts))
    coverage = int(np.count_nonzero(counts > 0.0))
    return f"class_{dominant}_coverage_{coverage}"


def client_class_score(
    label_counts: np.ndarray,
    rarity: np.ndarray,
    *,
    per_class_quality: np.ndarray | None = None,
) -> float:
    """Score how much a client contributes to rare, well-modeled classes."""
    counts = np.asarray(label_counts, dtype=np.float64)
    if counts.size == 0 or float(counts.sum()) <= EPS:
        return 1.0
    support = counts / max(float(counts.sum()), EPS)
    class_values = np.asarray(rarity, dtype=np.float64)
    if per_class_quality is not None and per_class_quality.size == class_values.size:
        quality = np.clip(np.asarray(per_class_quality, dtype=np.float64), 0.0, 1.0)
        known_quality = quality > 0.0
        if bool(np.any(known_quality)):
            class_values = class_values * np.where(known_quality, quality, 1.0)
    return float(np.dot(support, class_values))


def class_aware_aggregation_records(
    records: list[dict[str, Any]],
    *,
    num_classes: int,
    rare_class_strength: float,
    quality_weight_blend: float,
    cluster_balance_strength: float,
    min_multiplier: float,
    max_multiplier: float,
    label_smoothing: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Build FedMADE-inspired aggregation weights from local class profiles.

    The server keeps FedAvg's sample-count prior, then applies:
    - a rare-class multiplier from inverse global class frequency,
    - a quality multiplier from local macro metrics/per-class recall,
    - a cluster balancing multiplier so many similar majority clients do not
      dominate a round.
    """
    if not records:
        return []

    lower = max(float(min_multiplier), EPS)
    upper = max(float(max_multiplier), lower)
    class_strength = max(float(rare_class_strength), 0.0)
    quality_blend = float(np.clip(quality_weight_blend, 0.0, 1.0))
    cluster_strength = max(float(cluster_balance_strength), 0.0)

    parsed_records: list[dict[str, Any]] = []
    global_counts = np.zeros(int(num_classes), dtype=np.float64)
    for record in records:
        label_counts = parse_class_vector(record.get("label_histogram"), num_classes)
        parsed = {**record, "label_counts": label_counts}
        parsed_records.append(parsed)
        global_counts += label_counts

    rarity = class_rarity_vector(global_counts, smoothing=label_smoothing)
    raw_scores = []
    for record in parsed_records:
        per_class_quality = parse_class_vector(record.get("per_class_recall"), num_classes)
        score = client_class_score(
            record["label_counts"],
            rarity,
            per_class_quality=per_class_quality,
        )
        record["class_score"] = score
        record["cluster_id"] = profile_cluster_id(record["label_counts"])
        raw_scores.append(score)

    mean_score = max(float(np.mean(raw_scores)), EPS)
    quality_values = [
        float(np.clip(record.get("quality", 0.0), 0.0, 1.0)) for record in parsed_records
    ]
    mean_quality = max(float(np.mean(quality_values)), EPS)

    cluster_masses: dict[str, float] = {}
    for record in parsed_records:
        cluster_id = str(record["cluster_id"])
        cluster_masses[cluster_id] = cluster_masses.get(cluster_id, 0.0) + max(
            float(record.get("num_examples", 0.0)),
            1.0,
        )
    mean_cluster_mass = max(float(np.mean(list(cluster_masses.values()))), EPS)

    weighted: list[dict[str, Any]] = []
    for record, quality in zip(parsed_records, quality_values, strict=True):
        num_examples = max(float(record.get("num_examples", 0.0)), 1.0)

        centered_class_score = (float(record["class_score"]) / mean_score) - 1.0
        class_multiplier = float(
            np.clip(1.0 + (class_strength * centered_class_score), lower, upper)
        )

        relative_quality = quality / mean_quality
        quality_multiplier = float(
            np.clip((1.0 - quality_blend) + (quality_blend * relative_quality), lower, upper)
        )

        cluster_mass = max(cluster_masses[str(record["cluster_id"])], EPS)
        cluster_multiplier = float(
            np.clip((mean_cluster_mass / cluster_mass) ** cluster_strength, lower, upper)
        )

        aggregation_weight = (
            num_examples * class_multiplier * quality_multiplier * cluster_multiplier
        )
        weighted.append(
            {
                **record,
                "class_multiplier": class_multiplier,
                "quality_multiplier": quality_multiplier,
                "cluster_multiplier": cluster_multiplier,
                "aggregation_weight": float(max(aggregation_weight, EPS)),
            }
        )

    return weighted
