from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

EPS = 1e-8


def _metric_lookup(metrics: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[float, bool]:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value), True
        except (TypeError, ValueError):
            continue
    return 0.0, False


def gate_utility(
    utility: float,
    *,
    utility_temperature: float,
    max_utility: float,
    utility_threshold: float,
) -> float:
    utility = float(np.clip(utility / max(utility_temperature, EPS), 0.0, max_utility))
    if utility < utility_threshold:
        return 0.0
    return utility


def critic_utility_score(raw_utility: float, *, utility_temperature: float) -> float:
    scaled = max(float(raw_utility) / max(float(utility_temperature), EPS), 0.0)
    return float(scaled / (1.0 + scaled))


def combine_utility_score(
    *,
    audit_score: float,
    critic_score: float,
    critic_blend: float,
) -> float:
    blend = float(np.clip(critic_blend, 0.0, 1.0))
    audit = float(np.clip(audit_score, 0.0, 1.0))
    critic = float(np.clip(critic_score, 0.0, 1.0))
    return float(((1.0 - blend) * audit) + (blend * critic))


def alignment_multiplier(
    cosine: float,
    *,
    alignment_strength: float,
    min_multiplier: float,
    max_multiplier: float,
) -> float:
    """FedAWA-style bounded multiplier from client-vector/update alignment."""
    if abs(float(alignment_strength)) <= EPS:
        return 1.0
    multiplier = float(np.exp(float(alignment_strength) * float(np.clip(cosine, -1.0, 1.0))))
    lower = max(float(min_multiplier), EPS)
    upper = max(float(max_multiplier), lower)
    return float(np.clip(multiplier, lower, upper))


def validation_team_reward(
    metrics: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    default_weights = {
        "closed_set_f1": 0.30,
        "balanced_accuracy": 0.20,
        "open_set_auroc": 0.20,
        "open_set_unknown_f1": 0.15,
        "open_set_rejection": 0.15,
    }
    merged_weights = {**default_weights}
    if weights is not None:
        for key, value in weights.items():
            merged_weights[key] = float(value)

    component_specs = {
        "closed_set_f1": ("f1_macro", "openset_f1_macro", "test/macro_f1", "macro_f1"),
        "balanced_accuracy": ("balanced_accuracy", "test/balanced_accuracy", "openset_known_acc"),
        "open_set_auroc": ("openset_auroc", "open_set/auroc", "auroc"),
        "open_set_unknown_f1": ("openset_unknown_f1", "open_set/unknown_f1"),
    }
    components: dict[str, float] = {}
    for name, aliases in component_specs.items():
        value, available = _metric_lookup(metrics, aliases)
        if available:
            components[name] = float(np.clip(value, 0.0, 1.0))

    rejection_terms: list[float] = []
    unknown_recall, has_unknown_recall = _metric_lookup(
        metrics, ("openset_unknown_recall", "open_set/unknown_detection_rate")
    )
    if has_unknown_recall:
        rejection_terms.append(float(np.clip(unknown_recall, 0.0, 1.0)))
    fpr95, has_fpr95 = _metric_lookup(metrics, ("openset_fpr95", "open_set/fpr95"))
    if has_fpr95:
        rejection_terms.append(float(1.0 - np.clip(fpr95, 0.0, 1.0)))
    if rejection_terms:
        components["open_set_rejection"] = float(np.mean(rejection_terms))

    denom = sum(max(float(merged_weights.get(name, 0.0)), 0.0) for name in components) or 1.0
    reward = sum(
        max(float(merged_weights.get(name, 0.0)), 0.0) * components[name]
        for name in components
    ) / denom
    return float(np.clip(reward, 0.0, 1.0))


def centered_utility(
    *,
    score: float,
    round_mean_score: float,
    utility_strength: float,
    min_utility: float,
    max_utility: float,
    utility_threshold: float,
) -> float:
    score = float(np.clip(score, 0.0, 1.0))
    mean = float(np.clip(round_mean_score, 0.0, 1.0))
    utility = 1.0 + (2.0 * max(float(utility_strength), 0.0) * (score - mean))
    utility = float(np.clip(utility, max(float(min_utility), EPS), max(float(max_utility), EPS)))
    if score < utility_threshold and score < mean:
        return 0.0
    return utility


def select_utility_records(
    selection_records: list[dict[str, Any]],
    *,
    server_round: int,
    min_selected_clients: int,
    max_selected_fraction: float,
    warmup_rounds: int,
) -> list[dict[str, Any]]:
    if not selection_records:
        return []

    warmup = logical_round(server_round) <= warmup_rounds
    sorted_records = sorted(selection_records, key=lambda item: item["utility"], reverse=True)

    if warmup:
        selected = sorted_records
    else:
        selected = [record for record in sorted_records if record["utility"] > 0.0]
        minimum = min(max(min_selected_clients, 1), len(sorted_records))
        if len(selected) < minimum:
            selected = sorted_records[:minimum]

        maximum = max(
            minimum,
            int(np.ceil(len(sorted_records) * np.clip(max_selected_fraction, 0.0, 1.0))),
        )
        selected = selected[:maximum]

    selected_ids = {record["cid"] for record in selected}
    for record in selection_records:
        record["selected"] = record["cid"] in selected_ids
    return selected


def logical_round(server_round: int) -> int:
    return int((server_round + 1) // 2)
