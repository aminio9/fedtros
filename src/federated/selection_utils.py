from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-8


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
