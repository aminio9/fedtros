<<<<<<< HEAD
import torch
=======
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
>>>>>>> ea28efe (Initial commit with updated source code)

from src.federated.selection_utils import (
    alignment_multiplier,
    centered_utility,
    combine_utility_score,
    critic_utility_score,
    select_utility_records,
    validation_team_reward,
)
<<<<<<< HEAD
from src.federated.server_models import AsyncCritic, CentralizedAggregator


=======
from src.federated.server import FMRLAdaptiveVectorAlignedAggregationStrategy
from src.federated.server_models import AsyncCritic, CentralizedAggregator


def _config_dir() -> str:
    return str((Path(__file__).resolve().parents[1] / "src" / "configs").resolve())


def _bare_fmrl_strategy():
    strategy = object.__new__(FMRLAdaptiveVectorAlignedAggregationStrategy)
    strategy.cfg = OmegaConf.create({"model": {"num_actions": 3}})
    strategy.profile_balance_strength = 1.0
    strategy.profile_quality_blend = 0.35
    strategy.profile_cluster_strength = 0.50
    strategy.profile_min_multiplier = 0.30
    strategy.profile_max_multiplier = 3.0
    strategy.profile_label_smoothing = 1.0
    strategy.drift_penalty_strength = 0.35
    strategy.drift_min_multiplier = 0.50
    strategy.server_optimizer_name = "adam"
    strategy.server_beta1 = 0.90
    strategy.server_beta2 = 0.99
    strategy.server_tau = 1e-3
    strategy.aggregation_lr = 0.02
    strategy.server_momentum = None
    strategy.server_second_moment = None
    return strategy


>>>>>>> ea28efe (Initial commit with updated source code)
def test_async_critic_outputs_nonnegative_utility():
    critic = AsyncCritic(hidden_dim=8, latent_dim=3, scalar_dim=4)
    utility = critic(torch.randn(2, 3), torch.rand(2, 4))

    assert utility.shape == (2, 1)
    assert torch.all(utility >= 0)


def test_centralized_aggregator_outputs_system_utility():
    mixer = CentralizedAggregator(num_agents=3, state_dim=15, hidden_dim=8)
    q_total = mixer(torch.rand(1, 3), torch.rand(1, 15))

    assert q_total.shape == (1, 1)


def test_fmrl_ava_utility_is_fedavg_neutral_for_iid_like_scores():
    utilities = [
        centered_utility(
            score=0.55,
            round_mean_score=0.55,
            utility_strength=0.75,
            min_utility=0.25,
            max_utility=2.0,
            utility_threshold=0.1,
        )
        for _ in range(3)
    ]

    assert utilities == [1.0, 1.0, 1.0]


def test_fmrl_ava_downweights_or_drops_relative_low_quality_clients():
    low = centered_utility(
        score=0.05,
        round_mean_score=0.50,
        utility_strength=0.75,
        min_utility=0.25,
        max_utility=2.0,
        utility_threshold=0.1,
    )
    high = centered_utility(
        score=0.80,
        round_mean_score=0.50,
        utility_strength=0.75,
        min_utility=0.25,
        max_utility=2.0,
        utility_threshold=0.1,
    )

    assert low == 0.0
    assert high > 1.0


def test_fmrl_ava_combines_audit_score_with_bounded_critic_residual():
    critic_score = critic_utility_score(1.0, utility_temperature=1.0)
    combined = combine_utility_score(
        audit_score=0.8,
        critic_score=critic_score,
        critic_blend=0.15,
    )

    assert 0.0 <= critic_score <= 1.0
    assert 0.5 < combined < 0.8


def test_fmrl_ava_selects_only_nonzero_utility_clients():
    selection_records = [
        {"cid": "1", "utility": 0.25, "selected": False},
        {"cid": "2", "utility": 0.00, "selected": False},
        {"cid": "3", "utility": 0.10, "selected": False},
        {"cid": "4", "utility": 0.00, "selected": False},
    ]

    selected = select_utility_records(
        selection_records,
        server_round=2,
        min_selected_clients=1,
        max_selected_fraction=1.0,
        warmup_rounds=0,
    )

    assert [record["cid"] for record in selected] == ["1", "3"]
    assert [record["selected"] for record in selection_records] == [
        True,
        False,
        True,
        False,
    ]


def test_alignment_multiplier_rewards_update_agreement():
    aligned = alignment_multiplier(
        1.0,
        alignment_strength=0.5,
        min_multiplier=0.5,
        max_multiplier=2.0,
    )
    neutral = alignment_multiplier(
        0.0,
        alignment_strength=0.5,
        min_multiplier=0.5,
        max_multiplier=2.0,
    )
    opposed = alignment_multiplier(
        -1.0,
        alignment_strength=0.5,
        min_multiplier=0.5,
        max_multiplier=2.0,
    )

    assert opposed < neutral < aligned


def test_validation_team_reward_is_open_set_aware():
    strong_open_set = validation_team_reward(
        {
            "f1_macro": 0.90,
            "balanced_accuracy": 0.85,
            "openset_auroc": 0.92,
            "openset_unknown_f1": 0.80,
            "openset_unknown_recall": 0.75,
            "openset_fpr95": 0.10,
        }
    )
    weak_open_set = validation_team_reward(
        {
            "f1_macro": 0.90,
            "balanced_accuracy": 0.85,
            "openset_auroc": 0.20,
            "openset_unknown_f1": 0.10,
            "openset_unknown_recall": 0.05,
            "openset_fpr95": 0.95,
        }
    )

    assert 0.0 <= weak_open_set < strong_open_set <= 1.0


def test_validation_team_reward_does_not_penalize_missing_open_set_metrics():
    reward = validation_team_reward(
        {
            "f1_macro": 0.90,
            "balanced_accuracy": 0.80,
        }
    )

    assert abs(reward - 0.86) < 1e-9
<<<<<<< HEAD
=======


def test_fmrl_ava_profile_multipliers_break_fedavg_tie_under_label_skew():
    strategy = _bare_fmrl_strategy()
    records = [
        {
            "cid": "majority",
            "num_examples": 1000.0,
            "label_histogram": '{"0": 1000, "1": 0, "2": 0}',
            "quality": 0.60,
        },
        {
            "cid": "minority",
            "num_examples": 100.0,
            "label_histogram": '{"0": 0, "1": 90, "2": 10}',
            "quality": 0.90,
        },
    ]

    strategy._apply_profile_multipliers(records)

    by_client = {record["cid"]: record for record in records}
    assert by_client["minority"]["profile_multiplier"] > by_client["majority"]["profile_multiplier"]
    assert by_client["minority"]["class_multiplier"] > by_client["majority"]["class_multiplier"]


def test_fmrl_ava_drift_penalty_downweights_large_outlier_update():
    strategy = _bare_fmrl_strategy()
    records = [
        {"cid": "stable_a", "delta_norm": 1.0},
        {"cid": "stable_b", "delta_norm": 1.1},
        {"cid": "outlier", "delta_norm": 5.0},
    ]

    strategy._apply_drift_multipliers(records)

    by_client = {record["cid"]: record for record in records}
    assert by_client["outlier"]["drift_multiplier"] < by_client["stable_a"]["drift_multiplier"]
    assert by_client["outlier"]["drift_multiplier"] >= strategy.drift_min_multiplier


def test_fmrl_ava_server_adam_step_is_not_plain_fedavg_delta():
    strategy = _bare_fmrl_strategy()
    normalized_delta = [np.array([0.1, -0.2], dtype=np.float32)]

    optimized = strategy._server_optimized_delta(normalized_delta)
    plain = [strategy.aggregation_lr * delta for delta in normalized_delta]

    assert not np.allclose(optimized[0], plain[0])
    assert np.sign(optimized[0]).tolist() == np.sign(normalized_delta[0]).tolist()


def test_fmrl_ava_method_overlay_uses_glow_safe_defaults():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["experiment=exp3", "+method=fmrl_ava"])

    assert cfg.federated.strategy.name == "fmrl_ava"
    assert cfg.federated.strategy.profile_balance_strength == 0.0
    assert cfg.federated.strategy.profile_min_multiplier == 1.0
    assert cfg.federated.strategy.drift_penalty_strength > 0.0
    assert cfg.federated.strategy.server_optimizer == "none"
>>>>>>> ea28efe (Initial commit with updated source code)
