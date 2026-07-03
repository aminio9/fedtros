import json
import logging
import sys
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.agents.agent import Agent
from src.agents.policy import EpsilonGreedyPolicy, EpsilonScheduler
from src.rl.environment import BlockchainIntrusionEnv
from src.rl.replay_buffer import ExperienceReplayBuffer

logger = logging.getLogger("LocalTraining")


def run_local_training_round(
    agent: Agent,
    env: BlockchainIntrusionEnv,
    buffer: ExperienceReplayBuffer,
    policy: EpsilonGreedyPolicy,
    epsilon_scheduler: EpsilonScheduler,
    cfg_training: DictConfig,
    device: torch.device,
    proximal_mu: float = 0.0,
    logger: logging.Logger | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Run one FL round worth of local experience collection/training.

    Returns:
        total_steps: number of environment steps executed
        metrics: dict with reward/loss/Q statistics for this round
    """

    local_episodes = cfg_training.local_episodes_per_round
    steps_per_episode = cfg_training.steps_per_episode
    min_buffer_size = cfg_training.min_buffer_size
    batch_size = cfg_training.batch_size
    target_update_freq = cfg_training.target_update_freq
    tau = cfg_training.tau

    # Optional deterministic seeding if cfg_training.seed exists
    base_seed = getattr(cfg_training, "seed", None)

    active_logger = logger or logging.getLogger("LocalTraining")

    total_steps = 0
    total_reward = 0.0
    total_train_steps = 0
    total_td_loss = 0.0
    total_kl_loss = 0.0
    total_prox_loss = 0.0
    total_avg_q = 0.0
    total_correct = 0
    action_counts: dict[int, int] = {}
    label_counts: dict[int, int] = {}
    reward_values: list[float] = []
    reward_by_label: dict[int, list[float]] = {}
    scalar_metric_totals: dict[str, float] = {}
    started_training = False

    interactive = sys.stdout.isatty()
    episode_pbar = tqdm(
        range(local_episodes),
        desc="Local Round Progress",
        position=0,
        leave=True,
        disable=not interactive,
    )
    active_logger.info("Starting local training for %s episodes.", local_episodes)

    for ep_idx in episode_pbar:
        episode_reward = 0.0
        episode_td_loss = 0.0
        episode_kl_loss = 0.0
        episode_prox_loss = 0.0
        episode_q_value = 0.0
        episode_train_steps = 0

        if base_seed is None:
            state_s, _ = env.reset()
        else:
            # Different, reproducible seed per episode
            state_s, _ = env.reset(seed=int(base_seed) + int(ep_idx))

        step_pbar = tqdm(
            range(steps_per_episode),
            desc=f"  Episode {ep_idx + 1:02d}/{local_episodes}",
            position=1,
            leave=False,
            disable=not interactive or local_episodes == 1,
        )

        for _ in step_pbar:
            epsilon = epsilon_scheduler.get_epsilon()
            action = policy.select_action(state_s, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)

            true_label = info.get("true_label")
            done = float(terminated or truncated)
            true_label_int = int(true_label)
            action_int = int(action)

            buffer.push(state_s, action_int, reward, next_state, done, true_label_int)
            state_s = next_state

            episode_reward += reward
            reward_values.append(float(reward))
            reward_by_label.setdefault(true_label_int, []).append(float(reward))
            total_steps += 1
            total_correct += int(action_int == true_label_int)
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            label_counts[true_label_int] = label_counts.get(true_label_int, 0) + 1

            # Start training once buffer has enough samples
            if len(buffer) >= min_buffer_size:
                if not started_training:
                    # active_logger.info(
                    #     "Replay buffer warm-up complete (size=%s >= min_buffer_size=%s). Starting updates.",
                    #     len(buffer),
                    #     min_buffer_size,
                    # )
                    started_training = True
                batch = buffer.sample(batch_size, device)
                step_metrics = agent.train_step(batch, proximal_mu=proximal_mu)
                td_loss = float(step_metrics.get("loss/q_td", 0.0))
                kl_loss = float(step_metrics.get("loss/prior_kl", 0.0))
                prox_loss = float(step_metrics.get("loss/proximal", 0.0))
                avg_q = float(step_metrics.get("q/pred_mean", 0.0))

                episode_train_steps += 1
                episode_td_loss += td_loss
                episode_kl_loss += kl_loss
                episode_prox_loss += prox_loss
                episode_q_value += avg_q

                total_train_steps += 1
                total_td_loss += td_loss
                total_kl_loss += kl_loss
                total_prox_loss += prox_loss
                total_avg_q += avg_q
                for key, value in step_metrics.items():
                    if isinstance(value, (int, float)):
                        scalar_metric_totals[key] = scalar_metric_totals.get(key, 0.0) + float(
                            value
                        )

                # Soft-update the target network periodically
                if total_steps % target_update_freq == 0:
                    agent.update_target_network(tau)

            if terminated:
                break

        step_pbar.close()
        epsilon_scheduler.step()
        total_reward += episode_reward

        avg_ep_td = episode_td_loss / episode_train_steps if episode_train_steps else 0.0
        avg_ep_kl = episode_kl_loss / episode_train_steps if episode_train_steps else 0.0
        avg_ep_prox = episode_prox_loss / episode_train_steps if episode_train_steps else 0.0

        active_logger.info(
            "Episode %02d | Reward: %7.2f | Avg TD Loss: %6.4f | Avg KL Loss: %6.4f | "
            "Avg Prox Loss: %6.4f | Epsilon: %.4f",
            ep_idx + 1,
            episode_reward,
            avg_ep_td,
            avg_ep_kl,
            avg_ep_prox,
            epsilon_scheduler.get_epsilon(),
        )

    episode_pbar.close()

    if total_train_steps == 0:
        active_logger.warning(
            "Round finished with no parameter updates (buffer size %s < min_buffer_size %s). "
            "Increase steps_per_episode/local_episodes or lower min_buffer_size to start training earlier.",
            len(buffer),
            min_buffer_size,
        )

    avg_round_td = total_td_loss / total_train_steps if total_train_steps else 0.0
    avg_round_kl = total_kl_loss / total_train_steps if total_train_steps else 0.0
    avg_round_prox = total_prox_loss / total_train_steps if total_train_steps else 0.0
    avg_round_q = total_avg_q / total_train_steps if total_train_steps else 0.0
    averaged_step_metrics = {
        key: value / total_train_steps
        for key, value in scalar_metric_totals.items()
        if total_train_steps
    }
    reward_array = np.asarray(reward_values, dtype=float)
    reward_stats = {
        "reward_mean": float(reward_array.mean()) if reward_array.size else 0.0,
        "reward_std": float(reward_array.std()) if reward_array.size else 0.0,
        "reward_min": float(reward_array.min()) if reward_array.size else 0.0,
        "reward_max": float(reward_array.max()) if reward_array.size else 0.0,
    }
    per_class_reward = {
        str(label): float(np.asarray(values, dtype=float).mean())
        for label, values in sorted(reward_by_label.items())
    }
    class_reward_weights = getattr(env, "_class_reward_weights", None)
    if torch.is_tensor(class_reward_weights):
        class_reward_weights_json = json.dumps(
            [float(value) for value in class_reward_weights.detach().cpu().tolist()]
        )
    else:
        class_reward_weights_json = "[]"

    metrics = {
        "total_reward": total_reward,
        "avg_reward_per_episode": (total_reward / local_episodes if local_episodes else 0.0),
        "reward_per_step": total_reward / total_steps if total_steps else 0.0,
        "policy_accuracy": total_correct / total_steps if total_steps else 0.0,
        "avg_td_loss": avg_round_td,
        "avg_kl_loss": avg_round_kl,
        "avg_prox_loss": avg_round_prox,
        "avg_total_loss": float(averaged_step_metrics.get("loss/total", 0.0)),
        "avg_classification_loss": float(
            averaged_step_metrics.get("loss/classification", 0.0)
        ),
        "avg_supervised_contrastive_loss": float(
            averaged_step_metrics.get("loss/supervised_contrastive", 0.0)
        ),
        "avg_center_compactness_loss": float(
            averaged_step_metrics.get("loss/center_compactness", 0.0)
        ),
        "avg_q_value": avg_round_q,
        "q_value_mean": float(averaged_step_metrics.get("q/value_mean", 0.0)),
        "q_value_std": float(averaged_step_metrics.get("q/value_std", 0.0)),
        "q_value_min": float(averaged_step_metrics.get("q/value_min", 0.0)),
        "q_value_max": float(averaged_step_metrics.get("q/value_max", 0.0)),
        "kl_mean": float(averaged_step_metrics.get("kl/mean", avg_round_kl)),
        "kl_std": float(averaged_step_metrics.get("kl/std", 0.0)),
        "gradient_norm_prior": float(averaged_step_metrics.get("gradient/prior_norm", 0.0)),
        "gradient_norm_q": float(averaged_step_metrics.get("gradient/q_norm", 0.0)),
        "learning_rate_prior": float(averaged_step_metrics.get("lr/prior", 0.0)),
        "learning_rate_q_rl": float(averaged_step_metrics.get("lr/q_rl", 0.0)),
        "train_steps": float(total_train_steps),
        "total_steps": float(total_steps),
        "buffer_size": float(len(buffer)),
        "epsilon": float(epsilon_scheduler.get_epsilon()),
        "proximal_mu": float(proximal_mu),
        **reward_stats,
        "per_class_reward": json.dumps(per_class_reward, sort_keys=True),
        "class_reward_weights": class_reward_weights_json,
        "action_histogram": json.dumps(action_counts, sort_keys=True),
        "label_histogram": json.dumps(label_counts, sort_keys=True),
    }
    for key, value in averaged_step_metrics.items():
        metrics[f"train_step/{key}"] = float(value)

    active_logger.info("Local training finished. Ran %s steps.", total_steps)
    active_logger.info("Round Metrics: %s", metrics)

    return total_steps, metrics
