import json
import logging
from typing import Any

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

    total_steps = 0
    total_reward = 0.0
    total_train_steps = 0
    total_td_loss = 0.0
    total_kl_loss = 0.0
    total_avg_q = 0.0
    total_correct = 0
    action_counts: dict[int, int] = {}
    label_counts: dict[int, int] = {}
    started_training = False

    episode_pbar = tqdm(
        range(local_episodes),
        desc="Local Round Progress",
        position=0,
        leave=True,
    )
    episode_pbar.write(f"Starting local training for {local_episodes} episodes.")

    for ep_idx in episode_pbar:
        episode_reward = 0.0
        episode_td_loss = 0.0
        episode_kl_loss = 0.0
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
            disable=(local_episodes == 1),
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
            total_steps += 1
            total_correct += int(action_int == true_label_int)
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            label_counts[true_label_int] = label_counts.get(true_label_int, 0) + 1

            # Start training once buffer has enough samples
            if len(buffer) >= min_buffer_size:
                if not started_training:
                    # logger.info(
                    #     "Replay buffer warm-up complete (size=%s >= min_buffer_size=%s). Starting updates.",
                    #     len(buffer),
                    #     min_buffer_size,
                    # )
                    started_training = True
                batch = buffer.sample(batch_size, device)
                td_loss, kl_loss, avg_q = agent.train_step(batch)

                episode_train_steps += 1
                episode_td_loss += td_loss
                episode_kl_loss += kl_loss
                episode_q_value += avg_q

                total_train_steps += 1
                total_td_loss += td_loss
                total_kl_loss += kl_loss
                total_avg_q += avg_q

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

        episode_pbar.write(
            f"  Episode {ep_idx + 1:02d} | Reward: {episode_reward:7.2f} | "
            f"Avg TD Loss: {avg_ep_td:6.4f} | Avg KL Loss: {avg_ep_kl:6.4f} | "
            f"Epsilon: {epsilon_scheduler.get_epsilon():.4f}"
        )

    episode_pbar.close()

    if total_train_steps == 0:
        logger.warning(
            "Round finished with no parameter updates (buffer size %s < min_buffer_size %s). "
            "Increase steps_per_episode/local_episodes or lower min_buffer_size to start training earlier.",
            len(buffer),
            min_buffer_size,
        )

    avg_round_td = total_td_loss / total_train_steps if total_train_steps else 0.0
    avg_round_kl = total_kl_loss / total_train_steps if total_train_steps else 0.0
    avg_round_q = total_avg_q / total_train_steps if total_train_steps else 0.0

    metrics = {
        "total_reward": total_reward,
        "avg_reward_per_episode": (total_reward / local_episodes if local_episodes else 0.0),
        "reward_per_step": total_reward / total_steps if total_steps else 0.0,
        "policy_accuracy": total_correct / total_steps if total_steps else 0.0,
        "avg_td_loss": avg_round_td,
        "avg_kl_loss": avg_round_kl,
        "avg_q_value": avg_round_q,
        "train_steps": float(total_train_steps),
        "buffer_size": float(len(buffer)),
        "epsilon": float(epsilon_scheduler.get_epsilon()),
        "action_histogram": json.dumps(action_counts, sort_keys=True),
        "label_histogram": json.dumps(label_counts, sort_keys=True),
    }

    logger.info("Local training finished. Ran %s steps.", total_steps)
    logger.info("Round Metrics: %s", metrics)

    return total_steps, metrics
