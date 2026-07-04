import json
import logging
import sys
from typing import Any

import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.agents.agent import Agent
from src.agents.policy import EpsilonGreedyPolicy, EpsilonScheduler
from src.rl.environment import BlockchainIntrusionEnv
from src.rl.class_balance import effective_number_class_weights
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
    global_prototypes: torch.Tensor | None = None,
    global_prototype_mask: torch.Tensor | None = None,
    prototype_lambda: float = 0.0,
    prototype_feature: str = "latent_q",
    dkd_enabled: bool = False,
    dkd_round: int = 0,
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
    aux_ce_weight = float(getattr(cfg_training, "aux_ce_weight", 0.0))
    aux_ce_label_smoothing = float(getattr(cfg_training, "aux_ce_label_smoothing", 0.0))
    aux_ce_use_class_weights = bool(getattr(cfg_training, "aux_ce_use_class_weights", True))
    reward_class_weights = None
    if aux_ce_use_class_weights and hasattr(env, "get_reward_class_weights"):
        reward_class_weights = env.get_reward_class_weights(device)
    dkd_class_weights = None
    dkd_present_classes = None
    if dkd_enabled:
        dkd_class_weights = effective_number_class_weights(
            env.all_labels_a_t.detach().cpu(),
            int(getattr(env, "num_actions_nt", 1)),
            beta=float(getattr(cfg_training, "dkd_class_balance_beta", 0.999)),
            min_weight=float(getattr(cfg_training, "dkd_class_weight_min", 0.2)),
            max_weight=float(getattr(cfg_training, "dkd_class_weight_max", 5.0)),
            normalize=True,
            device=device,
        )
        counts = torch.bincount(
            env.all_labels_a_t.detach().cpu().long().clamp_min(0),
            minlength=int(getattr(env, "num_actions_nt", 1)),
        )[: int(getattr(env, "num_actions_nt", 1))]
        dkd_present_classes = (counts > 0).to(device)

    # Optional deterministic seeding if cfg_training.seed exists
    base_seed = getattr(cfg_training, "seed", None)

    active_logger = logger or logging.getLogger("LocalTraining")

    total_steps = 0
    total_reward = 0.0
    total_train_steps = 0
    total_td_loss = 0.0
    total_kl_loss = 0.0
    total_prox_loss = 0.0
    total_proto_loss = 0.0
    total_aux_ce_loss = 0.0
    total_dkd_task_loss = 0.0
    total_dkd_kd_loss = 0.0
    total_dkd_align_loss = 0.0
    total_dkd_agreement = 0.0
    total_dkd_confidence = 0.0
    total_dkd_align_score = 0.0
    total_dkd_teacher_task_loss = 0.0
    total_dkd_student_task_loss = 0.0
    total_dkd_t2s_loss = 0.0
    total_dkd_s2t_loss = 0.0
    total_dkd_teacher_batch_accuracy = 0.0
    total_dkd_student_batch_accuracy = 0.0
    total_dkd_correct_agreement = 0.0
    total_dkd_teacher_confidence = 0.0
    total_dkd_student_confidence = 0.0
    total_dkd_t2s_enabled = 0.0
    total_dkd_s2t_enabled = 0.0
    total_avg_q = 0.0
    total_correct = 0
    class_correct_counts: dict[int, int] = {}
    reward_weight_sum = 0.0
    action_counts: dict[int, int] = {}
    label_counts: dict[int, int] = {}
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
        episode_proto_loss = 0.0
        episode_aux_ce_loss = 0.0
        episode_dkd_task_loss = 0.0
        episode_dkd_kd_loss = 0.0
        episode_dkd_align_loss = 0.0
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

            correct = int(action_int == true_label_int)
            episode_reward += reward
            total_steps += 1
            total_correct += correct
            if correct:
                class_correct_counts[true_label_int] = class_correct_counts.get(true_label_int, 0) + 1
            reward_weight_sum += float(info.get("reward_weight", 1.0))
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
                td_loss, kl_loss, prox_loss, avg_q = agent.train_step(
                    batch,
                    proximal_mu=proximal_mu,
                    global_prototypes=global_prototypes,
                    global_prototype_mask=global_prototype_mask,
                    prototype_lambda=prototype_lambda,
                    prototype_feature=prototype_feature,
                    aux_ce_weight=aux_ce_weight,
                    aux_ce_label_smoothing=aux_ce_label_smoothing,
                    class_weights=reward_class_weights,
                    dkd_enabled=dkd_enabled,
                    dkd_round=dkd_round,
                    dkd_class_weights=dkd_class_weights,
                    dkd_present_classes=dkd_present_classes,
                )
                proto_loss = float(getattr(agent, "last_prototype_loss", 0.0))
                aux_ce_loss = float(getattr(agent, "last_aux_ce_loss", 0.0))
                dkd_task_loss = float(getattr(agent, "last_dkd_task_loss", 0.0))
                dkd_kd_loss = float(getattr(agent, "last_dkd_kd_loss", 0.0))
                dkd_align_loss = float(getattr(agent, "last_dkd_align_loss", 0.0))
                dkd_agreement = float(getattr(agent, "last_dkd_agreement", 0.0))
                dkd_confidence = float(getattr(agent, "last_dkd_confidence", 0.0))
                dkd_align_score = float(getattr(agent, "last_dkd_align_score", 0.0))
                dkd_teacher_task_loss = float(getattr(agent, "last_dkd_teacher_task_loss", 0.0))
                dkd_student_task_loss = float(getattr(agent, "last_dkd_student_task_loss", 0.0))
                dkd_t2s_loss = float(getattr(agent, "last_dkd_t2s_loss", 0.0))
                dkd_s2t_loss = float(getattr(agent, "last_dkd_s2t_loss", 0.0))
                dkd_teacher_batch_accuracy = float(getattr(agent, "last_dkd_teacher_batch_accuracy", 0.0))
                dkd_student_batch_accuracy = float(getattr(agent, "last_dkd_student_batch_accuracy", 0.0))
                dkd_correct_agreement = float(getattr(agent, "last_dkd_correct_agreement", 0.0))
                dkd_teacher_confidence = float(getattr(agent, "last_dkd_teacher_confidence", 0.0))
                dkd_student_confidence = float(getattr(agent, "last_dkd_student_confidence", 0.0))
                dkd_t2s_enabled = float(getattr(agent, "last_dkd_t2s_enabled", 0.0))
                dkd_s2t_enabled = float(getattr(agent, "last_dkd_s2t_enabled", 0.0))

                episode_train_steps += 1
                episode_td_loss += td_loss
                episode_kl_loss += kl_loss
                episode_prox_loss += prox_loss
                episode_proto_loss += proto_loss
                episode_aux_ce_loss += aux_ce_loss
                episode_dkd_task_loss += dkd_task_loss
                episode_dkd_kd_loss += dkd_kd_loss
                episode_dkd_align_loss += dkd_align_loss
                episode_q_value += avg_q

                total_train_steps += 1
                total_td_loss += td_loss
                total_kl_loss += kl_loss
                total_prox_loss += prox_loss
                total_proto_loss += proto_loss
                total_aux_ce_loss += aux_ce_loss
                total_dkd_task_loss += dkd_task_loss
                total_dkd_kd_loss += dkd_kd_loss
                total_dkd_align_loss += dkd_align_loss
                total_dkd_agreement += dkd_agreement
                total_dkd_confidence += dkd_confidence
                total_dkd_align_score += dkd_align_score
                total_dkd_teacher_task_loss += dkd_teacher_task_loss
                total_dkd_student_task_loss += dkd_student_task_loss
                total_dkd_t2s_loss += dkd_t2s_loss
                total_dkd_s2t_loss += dkd_s2t_loss
                total_dkd_teacher_batch_accuracy += dkd_teacher_batch_accuracy
                total_dkd_student_batch_accuracy += dkd_student_batch_accuracy
                total_dkd_correct_agreement += dkd_correct_agreement
                total_dkd_teacher_confidence += dkd_teacher_confidence
                total_dkd_student_confidence += dkd_student_confidence
                total_dkd_t2s_enabled += dkd_t2s_enabled
                total_dkd_s2t_enabled += dkd_s2t_enabled
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
        avg_ep_prox = episode_prox_loss / episode_train_steps if episode_train_steps else 0.0
        avg_ep_proto = episode_proto_loss / episode_train_steps if episode_train_steps else 0.0
        avg_ep_aux_ce = episode_aux_ce_loss / episode_train_steps if episode_train_steps else 0.0
        avg_ep_dkd_kd = episode_dkd_kd_loss / episode_train_steps if episode_train_steps else 0.0
        avg_ep_dkd_align = episode_dkd_align_loss / episode_train_steps if episode_train_steps else 0.0

        active_logger.info(
            "Episode %02d | Reward: %7.2f | Avg TD Loss: %6.4f | Avg KL Loss: %6.4f | "
            "Avg CE Loss: %6.4f | Avg DKD Loss: %6.4f | Avg Align Loss: %6.4f | "
            "Avg Prox Loss: %6.4f | Avg Proto Loss: %6.4f | Epsilon: %.4f",
            ep_idx + 1,
            episode_reward,
            avg_ep_td,
            avg_ep_kl,
            avg_ep_aux_ce,
            avg_ep_dkd_kd,
            avg_ep_dkd_align,
            avg_ep_prox,
            avg_ep_proto,
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
    avg_round_proto = total_proto_loss / total_train_steps if total_train_steps else 0.0
    avg_round_aux_ce = total_aux_ce_loss / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_task = total_dkd_task_loss / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_kd = total_dkd_kd_loss / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_align = total_dkd_align_loss / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_agreement = total_dkd_agreement / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_confidence = total_dkd_confidence / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_align_score = total_dkd_align_score / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_teacher_task = total_dkd_teacher_task_loss / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_student_task = total_dkd_student_task_loss / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_t2s = total_dkd_t2s_loss / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_s2t = total_dkd_s2t_loss / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_teacher_acc = total_dkd_teacher_batch_accuracy / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_student_acc = total_dkd_student_batch_accuracy / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_correct_agreement = total_dkd_correct_agreement / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_teacher_conf = total_dkd_teacher_confidence / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_student_conf = total_dkd_student_confidence / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_t2s_enabled = total_dkd_t2s_enabled / total_train_steps if total_train_steps else 0.0
    avg_round_dkd_s2t_enabled = total_dkd_s2t_enabled / total_train_steps if total_train_steps else 0.0
    avg_round_q = total_avg_q / total_train_steps if total_train_steps else 0.0
    per_class_accuracy = {
        str(class_id): class_correct_counts.get(class_id, 0) / count
        for class_id, count in sorted(label_counts.items())
        if count > 0
    }
    balanced_policy_accuracy = (
        sum(per_class_accuracy.values()) / len(per_class_accuracy) if per_class_accuracy else 0.0
    )

    metrics = {
        "total_reward": total_reward,
        "avg_reward_per_episode": (total_reward / local_episodes if local_episodes else 0.0),
        "reward_per_step": total_reward / total_steps if total_steps else 0.0,
        "policy_accuracy": total_correct / total_steps if total_steps else 0.0,
        "balanced_policy_accuracy": balanced_policy_accuracy,
        "avg_td_loss": avg_round_td,
        "avg_kl_loss": avg_round_kl,
        "avg_aux_ce_loss": avg_round_aux_ce,
        "aux_ce_weight": aux_ce_weight,
        "dkd_enabled": float(bool(dkd_enabled)),
        "avg_dkd_task_loss": avg_round_dkd_task,
        "avg_dkd_kd_loss": avg_round_dkd_kd,
        "avg_dkd_align_loss": avg_round_dkd_align,
        "avg_dkd_teacher_task_loss": avg_round_dkd_teacher_task,
        "avg_dkd_student_task_loss": avg_round_dkd_student_task,
        "avg_dkd_t2s_loss": avg_round_dkd_t2s,
        "avg_dkd_s2t_loss": avg_round_dkd_s2t,
        "dkd_teacher_batch_accuracy": avg_round_dkd_teacher_acc,
        "dkd_student_batch_accuracy": avg_round_dkd_student_acc,
        "dkd_correct_agreement": avg_round_dkd_correct_agreement,
        "dkd_teacher_confidence": avg_round_dkd_teacher_conf,
        "dkd_student_confidence": avg_round_dkd_student_conf,
        "dkd_t2s_enabled_rate": avg_round_dkd_t2s_enabled,
        "dkd_s2t_enabled_rate": avg_round_dkd_s2t_enabled,
        "dkd_lambda_kd": float(getattr(agent, "dkd_lambda_kd", 0.0)),
        "dkd_lambda_align": float(getattr(agent, "dkd_lambda_align", 0.0)),
        "dkd_temperature": float(getattr(agent, "last_dkd_temperature", 1.0)),
        "dkd_agreement": avg_round_dkd_agreement,
        "dkd_confidence": avg_round_dkd_confidence,
        "dkd_align_score": avg_round_dkd_align_score,
        "aux_ce_weight": aux_ce_weight,
        "avg_prox_loss": avg_round_prox,
        "avg_proto_loss": avg_round_proto,
        "prototype_lambda": float(prototype_lambda),
        "avg_q_value": avg_round_q,
        "train_steps": float(total_train_steps),
        "total_steps": float(total_steps),
        "buffer_size": float(len(buffer)),
        "epsilon": float(epsilon_scheduler.get_epsilon()),
        "proximal_mu": float(proximal_mu),
        "action_histogram": json.dumps(action_counts, sort_keys=True),
        "label_histogram": json.dumps(label_counts, sort_keys=True),
        "per_class_policy_accuracy": json.dumps(per_class_accuracy, sort_keys=True),
        "mean_reward_weight": reward_weight_sum / total_steps if total_steps else 1.0,
    }

    active_logger.info("Local training finished. Ran %s steps.", total_steps)
    active_logger.info("Round Metrics: %s", metrics)

    return total_steps, metrics
