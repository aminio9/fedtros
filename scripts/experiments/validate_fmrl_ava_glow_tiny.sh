#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

invoke_hydra_run() {
  poetry run python run.py "$@"
}

echo "Running pytest..."
poetry run python -m pytest tests -q

echo "Running FMRL-AVA-GLOW smoke..."
invoke_hydra_run experiment=smoke +method=fmrl_ava_glow runtime=tiny output=tiny

echo "Running FMRL-AVA-GLOW alpha=0.1 tiny validation..."
invoke_hydra_run experiment=exp3 +method=fmrl_ava_glow seed=42 dataset.preprocessing.alpha=0.1 runtime=tiny output=tiny

echo "Running fair FedAvg contextual-bandit comparison..."
invoke_hydra_run experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1 runtime=tiny output=tiny \
  training.rl_mode=contextual_bandit \
  training.gamma=0.0 \
  training.epsilon_start=0.30 \
  training.epsilon_end=0.02 \
  training.epsilon_decay_rate=0.97 \
  training.loss_weights.prior_kl=0.5 \
  training.loss_weights.q_td=0.25 \
  training.loss_weights.bandit_q=1.0 \
  training.loss_weights.classification=2.0 \
  training.imbalance.enabled=true \
  training.imbalance.weight_mode=effective_number \
  training.imbalance.effective_number_beta=0.999 \
  training.imbalance.min_weight=0.3 \
  training.imbalance.max_weight=3.0 \
  training.imbalance.class_balanced_sampling=true \
  training.imbalance.weighted_reward=true \
  training.classification_loss.name=focal \
  training.classification_loss.focal_gamma=1.5 \
  training.classification_loss.use_class_weights=true \
  training.auxiliary_losses.supervised_contrastive_lambda=0.02 \
  training.auxiliary_losses.center_loss_lambda=0.01 \
  training.kl.free_nats=0.25 \
  training.kl.warmup_steps=200

echo "Validation complete."
