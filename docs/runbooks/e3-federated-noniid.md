# E3 Federated Non-IID Runbook

## Objective

Compare FedAvg, FedProx, FMRL-AVA, and the centralized pooled baseline under matched Dirichlet partitions.

## Hydra Config Used

`experiment=exp3` with `dataset.preprocessing.alpha=0.1` or `10.0`.

## Override Examples

```bash
python run.py experiment=exp3 +method=fmrl_ava seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fedprox seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=centralized_no_osr seed=42 dataset.preprocessing.alpha=0.1
```

## Execution Commands

```bash
python run.py experiment=exp3 +method=fmrl_ava_glow seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fmrl_ava seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fedprox seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=centralized_no_osr seed=42 dataset.preprocessing.alpha=0.1 tracking.run_id=e3_central_alpha0.1_seed42
bash scripts/experiments/e3_federated_noniid.sh
bash scripts/experiments/validate_fmrl_ava_glow_tiny.sh
```

## Expected Outputs

- `federated_history.csv`
- `federated_round_metrics.csv`
- `communication_metrics.csv`
- `evaluation_metrics.json`

## Checkpoints

- `global_model_round_*.pt`
- `global_model_latest.pt`
- `best_model.pt`
- `latest_checkpoint.pt`

## Logs

- `run.log`
- `debug.log`
- `fmrl_ava_monitoring.jsonl` for FMRL-AVA

## Metrics

- `test/accuracy`
- `test/macro_f1`
- `test/balanced_accuracy`
- `federated/rounds`
- `federated/flower_rounds`
- `fmrl_ava_validation_reward`
- `fmrl_ava_support_reward`
- `fmrl_ava_team_reward_target`
- `alignment_cosine` and `alignment_multiplier` inside `fmrl_ava_monitoring.jsonl`

## Artifacts

- `processed/`
- `plots/`
- `plot_manifest.json`

## Validation

- Confirm the same seed and alpha are reused across all methods.
- Confirm `federated.resume_from` is available for restartable rounds.
- Confirm the client count matches the preprocessing partition count.

## Troubleshooting

- If client data are missing, rerun preprocessing first.
- If FMRL-AVA selects no clients, check `strategy.utility_threshold` and
  `strategy.min_selected_clients`.
- If FMRL-AVA behaves like FedAvg under hard non-IID, inspect `fmrl_ava_monitoring.jsonl`
  for near-constant utilities, near-constant alignment multipliers, or
  `strategy.alignment_strength=0.0`.
- If communication metrics are empty, verify `federated_history.csv` exists.

## FMRL-AVA-GLOW patch: research-grade non-IID alpha=0.1 mode

FMRL-AVA-GLOW is the fixed FMRL-AVA configuration used for severe Dirichlet label skew. GLOW means: Gradient-safe contextual-bandit local RL, Local proximal and latent regularization, Outcome-rewarded server critic trained from validation advantage, and Warm FedAvg-anchored weakly aligned aggregation.

The local traffic environment is treated as a contextual-bandit classification problem. A client action predicts the label of the current traffic sample, but it does not cause the next traffic sample. For that reason `training.rl_mode` is `contextual_bandit` and `training.gamma` is set to `0.0`. The old TD path remains in the code with a small weight for backward compatibility, but the main local Q signal is now full-action bandit supervision: the true class receives `reward.correct`, locally present wrong classes receive `reward.incorrect`, and locally absent classes receive no Q-gradient.

Missing-class protection is active only during local client training. Each client reports its local label histogram; the agent stores `local_class_counts` and masks absent local logits in the supervised classification loss using `training.missing_class_gradient.mask_value`. The epsilon-greedy policy also restricts random and greedy choices to labels available on the client shard. Global evaluation and inference do not apply this mask.

Class-aware aggregation multipliers are disabled for FMRL-AVA-GLOW because prior project experiments showed they hurt this method. The class-aware code remains available for FedMADE-style experiments, but `profile_balance_strength`, `profile_quality_blend`, and `profile_cluster_strength` are set to zero and the profile multiplier bounds are fixed to `1.0` for GLOW.

Server selection is coverage-safe. Warmup rounds select all clients. After warmup, the server may drop only low-utility clients while keeping at least 90% of the sampled clients and preserving nonzero aggregate support for every class. For the standard 10-client alpha=0.1 setting this keeps 9-10 clients instead of turning selection into a class-collapse machine, because apparently non-IID data was not chaotic enough already.

Vector alignment is now FedAvg-anchored. The reference delta is computed from plain sample-count FedAvg before utility, drift, profile, or alignment modifiers. Alignment is a weak bounded stability signal only, with default bounds `[0.95, 1.05]` in GLOW.

The server critic/mixer is trained from outcome reward. `_current_utility_tensor()` now uses differentiable critic outputs, and `_train_server_models()` optimizes mixer MSE plus critic MSE targets adjusted by validation/support advantage. Critic influence on selection is delayed by `critic_activation_round`; before activation, selection uses the deterministic audit score. The critic is auxiliary and lightly blended after activation, not a magical aggregation oracle wearing a lab coat.

Server Adam/Yogi is disabled in the first fixed method. GLOW uses `server_optimizer: none` and `aggregation_lr: 1.0`, so the server applies the bounded weighted delta directly. Adaptive server optimizers remain in code for later ablations, but they are not active in the first stability patch.

Primary metrics are macro-F1, balanced accuracy, worst-class F1, and minority-class recall. Overall accuracy is secondary because the dataset is imbalanced and majority-class accuracy can look impressive while minority classes quietly burn.

### GLOW ablation plan

B0: FedAvg current baseline.
B1: FedAvg plus fixed contextual-bandit RL.
B2: FMRL-AVA current pre-GLOW behavior.
B3: FMRL-AVA plus contextual-bandit RL.
B4: add missing-class gradient mask.
B5: add local proximal regularization.
B6: add coverage-safe selection.
B7: add weak FedAvg-anchored vector alignment.
B8: add trained server critic after round 40.

### Validation commands

```bash
python -m pytest tests -q
python run.py experiment=smoke +method=fmrl_ava_glow runtime=tiny output=tiny
python run.py experiment=exp3 +method=fmrl_ava_glow seed=42 dataset.preprocessing.alpha=0.1 runtime=tiny output=tiny
```

Fair FedAvg comparison with the same local RL fixes:

```bash
python run.py experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1 \
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
  training.classification_loss.focal_gamma=1.5 \
  training.auxiliary_losses.supervised_contrastive_lambda=0.02 \
  training.auxiliary_losses.center_loss_lambda=0.01 \
  training.kl.free_nats=0.25 \
  training.kl.warmup_steps=200 \
  training.missing_class_gradient.enabled=true \
  training.missing_class_gradient.mask_value=-20.0
```

### Sources used for method justification

- Wong, H. Y., Lim, C. K., Chan, C. S. “Stratify: Rethinking Federated Learning for Non-IID Data through Balanced Sampling.” Pattern Recognition, 2026. DOI: 10.1016/j.patcog.2026.113900.
- Chowdhury, S., et al. “Confusion-Calibrated Cross-Entropy and Class-Specialized Aggregation for Robust Federated Learning under Extreme Data Heterogeneity.” Knowledge-Based Systems, 2026. DOI: 10.1016/j.knosys.2026.115497.
- Saha, P., Mishra, D., Wagner, F., Kamnitsas, K., Noble, J. A. “FedExIT - Missing Class-agnostic Semi-Supervised Federated Learning with Extreme Imbalance Tackling Scheme.” Information Fusion, 2026. DOI: 10.1016/j.inffus.2025.104080.
- Wang, X., Wu, Z., Zhu, J. “FedAPE: Heterogeneous Federated Learning with Attention-guided Aggregation and Prototype Enhancement.” Future Generation Computer Systems, 2026. DOI: 10.1016/j.future.2026.108417.
- Jing, Y., Guo, B., Li, N., Xu, R., Yu, Z. “Federated Multi-Agent Reinforcement Learning: A Comprehensive Survey of Methods, Applications and Challenges.” Expert Systems with Applications, 2025. DOI: 10.1016/j.eswa.2025.128729.
- Giuseppi, A., Menegatti, D., Pietrabissa, A. “Enhancing Federated Reinforcement Learning: A Consensus-based Approach for Both Homogeneous and Heterogeneous Agents.” Machine Intelligence Research, 2025. DOI: 10.1007/s11633-025-1550-8.
- Mnih, V., et al. “Human-level control through deep reinforcement learning.” Nature, 2015. DOI: 10.1038/nature14236.
- van Hasselt, H., Guez, A., Silver, D. “Deep Reinforcement Learning with Double Q-learning.” AAAI, 2016. DOI: 10.1609/aaai.v30i1.10295.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. “Focal Loss for Dense Object Detection.” ICCV, 2017. DOI: 10.1109/ICCV.2017.324.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., Belongie, S. “Class-Balanced Loss Based on Effective Number of Samples.” CVPR, 2019. DOI: 10.1109/CVPR.2019.00949.
- Hou, W., Chen, T., Wang, F., Wu, T., Zheng, Z., Tang, S., Lim, W. Y. B. “FedAdamom: Adaptive Momentum for Improved Generalization in Federated Optimization.” CVPR, 2026. Use the official CVPR OpenAccess URL if no DOI is available; do not invent one.

## 2026-07 FMRL-AVA-GLOW stability patch

The uploaded alpha=0.1 GLOW run was incomplete and must not be treated as a final test. It stopped before producing final `test_metrics.json`. The observed collapse starts during continued local training plus aggregation, not in AVA/open-set rejection, because open-set evaluation was disabled for that run.

The first fixed target is `fmrl_ava_glow_stable`: it executes the FMRL two-phase path but behaves like sample-count FedAvg. It selects all clients through a long warmup, disables utility/profile/drift/alignment multipliers, disables critic selection, disables server Adam, and uses `aggregation_lr: 1.0`. If this config does not come close to FedAvg, the bug is in the FMRL codepath rather than the adaptive algorithm.

`fmrl_ava_glow` now uses safer local training defaults: `gamma: 0.0`, lower KL pressure (`prior_kl: 0.10`, `free_nats: 1.0`, `warmup_steps: 1000`), stronger supervised classification (`classification: 3.0`), weaker TD/bandit contribution, focal gamma `1.0`, and `local_proximal_mu: 0.001`. The server critic is not allowed to affect selection by default: `critic_blend: 0.0`, `critic_activation_round: 999999`, and `critic_active_blend: 0.0`. The mixer output is bounded to `[0,1]`, so the auxiliary server loss cannot explode into fake utility values in the tens of thousands. Small mercy, finally.

Final evaluation now prefers `best_model.pt` when `evaluation.use_best_checkpoint: true`, falling back to the configured checkpoint only when no best checkpoint exists or the setting is disabled. This prevents a collapsed final round from silently overwriting a much stronger validation checkpoint.

Recommended ablation commands:

```bash
python run.py experiment=exp3 +method=fedavg seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fmrl_ava_glow_stable seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fmrl_ava_glow_rl seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fmrl_ava_glow_rl_prox seed=42 dataset.preprocessing.alpha=0.1
python run.py experiment=exp3 +method=fmrl_ava_glow_rl_prox_align seed=42 dataset.preprocessing.alpha=0.1
```

Primary selection/reporting metrics remain validation/test macro-F1, balanced accuracy, worst-class F1, and per-class recall. Accuracy is secondary because Normal-class dominance can make a bad minority-class model look annoyingly competent.

## FMRL-AVA-GLOW-TWA server aggregation ablation

Use this after confirming `fmrl_ava_glow_stable` is close to FedAvg. TWA keeps all clients in performance mode but changes the server weighting so a giant majority-class shard cannot dominate the global delta.

```bash
python run.py experiment=exp3 +method=fmrl_ava_glow_twa seed=42 dataset.preprocessing.alpha=0.1
```

Core server settings:

```yaml
sample_power: 0.75
max_client_weight_fraction: 0.40
utility_strength: 0.15
alignment_strength: 0.03
drift_penalty_strength: 0.05
critic_blend: 0.0
server_optimizer: none
```

Interpretation:

- `sample_power < 1.0` tempers sample-count dominance.
- `max_client_weight_fraction` caps the final normalized update share of any single client.
- The cap is class-agnostic and does not re-enable class-aware aggregation multipliers.
- Utility now treats very high KL as instability rather than novelty.
- Primary comparison metrics are macro-F1, balanced accuracy, worst-class F1, and per-class recall. Accuracy remains secondary under alpha=0.1.
