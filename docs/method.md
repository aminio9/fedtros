# Method

This document describes the implemented code, not a paper-only design.

## Main Pipeline

The active experiment runner builds the original CVAE-DQN stack through `src.models.cvae_dqn.OpenSetQChainModelFactory`.

Core files:

- `src/models/models.py`: implementation of prior, recognition, Q, target-Q, and generator networks.
- `src/models/cvae_dqn.py`: canonical import path that wraps the existing implementation.
- `src/agents/agent.py`: two-optimizer training step and generator training.
- `src/rl/environment.py`: sampled tabular classification environment.
- `src/rl/local_training.py`: local replay loop used by centralized and federated runs.
- `src/evaluation/open_set.py`: high-level EVT open-set evaluation.
- `src/openset/evt.py`: EVT/GPD fitting and persistence.

`src/evaluation/openset_eval.py` remains only as a compatibility shim.

## Model Contract

The CVAE-DQN model is still a set of modules, not a single monolithic `forward`.

| Module | Input | Output | Use |
| --- | --- | --- | --- |
| Prior network | `s` | `mu_p, logvar_p` | prediction features and KL target |
| Recognition network | `s, a` | `mu_q, logvar_q` | TD latent and reconstruction latent |
| Main Q network | `z, s` | class-score/Q logits | action selection, CE, TD, evaluation |
| Target Q network | `z, s` | target Q logits | Double-DQN bootstrap |
| Generator | `z, a` | reconstructed `s_hat` | optional reconstruction training and EVT score |

`src.models.interface.CVAEQChainModelAdapter` exposes the CVAE-DQN stack through
a shared dictionary output contract used by smoke tests:

```python
{
    "logits": ...,
    "features": ...,
    "reconstruction": ...,
    "mu": ...,
    "logvar": ...,
    "q_values": ...,
    "aux": ...,
}
```

The `run.py` training and evaluation pipelines instantiate CVAE-DQN models via
`OpenSetQChainModelFactory`.

## Federated Methods

Implemented strategy overlays:

- `+method=fedavg`: sample-count FedAvg.
- `+method=fedprox`: FedAvg plus local proximal penalty.
- `+method=fmrl_ava`: two-phase FMRL-AVA selection and adaptive vector-aligned aggregation.
- `+method=fedmade`: FedMADE-inspired class-aware aggregation from label histograms, quality metrics, and client profile clusters.

The FedMADE implementation is inspired by the paper but is not a verbatim reproduction because this repository does not assume a server-side auxiliary validation dataset.

## Open-Set Method

The first-class open-set evaluator is EVT over generator reconstruction error:

1. Fit per-class EVT tails on known validation samples only.
2. Calibrate a global threshold on known validation unknown-probability scores.
3. Evaluate the open-set test tensor, where unknown labels are encoded with `open_set.evt.unknown_label_id`.
4. Reject predictions whose score exceeds the calibrated threshold.

MSP, energy, prototype-distance, Mahalanobis, and no-rejection scorers are implemented in `src/openset/scorers.py` with threshold utilities in `src/openset/thresholding.py`. They are available for tests and future baseline runners but are not yet first-class `run.py` evaluators.
The `openmax_evt` config name is a scaffold, not a real OpenMax implementation.

## Claim Discipline

The environment treats static tabular rows as an MDP with action-independent transitions. The DQN machinery should be described as a value-based classifier with replay and target-network stabilization, not as proof that the IDS task requires RL.

Paper claims require matched supervised, federated, and open-set baselines with multi-seed confidence intervals.

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
