# DKD-FedOS Implementation

DKD-FedOS is a Sentinel-inspired dynamic knowledge-distillation strategy adapted to the CVAE-DQN blockchain-traffic intrusion detector.

## Motivation

FedGPA is useful under moderate non-IID data, but the project logs showed an extreme split where some clients had very few samples or only one local class. In that setting, a strongly personalized classifier can preserve local one-class bias and perform badly on the shared all-class test set. DKD-FedOS changes the federation target: the full CVAE-DQN model stays local as a personalized teacher, and only a compact student classifier is aggregated globally.

## Paper mapping

Sentinel uses a dual-model pFed-IDS design:

- personalized teacher model kept local;
- lightweight student model shared globally;
- class-balanced task loss;
- adaptive bidirectional knowledge distillation;
- lightweight feature alignment;
- normalized equal-weight pseudo-gradient aggregation on the server.

DKD-FedOS maps this into this codebase as follows:

| Sentinel concept | DKD-FedOS implementation |
|---|---|
| Teacher model | existing CVAE-DQN agent: `prior_net`, `recognition_net`, `value_net_main`, `generation_net` |
| Student model | `StudentIDSModel`, a compact MLP classifier over traffic features |
| Teacher logits | Q-values from `value_net_main` |
| Teacher features | normalized latent vector concatenated with normalized Q-values |
| Student features | final hidden layer of the compact student MLP |
| Task loss | class-balanced CE for teacher Q logits and student logits |
| KD loss | adaptive bidirectional KL with round-dependent temperature |
| Alignment loss | MSE + cosine alignment from teacher feature space to student feature space |
| Server aggregation | equal-weight normalized pseudo-gradient aggregation with momentum |

## Files added

- `src/models/student.py`: compact shared student model.
- `src/rl/class_balance.py`: effective-number class weights and class-balanced CE.
- `src/rl/distillation.py`: temperature schedule, bidirectional KD, and feature alignment helpers.
- `src/configs/method/dkd_fedos.yaml`: method config.

## Files modified

- `src/agents/agent.py`: adds student model, aligner, DKD losses, adaptive lambda updates, and absent-class output-row protection.
- `src/rl/local_training.py`: passes DKD options into training and logs DKD metrics.
- `src/federated/client.py`: adds `phase=dkd_fedos`; receives global student and uploads only the updated student.
- `src/federated/server.py`: adds `DKDFedOSStrategy` using normalized equal-weight pseudo-gradient aggregation.
- `scripts/experiments/*`: adds DKD-FedOS commands to main experiment scripts.

## Local objective

For each mini-batch, the teacher still trains with the RL/CVAE objective. DKD-FedOS adds Sentinel-style student knowledge transfer:

```text
L_total = L_TD + L_KL + beta_CE L_CE_teacher
        + L_task
        + lambda_KD L_KD
        + lambda_align L_align
```

where:

```text
L_task = 0.5 * CBCE(Q_teacher, y) + 0.5 * CBCE(student_logits, y)
```

`L_KD` is bidirectional teacher/student KL with adaptive temperature, and `L_align` is MSE + cosine feature alignment.

## Server aggregation

The server receives only student parameters. For each client update:

```text
g_i = theta_global_student - theta_i_student
ghat_i = g_i / (||g_i||_2 + eps)
gbar = mean_i(ghat_i)
v_r = beta * v_{r-1} + (1 - beta) * gbar
theta_global_student = theta_global_student - eta * v_r
```

This avoids sample-count domination by huge clients and makes each participating client contribute equally after normalization.

## Absent-class protection

When a local client lacks class `k`, the method zeros gradients for class `k` rows in:

- teacher Q-head `advantage_fc2`;
- student classifier head.

This prevents one-class clients from locally overwriting missing-class decision rows. Missing-class knowledge is then recovered mainly through the global student via distillation.

## Run commands

IID closed-set smoke run:

```bash
poetry run python run.py experiment=exp1 +method=dkd_fedos seed=42 \
  federated.num_clients=3 federated.num_rounds=3 \
  training.local_episodes_per_round=10 dataset.preprocessing.iid=true
```

Main non-IID run:

```bash
poetry run python run.py experiment=exp3 +method=dkd_fedos seed=42 \
  federated.num_clients=10 federated.num_rounds=100 \
  training.local_episodes_per_round=10 dataset.preprocessing.alpha=0.1 \
  2>&1 | tee dkd_fedos_exp3_alpha01_10clients.log
```

Open-set non-IID run:

```bash
poetry run python run.py experiment=exp4 +method=dkd_fedos seed=42 \
  federated.num_clients=10 federated.num_rounds=100 \
  training.local_episodes_per_round=10 dataset.preprocessing.alpha=0.1 \
  2>&1 | tee dkd_fedos_exp4_alpha01_10clients.log
```

## Important logs

Watch these metrics:

- `avg_dkd_task_loss`
- `avg_dkd_kd_loss`
- `avg_dkd_align_loss`
- `dkd_lambda_kd`
- `dkd_lambda_align`
- `dkd_temperature`
- `dkd_agreement`
- `dkd_confidence`
- `dkd_fedos_mean_student_grad_norm`
- `label_histogram`
- `label_coverage`

The method is working if weak clients no longer collapse to one-class predictions on the shared closed-set test after several rounds.
