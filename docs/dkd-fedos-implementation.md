# DKD-FedOS v2 Implementation

DKD-FedOS is a Sentinel-inspired dynamic knowledge-distillation strategy adapted to the CVAE-DQN blockchain-traffic detector.  It is a separate method from FedGPA and runs with:

```bash
poetry run python run.py experiment=exp3 +method=dkd_fedos seed=42 \
  federated.num_clients=10 federated.num_rounds=100 \
  training.local_episodes_per_round=10 dataset.preprocessing.alpha=0.1
```

## Paper-faithful design target

Sentinel trains a personalized teacher and a lightweight shared student on each client.  Only the student is sent to the server.  The server aggregates student pseudo-gradients using L2 normalization, equal weighting, and momentum.  DKD-FedOS keeps that federation pattern but adapts the teacher to this project:

- **Teacher:** local CVAE-DQN model (`prior_net`, `recognition_net`, `value_net_main`, `generation_net`)
- **Student:** shared lightweight MLP classifier (`StudentIDSModel`)
- **Uploaded:** student parameters only
- **Aggregated:** student parameters only
- **Final inference:** teacher, because the teacher owns the CVAE-DQN and open-set reconstruction path

## Important v2 correction

The first DKD-FedOS implementation mixed teacher RL loss, teacher CE, student CE, KD, and alignment in one backward pass.  That made absent-class protection too blunt: it could block both harmful local gradients and useful global KD gradients.

v2 separates the update into three stages.

### Stage A: local teacher update

The CVAE-DQN teacher learns from local evidence:

```text
L_teacher_local = TD_loss + KL_loss + teacher_CBCE
```

Local missing-class output rows are protected here, so a client that only sees one or two classes cannot overwrite every absent output row.

### Stage B: student update

The shared student learns local labels and teacher knowledge:

```text
L_student = student_CBCE + lambda_KD * KD(teacher -> student) + lambda_align * alignment
```

The local student CE is computed only over locally present classes.  This prevents local CE from pushing absent class logits down.

### Stage C: global student -> teacher update

The teacher receives global knowledge from the student:

```text
L_teacher_global = lambda_KD * KD(student -> teacher)
```

Absent-class protection is intentionally **not** applied here.  This is the missing-class transfer path.  If a client has no samples for class `k`, the only safe way to recover class `k` knowledge is through the global student.

## Warm-up and gating

DKD-FedOS v2 uses staged KD because the teacher is a CVAE-DQN/RL model rather than Sentinel's plain DNN classifier.

Default schedule:

```yaml
dkd_teacher_to_student_start_round: 4
dkd_student_to_teacher_start_round: 6
dkd_alignment_start_round: 4
dkd_min_student_confidence: 0.0
```

Round 1 to 3 trains teacher/student with local losses only.  Teacher-to-student KD and alignment start later.  Student-to-teacher KD starts after the student has received global aggregation for multiple rounds.

## Server aggregation

The server keeps the Sentinel-style normalized pseudo-gradient update:

```text
g_i = theta_global_student - theta_i_student
g_hat_i = g_i / (||g_i||_2 + eps)
g_bar = mean(g_hat_i)
v = beta * v + (1 - beta) * g_bar
theta_global_student = theta_global_student - eta * v
```

v2 also adds reliability filtering:

```yaml
min_reliable_samples: 500
```

Clients below this threshold still receive the global student and are evaluated, but they are excluded from the server update.  This avoids giving a 45-sample client the same aggregation authority as a large, stable client.

## Logging added

Each DKD-FedOS run now reports:

- local class histogram, present classes, missing classes, imbalance ratio
- teacher local/global closed-set evaluation
- student local/global closed-set evaluation
- teacher batch accuracy
- student batch accuracy
- teacher-student agreement
- correct agreement
- teacher/student confidence
- teacher-to-student and student-to-teacher KD losses
- KD enable rates
- alignment loss and alignment score
- included/excluded aggregation clients
- raw and normalized student gradient norms

These logs are required before trusting a DKD-FedOS result.  If the global student improves but the teacher does not, the failure is in student-to-teacher transfer.  If the global student does not improve, the failure is in student training or server aggregation.

## Recommended validation order

Do not start with the harshest split.

```bash
# smoke test
poetry run python run.py experiment=exp1 +method=dkd_fedos seed=42 \
  federated.num_clients=3 federated.num_rounds=3 \
  training.local_episodes_per_round=5 dataset.preprocessing.iid=true

# moderate non-IID
poetry run python run.py experiment=exp3 +method=dkd_fedos seed=42 \
  federated.num_clients=10 federated.num_rounds=30 \
  training.local_episodes_per_round=5 dataset.preprocessing.alpha=0.5

# harsh non-IID after the previous runs are stable
poetry run python run.py experiment=exp3 +method=dkd_fedos seed=42 \
  federated.num_clients=10 federated.num_rounds=100 \
  training.local_episodes_per_round=10 dataset.preprocessing.alpha=0.1
```

For open-set experiments, ensure the unknown class is not part of known training.  If FoT is the unknown class, the known classifier should train on Normal, BP, DoS, and MitM only.

## v3 correction: dataset DKD + warm student aggregation

After inspecting the IID smoke-test logs, the local student was learning but the aggregated global student collapsed to a single class.  DKD-FedOS v3 fixes the two root causes.

### 1. DKD no longer depends only on replay-buffer batches

Sentinel Algorithm 1 trains teacher and student on local dataset mini-batches `(X, Y) in D_i`.  v3 adds a dataset-based DKD loop after the RL teacher loop:

```text
run RL CVAE-DQN teacher training
then run DKD mini-batch training over env.all_features_s / env.all_labels_a_t
```

This keeps the RL model behavior while making the teacher/student distillation closer to the paper.  Config controls:

```yaml
dkd_dataset_training: true
dkd_local_epochs: 1
dkd_batch_size: 256
```

### 2. Early global student aggregation uses equal averaging

The normalized-gradient server update from Sentinel can be unstable when the global student is random and local students move far from it.  v3 therefore uses equal averaging during warm-up, then switches to normalized-gradient aggregation:

```yaml
student_aggregation_mode: warm_avg_then_normalized
student_avg_warmup_rounds: 3
normalized_server_momentum: 0.9
```

Expected behavior in IID smoke tests:

```text
LOCAL_STUDENT_AFTER_LOCAL_TRAIN: high and improving
GLOBAL_STUDENT_AFTER_SERVER_AGG: should not collapse to 7.14%
```

If the global student still collapses, inspect:

```text
prediction_max_ratio
dkd_fedos_distance_to_avg_before
dkd_fedos_distance_to_avg_after
dkd_fedos_global_norm_before
dkd_fedos_global_norm_after
```

### 3. Evaluation names are now method-correct

DKD-FedOS uses:

```text
TEACHER_AFTER_LOCAL_TRAIN
LOCAL_STUDENT_AFTER_LOCAL_TRAIN
GLOBAL_STUDENT_AFTER_SERVER_AGG
```

The old `TEACHER_GLOBAL_POST_AGG` naming was removed because the teacher is not globally aggregated in Sentinel-style training.

## DKD-FedOS v4 safety update: dataset DKD does not break RL

Dataset mini-batch DKD now uses `env.all_features_s` and `env.all_labels_a_t` only as a supervised local dataset for the student.  It does **not** push samples into the environment, does **not** step the environment, and does **not** change the replay buffer.

The safe default is:

```yaml
training:
  dkd_dataset_training: true
  dkd_dataset_update_teacher: false
  dkd_update_teacher_from_student: false
  dkd_teacher_task_weight: 0.0
  dkd_student_to_teacher_start_round: 999
```

With this setup, RL remains the owner of teacher updates:

```text
Replay/RL path updates:
  prior_net
  recognition_net
  value_net_main
  target_q

Dataset DKD path updates:
  student_model
  teacher_to_student_aligner
```

The teacher is used as a frozen guide during dataset DKD:

```text
with no_grad:
  teacher_logits, teacher_features = CVAE-DQN teacher(X)

student_loss = student_CBCE + teacher_to_student_KD + feature_alignment
```

Student-to-teacher KD is disabled by default because a weak global student can poison the RL teacher.  Enable it only as a later ablation after `GLOBAL_STUDENT_AFTER_SERVER_AGG` is stable and non-collapsed.

The training log now includes:

```text
dkd_dataset_updates_teacher
dkd_update_teacher_from_student
dkd_replay_buffer_size_before
dkd_replay_buffer_size_after
dkd_replay_buffer_delta
```

For the safe default, expected values are:

```text
dkd_dataset_updates_teacher = 0
dkd_update_teacher_from_student = 0
dkd_replay_buffer_delta = 0
```

## DKD-FedOS v5 student-anchor fix

v5 fixes the issue where one-class or low-coverage clients barely changed the local student, or changed it only toward their local class.  The student now uses full-logit class-balanced CE plus a frozen copy of the incoming global student as an anchor.

At the start of each client round:

```text
server global student -> client student
client copies that loaded student into student_anchor_model (frozen)
```

During dataset DKD:

```text
L_student = CBCE_full(student_logits, y)
          + lambda_anchor * KL(anchor_logits || student_logits)
          + lambda_KD * KD(teacher_logits || student_logits)
          + lambda_align * feature_alignment
```

The anchor weight grows when label coverage is low:

```text
anchor_weight = base_anchor_weight * max(min_anchor, (1 - label_coverage)^power)
```

So a one-class client can still learn its local class, but the frozen global-student anchor prevents it from erasing logits for missing classes.

The teacher remains RL-safe by default:

```yaml
dkd_dataset_update_teacher: false
dkd_update_teacher_from_student: false
```

Only the student and aligner are updated by dataset DKD.

## v5 aggregation fix

Server aggregation is now reliability-weighted by default:

```yaml
student_aggregation_mode: reliability_weighted_average
```

Each client receives a reliability score from:

```text
sample_factor * label_coverage * class_entropy
```

This prevents a high-sample but one-class client from dominating the global student.  The monitor log includes reliability weights for each included client.

## v5 stronger student

The default student is now stronger but still much smaller than the CVAE-DQN teacher:

```yaml
dkd_student_hidden_dims: [256, 128, 64]
dkd_student_activation: gelu
dkd_student_norm: layernorm
dkd_student_dropout: 0.1
```

## v5 debugging logs

Client fit metrics now include:

```text
dkd_student_norm_before_load
dkd_student_norm_after_load
dkd_student_norm_after_train
dkd_student_load_delta_norm
dkd_student_train_delta_norm
student_before_local_accuracy
student_before_local_prediction_histogram
avg_dkd_global_anchor_loss
dkd_global_anchor_weight
```

These show whether the global student is actually loaded and whether local student training changes it.

## Phase 1 open-set student reconstruction head

The Phase 1 open-set update adds an optional reconstruction decoder to the
shared `StudentIDSModel`.  The old closed-set student path is unchanged unless
`training.dkd_student_reconstruction_enabled=true`.

When enabled, the student contains:

```text
student backbone E_s
student classifier C_s
student decoder G_s
```

The decoder receives the student feature vector and a class condition:

```text
h_s = E_s(x)
logits_s = C_s(h_s)
x_hat = G_s(h_s, y)
L_rec = ||x - x_hat||^2
```

This reconstruction loss is added only to the student-side DKD objective.  It
does not update the local CVAE-DQN teacher and it does not change replay-buffer
RL.  Because the decoder is part of the student `state_dict`, DKD-FedOS server
aggregation automatically shares it with the student encoder/classifier.

The feature is enabled only for open-set experiments E2 and E4 in this phase:

```yaml
training:
  dkd_student_reconstruction_enabled: true
  dkd_student_reconstruction_weight: 0.10
```

Closed-set E1/E3 keep the previous classifier-only DKD-FedOS behavior.  The
local teacher `generation_net` still exists and can train in open-set runs, but
Phase 1 prepares the global student decoder that Phase 2 will use for class-wise
EVT unknown rejection.

## Phase 2 open-set EVT refactor

Phase 2 replaces the local teacher/generator EVT path with a global student-decoder EVT path for open-set experiments E2/E4. The local CVAE-DQN teacher remains private and personalized. Only the student family is aggregated:

```text
student encoder + student classifier + student decoder
```

EVT calibration is class-wise and uses the reconstruction-error tail of correctly classified known validation samples. The high threshold is selected with a Mean Excess Function heuristic and the exceedances are fitted with a Generalized Pareto Distribution by maximum likelihood. The final open-set decision uses only:

```text
reconstruction_error > class_evt_threshold
```

The Yang 2025 dynamic update stage is intentionally not implemented.
