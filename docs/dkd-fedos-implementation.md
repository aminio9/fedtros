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
