# Logging And Tracking

Tracking is local-only through `src/tracking/local.py`.

Each initialized run writes:

- `run.log`: normal execution log.
- `debug.log`: debug-level log.
- `metrics.jsonl`: append-only metrics.
- `metrics.csv`: table regenerated from JSONL.
- `metadata.json`: experiment name, run ID, timestamp, seed, device, git commit if available, dataset, model, method, Python, platform, and PyTorch version.
- `config.yaml`: raw Hydra config.
- `resolved_config.yaml`: resolved Hydra config.
- `federated_history.csv`: long-format per-round Flower metrics when federated training is run.
- `fmrl_ava_monitoring.jsonl`: per-round FMRL-AVA records when `+method=fmrl_ava` is used.
- `fedgpa_monitoring.jsonl`: per-round FedGPA prototype and personalized aggregation records when `+method=fedgpa` is used.

Federated simulations use per-client logger names such as `Client.1` and `Client.2`, so console output stays tagged even when clients execute in-process. Progress bars are disabled when stdout is not a tty to avoid overwriting log lines.

FMRL-AVA monitoring records include selected clients, selected fraction, per-client utilities, `base_aggregation_weight`, `alignment_cosine`, `alignment_multiplier`, final `aggregation_weight`, `support_reward`, `validation_team_reward`, raw validation reward before EMA when available, and the final mixer target used to train the server-side critics and mixer.

FedGPA logs include `alpha_self`, `beta_self`, prototype class coverage, and round aggregation metadata. Local-training logs now also report `Avg CE Loss`, `balanced_policy_accuracy`, `per_class_policy_accuracy`, and `mean_reward_weight` when class-balanced RL stabilizers are enabled.

No online service is required.


## DKD-FedOS metrics

`+method=dkd_fedos` adds dynamic distillation metrics to the local training log:

- `avg_dkd_task_loss`: class-balanced teacher/student task loss.
- `avg_dkd_kd_loss`: adaptive bidirectional knowledge-distillation loss.
- `avg_dkd_align_loss`: teacher-student feature alignment loss.
- `dkd_lambda_kd` and `dkd_lambda_align`: adaptive auxiliary weights.
- `dkd_temperature`: round-dependent KD temperature.
- `dkd_agreement` and `dkd_confidence`: teacher/student prediction agreement and joint confidence.
- `dkd_fedos_mean_student_grad_norm`: server-side normalized pseudo-gradient diagnostic.

Always inspect `label_histogram` and `label_coverage` when debugging non-IID results. A client with one local class cannot learn all classes from local labels; DKD-FedOS is designed to transfer missing-class knowledge through the global student.

## DKD-FedOS v2 update

The DKD-FedOS implementation now follows a stricter Sentinel-style teacher/student separation. The CVAE-DQN teacher remains local, while only the lightweight student is federated. Local teacher learning, student learning, and global student-to-teacher KD are separated into distinct optimization stages. This avoids blocking missing-class knowledge transfer when absent-class local-gradient protection is enabled.

New logs distinguish teacher and student evaluation:
- `TEACHER_LOCAL_PRE_AGG`
- `STUDENT_LOCAL_PRE_AGG`
- `TEACHER_GLOBAL_POST_AGG`
- `STUDENT_GLOBAL_POST_AGG`

The server also reports included/excluded clients, reliability filtering, raw gradient norms, normalized gradient norms, and student aggregation statistics.
