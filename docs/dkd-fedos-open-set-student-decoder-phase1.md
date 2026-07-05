# DKD-FedOS Phase 1: Global Student Reconstruction Head

This update implements only **Phase 1** of the planned open-set redesign.
It does **not** replace the EVT evaluator yet and it does **not** implement
Yang et al.'s dynamic update stage.

## Motivation

The old open-set reconstruction path depends on the local CVAE-DQN teacher
`generation_net`. That is weak under non-IID data because a client-local
reconstruction model may not see every known class. A known class that is
missing locally can produce a high reconstruction error and be falsely rejected
as unknown.

Phase 1 prepares a safer open-set path by adding an optional reconstruction
head to the globally aggregated DKD-FedOS student. The global student can then
learn a known-class reconstruction manifold from all participating clients.

## Scope of this phase

Implemented:

- `StudentIDSModel` can optionally include a class-conditioned decoder.
- DKD dataset training can add student reconstruction loss.
- The decoder is part of the student `state_dict`, so the server aggregates it
  automatically with the student encoder/classifier.
- The feature is disabled by default so closed-set experiments are unchanged.
- E2 and E4 explicitly enable this feature because they are the open-set IID and
  open-set non-IID experiments.

Not implemented yet:

- Student-decoder EVT evaluator.
- Class-wise EVT calibration from student reconstruction errors.
- Replacement of local-generator EVT during open-set inference.
- Dynamic update / storing unknown attacks from Yang et al. This stage is not
  part of the current method.

## Model change

Closed-set behavior remains:

```text
x -> student backbone -> student classifier -> known-class logits
```

When enabled for open-set experiments, the student also has:

```text
x -> student backbone h_s
(h_s, class_condition) -> student decoder -> reconstructed x_hat
```

During Phase 1 training, the decoder uses the true known label as the class
condition:

```text
L_student_rec = MSE(x_hat, x)
```

The DKD student objective becomes:

```text
L_student =
    L_CBCE
  + lambda_KD * L_teacher_to_student
  + lambda_anchor * L_global_anchor
  + lambda_align * L_feature_alignment
  + lambda_rec * L_student_rec
```

The local teacher is still frozen during dataset DKD by default. The student
reconstruction loss updates only the student family, not `prior_net`,
`recognition_net`, `value_net_main`, or `generation_net`.

## Config contract

Default, including E1/E3 closed-set experiments:

```yaml
training:
  dkd_student_reconstruction_enabled: false
  dkd_student_reconstruction_weight: 0.0
```

Open-set Phase 1 experiments E2 and E4:

```yaml
training:
  generator:
    enabled: true
  dkd_student_reconstruction_enabled: true
  dkd_student_reconstruction_weight: 0.10
  dkd_student_decoder_hidden_dims: [128, 256]
  dkd_student_decoder_dropout: 0.05
  dkd_student_decoder_class_condition: true
```

This means:

- E1 closed-set IID: old behavior, no student decoder training.
- E3 closed-set non-IID: old behavior, no student decoder training.
- E2 open-set IID: student decoder is trained.
- E4 open-set non-IID: student decoder is trained.

## Validity logs

Open-set E2/E4 runs should show:

```text
student_reconstruction_enabled=True
avg_dkd_student_reconstruction_loss=...
dkd_student_reconstruction_enabled_rate=1.0
```

Closed-set E1/E3 runs should show the feature disabled or zero-rate:

```text
student_reconstruction_enabled=False
dkd_student_reconstruction_enabled_rate=0.0
```

## Next phase

Phase 2 should replace the local-generator EVT path with:

```text
global student classifier -> predicted known class
global student decoder -> reconstruction error
class-wise EVT threshold -> known or unknown
```

That phase will follow Yang et al.'s reconstruction-error EVT principle, but it
will not include Yang's dynamic update stage.
