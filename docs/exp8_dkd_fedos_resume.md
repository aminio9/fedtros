# Resume Exp8 DKD-FedOS/Fed-DiGOS from a student checkpoint

The output folder contains strict-FL student checkpoints such as:

```text
dkd_fedos_student_latest.pt
dkd_fedos_student_round_0052.pt
```

These checkpoints store only the shared student model under `student_model`. They are intentionally not full teacher/agent checkpoints. The resume fix lets `federated.resume_from` load this student-only checkpoint and initialize the global student before continuing federated training.

Example: continue the MitM holdout run from round 52 to round 100:

```bash
bash scripts/experiments/resume_exp8_dkd_fedos.sh \
  outputs/e8_mitm_unknown_dkd_fedos_seed42/dkd_fedos_student_latest.pt \
  52 \
  MitM \
  e8_mitm_resume_from52_seed42 \
  100
```

The script computes `remaining_rounds = 100 - 52 = 48`, sets `federated.resume_round_offset=52`, and saves the next checkpoint as `dkd_fedos_student_round_0053.pt` in the new run directory.

Use a new `tracking.run_id` to avoid overwriting the original round files.
