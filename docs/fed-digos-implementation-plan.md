# Fed-DiGOS Implementation Plan and Design

Fed-DiGOS replaces the failed Feature-EVT path with a federated, student-attached, parameter-disentangled open-set generator branch.

## Motivation from run logs

The previous global Feature-EVT score failed because FoT was almost always predicted as Normal and was not sufficiently far from the Normal feature region. Known-class accuracy was high, but unknown recall stayed near zero. The fix is not a larger classifier; the fix is a student representation that explicitly preserves open-set evidence.

## Method basis

Fed-DiGOS combines these ideas:

- Federated OSR parameter disentanglement: FedPD++ separates open-set and closed-set subnetworks to avoid aggregation misalignment.
- Classification-reconstruction OSR: CROSR shows that reconstruction learning can preserve information useful for unknown detection without harming known classification.
- IDS open-set reconstruction EVT: Yang et al. use reconstruction-error distributions and EVT/GPD for unknown attack recognition in IIoT.
- Energy open-set NIDS: energy score is used as an additional classifier-side rejection signal.
- DKD/Sentinel: local teacher + lightweight shared student remains the known-class federated IDS backbone.

## DOI list used for the method

- FedPD++: `10.1007/s11263-026-02861-9`
- Yang IIoT open-set IDS: `10.1109/TIFS.2025.3546849`
- Sentinel DKD pFed-IDS: `10.1109/JIOT.2026.3650848`
- FedGPA: `10.1016/j.aiopen.2025.03.001`
- Energy-based Flow Classifier for NIDS: `10.1016/j.cose.2025.104569`
- CROSR: CVPR 2019 paper, no Crossref DOI found in the implementation notes; use CVF/arXiv citation.
- VAEMax: arXiv `2403.04193`.

## Architecture

Each client keeps the private RL teacher unchanged:

```text
teacher.prior_net_i
teacher.recognition_net_i
teacher.generation_net_i
teacher.value_net_i
```

The private teacher remains local and is not uploaded. Local teacher generator training is disabled in the Fed-DiGOS main path.

The federated student now contains:

```text
student.backbone + student.head                  # closed-set classifier
student.osr_encoder + osr_mu/logvar + osr_decoder # open-set generator branch
```

The OSR branch is inside the student and therefore federated with the student state dict. It is not a separate random CVAE and it is not the local RL teacher generator.

## Gradient isolation

Closed-set CE/KD/alignment updates:

```text
student.backbone + student.head + teacher_to_student_aligner
```

OSR reconstruction/KL/pseudo-unknown updates:

```text
student.osr_encoder + student.osr_mu_head + student.osr_logvar_head + student.osr_decoder
```

The OSR branch reads classifier features with detach by default:

```text
osr_input = backbone(x).detach()
```

So reconstruction loss does not drag classifier features around and break known-class behavior.

## Training loss

Known samples:

```text
L_known = recon_weight * MSE(x, x_hat) + beta_kl * KL(q(z|x,c) || N(0,I))
```

Pseudo-unknown samples are created by mixup, feature masking, and Gaussian boundary noise. They use a margin loss:

```text
L_pseudo = max(0, margin - recon_error(x_pseudo))
```

Total OSR loss:

```text
L_osr = L_known + margin_weight * L_pseudo
```

## Evaluation

For a test sample:

```text
logits = student(x)
c = argmax(logits)
score_gen = reconstruction_score(x conditioned on c)
score_energy = -T * logsumexp(logits / T)
score_proto = distance to nearest activation prototype of predicted class
```

Each score is calibrated with class-wise EVT/GPD using known validation samples only. Final unknown score:

```text
p_unknown = max(p_gen, p_energy, p_proto)
```

Final decision:

```text
Unknown if any calibrated gate rejects.
Otherwise return predicted known class c.
```

## Logs added

Client logs include:

```text
Fed-DiGOS local OSR | known=... class_counts=... coverage=... entropy=...
steps=... recon=... kl=... pseudo_score=... known_score=... gap=...
margin_ok=... delta_norm=...
```

Server logs include:

```text
DKD-FedOS/Fed-DiGOS aggregation | round=... cls_included=... osr_included=... osr_excluded=...
```

Evaluation logs include:

```text
FED-DIGOS OPEN-SET ACTIVE
Fed-DiGOS calibration | class=... T_gen=... T_energy=... T_proto=...
Fed-DiGOS final open-set | AUROC=... UnknownRecall=... KnownFU=...
```

Artifacts include:

```text
open_set_scores.csv
fed_digos_evt_thresholds.json
fed_digos_prototypes.json
fed_digos_calibration_scores.csv
known_unknown_score_quantiles.json
score_overlap_report.json
before_osr_confusion_matrix.csv
after_osr_confusion_matrix.csv
openset_report.txt
open_set_metrics.json
```
