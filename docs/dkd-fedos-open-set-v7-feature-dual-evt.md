# DKD-FedOS v7 Open-Set Replacement: Global Feature EVT + Gated Local Generator EVT

This update uses a clean two-boundary open-set detector:

1. **Primary detector:** global student feature-distance EVT.
2. **Optional enhancement:** support-gated local teacher-generator reconstruction EVT.

## Why the method changed

The method uses the globally aggregated DKD-FedOS student representation. Instead of asking whether a sample can be reconstructed by a shared decoder, it asks whether the sample lies inside the compact feature region of the predicted known class.

## Main backend: `student_feature_evt`

The default open-set backend is now:

```yaml
open_set:
  evt:
    backend: student_feature_evt
    score: mahalanobis_feature_distance
```

Training remains the same as DKD-FedOS v6 closed/open known-class training:

- local RL/CVAE-DQN teacher remains private;
- global student receives teacher-to-student KD;
- global anchor prevents local collapse;
- feature alignment starts from round 2;
- only student parameters and audit summaries are aggregated.

## Feature EVT calibration

After federated training, the global student checkpoint is loaded from:

```text
dkd_fedos_student_latest.pt
```

For each known validation sample:

```text
h = E_s(x)
logits = C_s(h)
```

For each known class `k`, correctly classified validation samples are used to estimate:

```text
mu_k      = mean feature vector
sigma2_k  = diagonal feature variance with shrinkage
```

Then class-wise diagonal Mahalanobis distance is computed:

```text
d_k(x) = sum_j ((h_j - mu_k,j)^2 / (sigma2_k,j + eps))
```

The upper tail of `d_k` is fitted with the existing Yang-style EVT/GPD module:

```text
threshold selection: MEF with quantile fallback
GPD parameter fitting: MLE
class-wise threshold: T_k
```

At inference:

```text
c = argmax C_s(E_s(x))
d = d_c(x)
if d > T_c:
    Unknown
else:
    class c
```

No uncertainty score, disagreement score, or weighted fusion is used.

## Optional backend: `dual_boundary_evt`

The optional dual-boundary backend applies the global feature EVT first, then applies local teacher-generator EVT only if the local generator has enough clean support for the predicted class.

Decision rule:

```text
if global_feature_distance > T_global,c:
    Unknown
elif local_generator_valid(i,c) and local_reconstruction_error > T_local,i,c:
    Unknown
else:
    class c
```

The local generator is valid only when:

```text
clean_count_i,c >= local_min_clean_count
EVT fitting succeeds
known false-reject rate <= local_max_known_reject_rate
```

This protects non-IID clients from falsely rejecting known classes that were missing or nearly missing locally.

Important: the final server-side DKD-FedOS evaluation cannot use local teacher/generator modules because DKD-FedOS intentionally uploads only the student. Therefore, `dual_boundary_evt` in final server evaluation runs the global feature EVT boundary and logs that the local branch is unavailable. To debug the local branch, enable client-side EVT ablation:

```yaml
open_set:
  evt:
    backend: dual_boundary_evt
    client_eval_enabled: true
    dual_boundary:
      enabled: true
```

## Backends

| Backend | Purpose | Default? |
|---|---|---:|
| `student_feature_evt` | Main v7 method: global student feature distance + class-wise EVT | Yes |
| `dual_boundary_evt` | Feature EVT plus support-gated local generator EVT | Ablation/enhancement |
| `teacher_generator` | Legacy local generator EVT | Ablation only |

## Correct experiment behavior

E1/E3 closed-set experiments remain unchanged: no open-set EVT.

E2/E4 open-set experiments now use feature EVT by default:

```text
E2: open-set IID, backend=student_feature_evt
E4: open-set non-IID, backend=student_feature_evt
```

## Required validity logs

For E2/E4 main runs, logs must include:

```text
GLOBAL STUDENT FEATURE-EVT ACTIVE
score=mahalanobis_feature_distance
OPEN-SET PROTOCOL ACTIVE
known_classes=[Normal, BP, DoS, MitM]
heldout_unknown=FoT
num_actions=4
Fitted global student Feature-EVT | class=...
Open-set metrics | backend=student_feature_evt
```

Logs should show the Feature-EVT backend and should not include reconstruction-decoder training metrics.
