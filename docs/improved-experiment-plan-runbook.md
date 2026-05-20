# Improved Experiment Plan Runbook

This runbook binds `D:\Research\Experimentplan\Improved_Experiment_Plan.docx`
to the executable scripts, Hydra settings, output files, tables, and plots in
this repository. It is the source of truth for reproducing the Q1 experiment
package and for writing the experiment/results section.

Companion documents:

- [q1-experiment-and-results-demo.md](q1-experiment-and-results-demo.md): paper-style experiment and results draft using the current local artifacts.
- [tiny-experiment-validation.md](tiny-experiment-validation.md): short validation plan that checks the whole pipeline with tiny round/client/episode settings.

## Coverage Status

| Plan requirement from the Word document | Repository support | Runbook section |
|---|---:|---|
| Primary B-NAT experiment | Covered | Sections 1-5 |
| Closed-set known-class metrics | Covered | Sections 4, 13 |
| Open-set unknown detection with EVT | Covered | Sections 4, 7, 13 |
| FedAvg, FedProx, and Proposed FMRL_LA comparison | Covered | Sections 5-6 |
| IID and non-IID Dirichlet partitions | Covered | Sections 3, 5-6, 9 |
| Scalability over N = 3, 10, 20, 50, 100 clients | Covered, requires aggregation CSV | Section 8 |
| Seed robustness over alpha = 0.1, 0.5, 1.0, 10, IID | Covered, requires aggregation CSV | Section 10 |
| Ablation: centralized, FL, OSR, full method | Partly covered; centralized runs use `scripts/train.py` plus evaluation | Section 11 |
| All 14 Q1 dashboard plots | Renderer covered; some plots require suite-level CSVs | Section 12 |
| External B-TAT, ToN-IoT, CIC-IDS2017 generalization | Planned, blocked until dataset configs/raw data are added | Section 9 |
| Statistical analysis and paper reporting | Covered as required reporting contract | Section 14 |
| Tiny end-to-end validation | Covered | Section 2 and companion tiny document |

## 1. Dataset and Label Alignment

The B-NAT source CSV contains these labels:

```text
Normal, BP, DoS, MitM, FoT
```

The executable primary open-set protocol uses four known labels and holds `FoT`
out as the unknown attack:

```powershell
dataset.known_labels=[Normal,BP,DoS,MitM] dataset.preprocessing.known_labels=[Normal,BP,DoS,MitM] model.num_actions=4 model.state_dim=31
```

`model.state_dim=31` matches the current preprocessing output:

- 18 numeric features
- one-hot categorical feature dimensions `[3,5,5]`

Use the exact label spelling from the repository (`FoT`, `MitM`). The Word plan
and older plotting template sometimes use `Fot` or `MITM`; treat those as
document/template spellings, not executable labels.

The canonical local raw file is:

```text
data/raw/BNaT.csv
```

If it is missing, fetch B-NAT from the official B-NAT repository or landing page
and place the extracted CSV at that path.

## 2. Common Commands and Modes

Run commands from the repository root:

```powershell
cd D:\Research\cf_marlos
poetry install -E dev
```

Full Q1 runs should use 100 federated logical rounds:

```powershell
$BASE = "dataset=bnat model=openset_qchain agent=double_q optimizer=adam scheduler=none experiment=baseline device.prefer=cpu dataset.known_labels=[Normal,BP,DoS,MitM] dataset.preprocessing.known_labels=[Normal,BP,DoS,MitM] model.num_actions=4 model.state_dim=31 training.batch_size=512 training.local_episodes_per_round=15 training.steps_per_episode=100 training.min_buffer_size=512 open_set.evt.enabled=true"
```

Tiny validation runs should avoid overwriting `data/processed`. Use a run-local
processed directory:

```powershell
$TINY = "dataset=bnat model=openset_qchain agent=double_q optimizer=adam scheduler=none experiment=baseline device.prefer=cpu dataset.known_labels=[Normal,BP,DoS,MitM] dataset.preprocessing.known_labels=[Normal,BP,DoS,MitM] model.num_actions=4 model.state_dim=31 training.batch_size=64 training.local_episodes_per_round=1 training.steps_per_episode=3 training.min_buffer_size=2 open_set.evt.enabled=true federated.num_clients=2 dataset.preprocessing.num_clients=2 federated.num_rounds=1 evaluation.batch_size=2048"
```

Then add per-run overrides:

```powershell
tracking.run_id=tiny_e2e_validation dataset.preprocessing.output_dir=outputs/tiny_e2e_validation/processed checkpointing.dir=outputs/tiny_e2e_validation checkpointing.best_model_path=outputs/tiny_e2e_validation/best_model.pt checkpointing.latest_checkpoint_path=outputs/tiny_e2e_validation/latest_checkpoint.pt checkpoint.path=outputs/tiny_e2e_validation/latest_checkpoint.pt evaluation.checkpoint_path=outputs/tiny_e2e_validation/best_model.pt
```

For a quick smoke test that does not use B-NAT:

```powershell
poetry run python scripts/smoke_test.py experiment=smoke device.prefer=cpu tracking.run_id=smoke_cpu model.num_actions=4 model.state_dim=31
```

Expected smoke artifacts:

```text
outputs/smoke_cpu/
  run.log
  debug.log
  metrics.csv
  metrics.jsonl
  smoke_test_metrics.json
```

## 3. Primary Protocol

Use B-NAT as the primary dataset. The known-class classifier trains only on:

```text
Normal, BP, DoS, MitM
```

Open-set evaluation uses held-out `FoT` records as unknown samples encoded as
`-1`. The same train/validation/test split and client partition must be reused
for all methods within each seed and heterogeneity setting.

Primary reporting settings:

| Item | Final Q1 setting |
|---|---|
| Primary dataset | B-NAT |
| Known labels | `Normal`, `BP`, `DoS`, `MitM` |
| Unknown label | `FoT` |
| Main client count | 10 |
| Main hard non-IID alpha | 0.1 |
| Mild non-IID alpha | 10 |
| Sensitivity alpha values | 0.1, 0.5, 1.0, 10, IID |
| Full federated rounds | 100 logical rounds |
| Proposed method | `FMRL_LA` |
| Baselines | `FedAvg`, `FedProx`, closed-set softmax, centralized variants |
| Closed-set primary metrics | macro F1, balanced accuracy |
| Open-set primary metrics | AUROC, AUPRC, FPR@95%TPR, unknown F1 |

For `FMRL_LA`, the Flower simulation uses two internal Flower phases per
logical round. Therefore `federated.num_rounds=100` means 200 Flower rounds.

## 4. Single Proposed Run

This command runs preprocessing, federated training, and evaluation:

```powershell
poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fmrl_alpha01_seed42 seed=42 federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
```

Render all supported plots from the run directory:

```powershell
poetry run python scripts/plot.py run_dir=outputs/fmrl_alpha01_seed42
```

Expected core artifacts:

```text
outputs/fmrl_alpha01_seed42/
  run.log
  debug.log
  metrics.jsonl
  metrics.csv
  metadata.json
  config.yaml
  resolved_config.yaml
  best_model.pt
  latest_checkpoint.pt
  global_model_latest.pt
  global_model_round_*.pt
  federated_round_metrics.csv
  fmrlla_monitoring.jsonl
  evaluation_metrics.json
  test_metrics.json
  open_set_metrics.json
  open_set_scores.csv
  open_set_roc_curve.csv
  open_set_pr_curve.csv
  before_osr_confusion_matrix.csv
  after_osr_confusion_matrix.csv
  evt/
  plots/
    plot_manifest.json
```

Some plot files may intentionally show "Missing data" unless the suite-level CSV
for that plot has been created. Section 12 lists every required CSV.

## 5. Main Method Comparison: Hard Non-IID Alpha 0.1

```powershell
poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fmrl_alpha01_seed42 seed=42 federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false

poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fedavg_alpha01_seed42 seed=42 federated.strategy.name=fedavg experiment.method=FedAvg federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false

poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fedprox_alpha01_seed42 seed=42 federated.strategy.name=fedprox experiment.method=FedProx federated.server.proximal_mu=0.01 federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false

poetry run python scripts/compare_runs.py $BASE tracking.run_id=compare_alpha01_seed42 runs='[outputs/fmrl_alpha01_seed42,outputs/fedavg_alpha01_seed42,outputs/fedprox_alpha01_seed42]'

poetry run python scripts/plot.py run_dir=outputs/compare_alpha01_seed42
```

`compare_runs.py` writes:

```text
outputs/compare_alpha01_seed42/run_comparison.csv
outputs/compare_alpha01_seed42/comparison_metrics.csv
```

`comparison_metrics.csv` is the data source for Plot 4.

## 6. Main Method Comparison: Mild Non-IID Alpha 10

```powershell
poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fmrl_alpha10_seed42 seed=42 federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=10 dataset.preprocessing.iid=false

poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fedavg_alpha10_seed42 seed=42 federated.strategy.name=fedavg experiment.method=FedAvg federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=10 dataset.preprocessing.iid=false

poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fedprox_alpha10_seed42 seed=42 federated.strategy.name=fedprox experiment.method=FedProx federated.server.proximal_mu=0.01 federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=10 dataset.preprocessing.iid=false

poetry run python scripts/compare_runs.py $BASE tracking.run_id=compare_alpha10_seed42 runs='[outputs/fmrl_alpha10_seed42,outputs/fedavg_alpha10_seed42,outputs/fedprox_alpha10_seed42]'

poetry run python scripts/plot.py run_dir=outputs/compare_alpha10_seed42
```

`comparison_metrics.csv` is the data source for Plot 3.

## 7. Open-Set Evaluation and Regeneration

`scripts/reproduce_experiment.py` already runs evaluation after federated
training. To regenerate evaluation from a saved checkpoint:

```powershell
poetry run python scripts/evaluate.py $BASE tracking.run_id=eval_fmrl_alpha01_seed42 checkpoint.path=outputs/fmrl_alpha01_seed42/best_model.pt evaluation.checkpoint_path=outputs/fmrl_alpha01_seed42/best_model.pt

poetry run python scripts/plot.py run_dir=outputs/eval_fmrl_alpha01_seed42
```

Open-set source files:

```text
open_set_scores.csv
open_set_metrics.json
open_set_roc_curve.csv
open_set_pr_curve.csv
before_osr_confusion_matrix.csv
after_osr_confusion_matrix.csv
```

Plot 5 uses `open_set_scores.csv` and, when available,
`open_set_metrics.json` for the calibrated EVT threshold.

## 8. Scalability Runs for Plot 1

Run the proposed method for each client count:

```powershell
foreach ($n in 3,10,20,50,100) {
  poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=scalability_n${n}_seed42 seed=42 federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=$n dataset.preprocessing.num_clients=$n federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
}
```

Create `outputs/scalability_suite/scalability.csv`:

```text
num_clients,final_accuracy
3,<test/accuracy from outputs/scalability_n3_seed42/evaluation_metrics.json>
10,<test/accuracy from outputs/scalability_n10_seed42/evaluation_metrics.json>
20,<test/accuracy from outputs/scalability_n20_seed42/evaluation_metrics.json>
50,<test/accuracy from outputs/scalability_n50_seed42/evaluation_metrics.json>
100,<test/accuracy from outputs/scalability_n100_seed42/evaluation_metrics.json>
```

Render:

```powershell
poetry run python scripts/plot.py run_dir=outputs/scalability_suite
```

## 9. Non-IID and Cross-Dataset Conditions

For the non-IID class-distribution plot:

```powershell
poetry run python scripts/preprocess.py $BASE tracking.run_id=noniid_alpha01_seed42 seed=42 federated.num_clients=10 dataset.preprocessing.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false

poetry run python scripts/plot.py run_dir=outputs/noniid_alpha01_seed42
```

The source data file is `client_class_distribution.csv`.

For cross-dataset Plot 8, the Word plan requires B-TAT, ToN-IoT, and
CIC-IDS2017. This repository currently has B-NAT support and utility scripts in
`data/raw/`, but no completed dataset configs for those three external datasets.
Before using Plot 8 as final evidence:

1. Place each raw dataset under `data/raw/`.
2. Add dataset-specific preprocessing config or equivalent overrides.
3. Register known/unknown label mappings.
4. Run the same method/baseline protocol.
5. Write `outputs/cross_dataset_suite/cross_dataset_metrics.csv`.

Required format:

```text
dataset,metric,metric_value
B-TAT,f1,<closed-known macro F1>
B-TAT,auroc,<open-set AUROC>
ToN-IoT,f1,<closed-known macro F1>
ToN-IoT,auroc,<open-set AUROC>
CIC-IDS2017,f1,<closed-known macro F1>
CIC-IDS2017,auroc,<open-set AUROC>
```

Render:

```powershell
poetry run python scripts/plot.py run_dir=outputs/cross_dataset_suite
```

## 10. Seed Robustness for Plot 11

Run at least five seeds for development and 10-15 seeds for final Q1 reporting
if compute allows:

```powershell
foreach ($seed in 42,43,44,45,46) {
  poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fmrl_alpha01_seed$seed seed=$seed federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
  poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fmrl_alpha05_seed$seed seed=$seed federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.5 dataset.preprocessing.iid=false
  poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fmrl_alpha1_seed$seed seed=$seed federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=1.0 dataset.preprocessing.iid=false
  poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fmrl_alpha10_seed$seed seed=$seed federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=10 dataset.preprocessing.iid=false
  poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=fmrl_iid_seed$seed seed=$seed federated.strategy.name=fmrl_la experiment.method=FMRL_LA federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.iid=true
}
```

Create `outputs/seed_robustness_suite/seed_robustness.csv`:

```text
seed,heterogeneity,accuracy
42,0.1,<test/accuracy>
42,0.5,<test/accuracy>
42,1.0,<test/accuracy>
42,10,<test/accuracy>
42,IID,<test/accuracy>
...
```

Render:

```powershell
poetry run python scripts/plot.py run_dir=outputs/seed_robustness_suite
```

## 11. Ablation Runs for Plot 14

```powershell
poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=ablation_full seed=42 federated.strategy.name=fmrl_la experiment.method=FMRL_LA open_set.evt.enabled=true federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false

poetry run python scripts/reproduce_experiment.py $BASE tracking.run_id=ablation_fedavg_no_osr seed=42 federated.strategy.name=fedavg experiment.method=FedAvg open_set.evt.enabled=false federated.num_clients=10 dataset.preprocessing.num_clients=10 federated.num_rounds=100 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false

poetry run python scripts/train.py $BASE tracking.run_id=ablation_central_osr seed=42 experiment.method=Centralized_OSR open_set.evt.enabled=true training.epochs=100 federated.num_clients=10 dataset.preprocessing.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false

poetry run python scripts/train.py $BASE tracking.run_id=ablation_central_no_osr seed=42 experiment.method=Centralized_No_OSR open_set.evt.enabled=false training.epochs=100 federated.num_clients=10 dataset.preprocessing.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
```

If centralized tensors are missing, run preprocessing first:

```powershell
poetry run python scripts/preprocess.py $BASE tracking.run_id=ablation_preprocess seed=42 federated.num_clients=10 dataset.preprocessing.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
```

Evaluate centralized checkpoints:

```powershell
poetry run python scripts/evaluate.py $BASE tracking.run_id=eval_ablation_central_osr seed=42 experiment.method=Centralized_OSR open_set.evt.enabled=true checkpoint.path=outputs/ablation_central_osr/best_model.pt evaluation.checkpoint_path=outputs/ablation_central_osr/best_model.pt federated.num_clients=10 dataset.preprocessing.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false

poetry run python scripts/evaluate.py $BASE tracking.run_id=eval_ablation_central_no_osr seed=42 experiment.method=Centralized_No_OSR open_set.evt.enabled=false checkpoint.path=outputs/ablation_central_no_osr/best_model.pt evaluation.checkpoint_path=outputs/ablation_central_no_osr/best_model.pt federated.num_clients=10 dataset.preprocessing.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
```

Create `outputs/ablation_suite/ablation_metrics.csv`:

```text
configuration,macro_f1
Base Model (Centralized, No OSR),<test/macro_f1 from eval_ablation_central_no_osr>
Base + FL (FedAvg),<test/macro_f1 from ablation_fedavg_no_osr>
Base + OSR (Centralized),<test/macro_f1 from eval_ablation_central_osr>
Proposed (FL + OSR),<test/macro_f1 from ablation_full>
```

Render:

```powershell
poetry run python scripts/plot.py run_dir=outputs/ablation_suite
```

## 12. Required Plot Data Contract

The renderer supports all 14 Q1 figures. These are the exact source files it
uses:

| # | Plot | Required file(s) |
|---:|---|---|
| 1 | Scalability: Nodes vs Final Accuracy | `scalability.csv` |
| 2 | Non-IID Data Distribution | `client_class_distribution.csv` |
| 3 | Convergence and Variance: Mild Non-IID | `comparison_metrics.csv` |
| 4 | Convergence and Variance: Hard Non-IID | `comparison_metrics.csv` |
| 5 | Known vs Unknown EVT Score Distribution | `open_set_scores.csv`, optional `open_set_metrics.json` |
| 6 | Openness vs AUROC Performance | `openness_metrics.csv` |
| 7 | ROC Curve for Unknown Zero-Day Attacks | `open_set_roc_curve.csv` |
| 8 | Cross-Dataset Generalization | `cross_dataset_metrics.csv` |
| 9 | Before OSR Confusion Matrix | `before_osr_confusion_matrix.csv` |
| 10 | After OSR Confusion Matrix | `after_osr_confusion_matrix.csv` |
| 11 | Seed Robustness Boxplot | `seed_robustness.csv` |
| 12 | t-SNE/UMAP Latent Space Separation | `latent_embeddings.csv` |
| 13 | Communication Efficiency | `communication_metrics.csv` |
| 14 | Architectural Ablation | `ablation_metrics.csv` |

Suite-level CSVs currently requiring manual or external aggregation/export:

```text
scalability.csv
openness_metrics.csv
cross_dataset_metrics.csv
seed_robustness.csv
latent_embeddings.csv
communication_metrics.csv
ablation_metrics.csv
```

Do not use a plot with "Missing data" text as final manuscript evidence.

## 13. Result Tables to Export for the Paper

Create these manuscript tables from saved artifacts only:

| Table | Source artifacts | Notes |
|---|---|---|
| Dataset summary | `preprocess_metadata.json`, `partition_manifest.jsonl`, `client_class_distribution.csv` | Include raw counts, known/unknown split, client partition settings. |
| Main B-NAT method comparison | `run_comparison.csv`, `evaluation_metrics.json` from FMRL_LA, FedAvg, FedProx | Report mean, std, 95% CI across paired seeds. |
| Open-set detection | `open_set_metrics.json`, `open_set_scores.csv` | Report AUROC, AUPRC, FPR@95%TPR, unknown F1. |
| Scalability | `scalability.csv` | Report N vs final accuracy and degradation from N=3. |
| Cross-dataset generalization | `cross_dataset_metrics.csv` | Blocked until external dataset configs are present. |
| Ablation | `ablation_metrics.csv` | Separate FL and OSR contributions. |
| Reproducibility | `metadata.json`, `resolved_config.yaml`, `run.log` | Include commit, seed, device, dataset, and method. |

The current local demo run `outputs/validation_minimal` is useful for format and
pipeline validation, but it is not final Q1 evidence because it used one logical
federated round.

## 14. Statistical Analysis

All final comparisons must be paired by seed and partition. For each seed, use
identical B-NAT splits and client manifests across FMRL_LA, FedAvg, and FedProx.

Required reporting:

- Mean, standard deviation, median, IQR, and 95% confidence interval.
- Paired Wilcoxon signed-rank test when normality is not supported; otherwise a paired t-test.
- Holm-Bonferroni correction across methods, datasets, and primary metrics.
- Effect sizes such as paired Cohen's d or rank-biserial correlation.
- Bootstrap confidence intervals for AUROC/AUPRC/FPR@95%TPR when per-sample scores are available.
- Practical significance thresholds declared before final analysis, for example macro F1 gain >= 0.02 or AUROC gain >= 0.03.

## 15. Final Acceptance Checklist

For every manuscript result, verify:

- `resolved_config.yaml` exists.
- `metadata.json` records seed, device, dataset, model, method, command context, and git commit when available.
- `run.log` and `debug.log` exist.
- `metrics.csv` and `metrics.jsonl` exist.
- `evaluation_metrics.json`, `test_metrics.json`, and `open_set_metrics.json` exist for evaluated open-set runs.
- `open_set_scores.csv` contains `y_true`, `raw_pred`, `y_pred`, `unknown_score`, and `is_unknown`.
- `open_set_roc_curve.csv` and `open_set_pr_curve.csv` exist for open-set reporting.
- `before_osr_confusion_matrix.csv` and `after_osr_confusion_matrix.csv` are used for Plots 9 and 10.
- `plots/plot_manifest.json` exists after plotting.
- Every required plot has a real source file; missing-data placeholders are excluded from final evidence.
- The paper draft cites the run directory and source metric file for each reported number.
