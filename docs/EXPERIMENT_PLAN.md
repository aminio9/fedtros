# Experiment Plan

This plan reconciles `D:\Research\Experimentplan\ExperimentPlan.docx`, `D:\Research\Experimentplan\testplot.py`, and the executable pipeline in this repository. Final figures must be generated from saved run outputs, not from the synthetic arrays in the original plotting template.

## Research Objective

Evaluate the proposed FMRL-LA federated intrusion-detection framework for:

- closed-set classification of known Blockchain traffic classes,
- open-set rejection of unknown or zero-day attacks,
- robustness to non-IID client partitions and random seeds,
- scalability with client count,
- communication efficiency and cross-dataset generalization.

## Core Run Configuration

- Base Hydra config: `src/configs/config_fl.yaml`.
- Primary dataset config: `dataset=bnat`.
- Primary experiment config: `experiment=baseline`.
- Primary model config: `model=openset_qchain`.
- Known labels: `Normal`, `BP`, `DoS`, `MitM`.
- Unknown labels: labels outside `known_labels`, encoded as `-1` for open-set evaluation.
- Default strategy: `federated.strategy.name=fmrl_la` and `experiment.method=FMRL_LA`.
- Baselines: `federated.strategy.name=fedavg experiment.method=FedAvg` and `federated.strategy.name=fedprox experiment.method=FedProx`.
- Main client counts: `3`, `10`, `20`, `50`, `100`.
- Main non-IID alphas: `0.1`, `0.5`, `1.0`, `10`, plus IID.
- Main seeds: at least `42 43 44 45 46`; use 10 or more for final robustness plots.
- Plot output: one file per experiment, `outputs/<run_id>/plots/NN_<plot_id>.png` and `.pdf`, plus `plot_manifest.json`.

Every run writes `config.yaml`, `resolved_config.yaml`, `metadata.json`, logs, metrics, checkpoints, evaluation CSV/JSON files, and plot source files under `outputs/<run_id>/`.

## Plot Inventory

| # | Plot | Source data file |
|---|------|------------------|
| 1 | Scalability: nodes vs final accuracy | `scalability.csv` |
| 2 | Non-IID client data distribution | `client_class_distribution.csv` |
| 3 | Mild non-IID convergence and variance | `comparison_metrics.csv` or `federated_history.csv` |
| 4 | Hard non-IID convergence and variance | `comparison_metrics.csv` or `federated_history.csv` |
| 5 | Known vs unknown score distributions | `open_set_scores.csv` |
| 6 | Openness vs AUROC | `openness_metrics.csv` |
| 7 | Unknown-detection ROC | `open_set_roc_curve.csv` |
| 8 | Cross-dataset generalization | `cross_dataset_metrics.csv` |
| 9 | Before-OSR confusion matrix | `before_osr_confusion_matrix.csv` |
| 10 | After-OSR confusion matrix | `after_osr_confusion_matrix.csv` |
| 11 | Seed robustness boxplot | `seed_robustness.csv` |
| 12 | t-SNE/UMAP latent separation | `latent_embeddings.csv` |
| 13 | Communication efficiency | `communication_metrics.csv` |
| 14 | Architectural ablation | `ablation_metrics.csv` |

Plots 9 and 10 must be built from open-set predictions. Plot 10 must not use `test_confusion_matrix.csv`, because that file is the closed-set test matrix and does not include unknown rejection.

## Single-Run Reproduction

Run preprocessing, federated training, open-set evaluation, and plotting:

```bash
poetry install
poetry run python scripts/reproduce_experiment.py seed=42 federated.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
poetry run python scripts/plot.py run_dir=outputs/<run_id>
```

If preprocessing and training are run separately:

```bash
poetry run python scripts/preprocess.py seed=42 federated.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
poetry run python scripts/federated_train.py tracking.run_id=train_fmrl_alpha01_seed42 seed=42 federated.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
poetry run python scripts/evaluate.py tracking.run_id=eval_fmrl_alpha01_seed42 checkpoint.path=outputs/train_fmrl_alpha01_seed42/best_model.pt
poetry run python scripts/plot.py run_dir=outputs/eval_fmrl_alpha01_seed42
```

## Experiment Commands

Use explicit `tracking.run_id` values when running sweeps so comparison commands are reproducible.

### Plot 1: Scalability

Run one full pipeline per client count:

```bash
poetry run python scripts/reproduce_experiment.py tracking.run_id=scalability_n3_seed42 seed=42 federated.num_clients=3 dataset.preprocessing.num_clients=3
poetry run python scripts/reproduce_experiment.py tracking.run_id=scalability_n10_seed42 seed=42 federated.num_clients=10 dataset.preprocessing.num_clients=10
poetry run python scripts/reproduce_experiment.py tracking.run_id=scalability_n20_seed42 seed=42 federated.num_clients=20 dataset.preprocessing.num_clients=20
poetry run python scripts/reproduce_experiment.py tracking.run_id=scalability_n50_seed42 seed=42 federated.num_clients=50 dataset.preprocessing.num_clients=50
poetry run python scripts/reproduce_experiment.py tracking.run_id=scalability_n100_seed42 seed=42 federated.num_clients=100 dataset.preprocessing.num_clients=100
```

Aggregate final accuracy into `scalability.csv` with columns `num_clients,final_accuracy`, place it in a comparison run directory, then run `scripts/plot.py`.

### Plot 2: Non-IID Distribution

Generate the partition and client distribution files:

```bash
poetry run python scripts/preprocess.py tracking.run_id=noniid_alpha01_seed42 seed=42 federated.num_clients=10 dataset.preprocessing.num_clients=10 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
poetry run python scripts/plot.py run_dir=outputs/noniid_alpha01_seed42
```

### Plots 3 and 4: Convergence

Run each method under mild and hard non-IID. Example for hard non-IID:

```bash
poetry run python scripts/reproduce_experiment.py tracking.run_id=fmrl_alpha01_seed42 seed=42 federated.strategy.name=fmrl_la experiment.method=FMRL_LA dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
poetry run python scripts/reproduce_experiment.py tracking.run_id=fedavg_alpha01_seed42 seed=42 federated.strategy.name=fedavg experiment.method=FedAvg dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
poetry run python scripts/reproduce_experiment.py tracking.run_id=fedprox_alpha01_seed42 seed=42 federated.strategy.name=fedprox experiment.method=FedProx dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
poetry run python scripts/compare_runs.py tracking.run_id=compare_alpha01_seed42 runs='[outputs/fmrl_alpha01_seed42,outputs/fedavg_alpha01_seed42,outputs/fedprox_alpha01_seed42]'
poetry run python scripts/plot.py run_dir=outputs/compare_alpha01_seed42
```

Repeat with `dataset.preprocessing.alpha=10` and `tracking.run_id=compare_alpha10_seed42`.

### Plots 5, 7, 9, and 10: Open-Set Evaluation

These are produced by `scripts/evaluate.py` from a trained checkpoint:

```bash
poetry run python scripts/evaluate.py tracking.run_id=eval_fmrl_alpha01_seed42 checkpoint.path=outputs/fmrl_alpha01_seed42/best_model.pt
poetry run python scripts/plot.py run_dir=outputs/eval_fmrl_alpha01_seed42
```

Expected source files: `open_set_scores.csv`, `open_set_roc_curve.csv`, `before_osr_confusion_matrix.csv`, and `after_osr_confusion_matrix.csv`.

### Plot 6: Openness vs AUROC

Run the open-set evaluation for each registered openness setting or unknown-label holdout. Save one row per setting in `openness_metrics.csv`:

```text
method,openness,auroc
FMRL_LA,0.1,0.95
FedAvg,0.1,0.86
```

Then render:

```bash
poetry run python scripts/plot.py run_dir=outputs/<openness_suite_run>
```

### Plot 8: Cross-Dataset Generalization

For each external dataset, set the dataset config/raw path, run preprocessing with the same known/unknown protocol, evaluate the frozen method, and write `cross_dataset_metrics.csv` with `dataset,metric,metric_value`.

```bash
poetry run python scripts/preprocess.py tracking.run_id=btat_preprocess seed=42 dataset.name=B-TAT dataset.preprocessing.raw_file=data/raw/B-TAT.csv
poetry run python scripts/evaluate.py tracking.run_id=btat_eval checkpoint.path=outputs/fmrl_alpha01_seed42/best_model.pt dataset.name=B-TAT
poetry run python scripts/plot.py run_dir=outputs/<cross_dataset_suite_run>
```

### Plot 11: Seed Robustness

Repeat the main run for each seed and alpha, then write `seed_robustness.csv` with `seed,heterogeneity,accuracy`.

```bash
poetry run python scripts/reproduce_experiment.py tracking.run_id=fmrl_alpha01_seed43 seed=43 dataset.preprocessing.alpha=0.1 dataset.preprocessing.iid=false
poetry run python scripts/reproduce_experiment.py tracking.run_id=fmrl_alpha10_seed43 seed=43 dataset.preprocessing.alpha=10 dataset.preprocessing.iid=false
poetry run python scripts/plot.py run_dir=outputs/<seed_robustness_suite_run>
```

### Plot 12: Latent Separation

Export model embeddings to `latent_embeddings.csv` with `x,y,label`. If the embedding exporter is external, keep its command and seed in the suite run log before rendering:

```bash
poetry run python scripts/plot.py run_dir=outputs/<latent_suite_run>
```

### Plot 13: Communication Efficiency

Federated simulation writes `federated_history.csv`. For the systems figure, create `communication_metrics.csv` with `method,cumulative_mb,accuracy` from transmitted model bytes and round-level accuracy:

```bash
poetry run python scripts/plot.py run_dir=outputs/<communication_suite_run>
```

### Plot 14: Architectural Ablation

Run the four ablation variants and save final macro F1 in `ablation_metrics.csv` with `configuration,macro_f1`:

```bash
poetry run python scripts/reproduce_experiment.py tracking.run_id=ablation_full seed=42 federated.strategy.name=fmrl_la open_set.evt.enabled=true experiment.method=FMRL_LA
poetry run python scripts/reproduce_experiment.py tracking.run_id=ablation_fedavg_no_osr seed=42 federated.strategy.name=fedavg open_set.evt.enabled=false experiment.method=FedAvg
poetry run python scripts/train.py tracking.run_id=ablation_central_osr seed=42 open_set.evt.enabled=true experiment.method=Centralized_OSR
poetry run python scripts/train.py tracking.run_id=ablation_central_no_osr seed=42 open_set.evt.enabled=false experiment.method=Centralized_No_OSR
poetry run python scripts/plot.py run_dir=outputs/<ablation_suite_run>
```

## Acceptance Checks

- `resolved_config.yaml` exists for every run.
- `metadata.json` records method, dataset, seed, device, and git commit when available.
- `open_set_scores.csv` includes both `raw_pred` and final `y_pred`.
- Plots 9 and 10 read `before_osr_confusion_matrix.csv` and `after_osr_confusion_matrix.csv`.
- `scripts/plot.py` produces separate numbered figures, not `complete_Q1_dashboard.png`.
- Missing source data results in a missing-data figure and warning, never synthetic evidence.
