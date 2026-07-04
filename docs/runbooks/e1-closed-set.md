# E1 Closed-Set Runbook

## Objective

Verify that the unified model preserves performance across the full B-NAT label set, with no unknown labels held out.

## Hydra Config Used

`experiment=exp1` with optional method overlays:

- `+method=fmrl_ava`
- `+method=fedavg`
- `+method=fedprox`
- `+method=centralized_no_osr` (commented reference only in the shell script)
- `dataset.known_labels=${dataset.source_labels}`

Default run-local paths:

- `dataset.preprocessing.output_dir=${tracking.run_dir}/processed`
- `dataset.preprocessing.iid=true`
- `tracking.run_id=e1_${experiment.method}_iid_seed${seed}`

`tracking.*` is derived from the root `output.*` config values rather than a separate config group.

## Override Examples

```bash
python run.py experiment=exp1 +method=fmrl_ava seed=42
python run.py experiment=exp1 +method=fedavg seed=42
python run.py experiment=exp1 +method=fedprox seed=42
# python run.py experiment=exp1 +method=centralized_no_osr seed=42
```

## Execution Commands

```bash
python run.py experiment=exp1 +method=fmrl_ava seed=42
python run.py experiment=exp1 +method=fedavg seed=42
python run.py experiment=exp1 +method=fedprox seed=42
# python run.py experiment=exp1 +method=centralized_no_osr seed=42
bash scripts/experiments/e1_closed_set.sh
```

## Expected Outputs

- `evaluation_metrics.json`
- `test_metrics.json`
- `test_confusion_matrix.csv`
- `federated_history.csv`
- `communication_metrics.csv`

## Checkpoints

- `best_model.pt`
- `latest_checkpoint.pt`
- `final_model.pt`

## Logs

- `run.log`
- `debug.log`
- `metrics.jsonl`
- `metrics.csv`
- `metadata.json`

## Metrics

- `test/accuracy`
- `test/balanced_accuracy`
- `test/macro_f1`
- `test/weighted_f1`

## Artifacts

- `resolved_config.yaml`
- `processed/`
- `plots/`
- `plot_manifest.json`

## Validation

- Confirm `resolved_config.yaml` records the chosen method overlay.
- Confirm `resolved_config.yaml` records `dataset.preprocessing.iid: true`.
- Confirm `evaluation_metrics.json` contains closed-set metrics.
- Confirm the run-local `processed/` directory exists.

## Troubleshooting

- If preprocessing fails, check `dataset.preprocessing.raw_file`.
- If the run reuses the wrong output directory, inspect `tracking.run_id`.
- If the checkpoint is missing, check `checkpointing.save_best` and
  `checkpointing.save_latest`.

## 2026-07 selective latent aggregation fix

The E1 IID 3-client log showed a different failure from the non-IID alpha=0.1 run. Local client models reached strong pre-aggregation metrics, but the global post-aggregation model repeatedly collapsed FoT after averaging. This points to full CVAE-DQN parameter averaging, especially prior/recognition/generator latent modules, rather than client selection or proximal regularization.

FMRL-AVA-GLOW-TWA now supports module-wise delta scaling:

```yaml
federated:
  strategy:
    module_delta_scales:
      prior_net: 0.25
      recognition_net: 0.10
      value_net_main: 1.0
      generation_net: 0.0
```

The global delta is still computed from all selected clients, but before applying the update the server scales each module group. The Main Q/classification network aggregates normally. Prior and recognition move slowly, acting like an EMA-style latent update. The generator is frozen by default in closed-set GLOW configs because it is not needed when generator training is disabled. This is not class-aware aggregation and does not privilege any label or client; it only prevents incompatible local latent spaces from being fully averaged.

For a pure FedAvg-equivalent diagnosis, `fmrl_ava_glow_stable.yaml` keeps all module scales at `1.0`. For the repaired method, use `fmrl_ava_glow_twa.yaml`.

Suggested IID 3-client rerun:

```bash
python run.py experiment=exp1 +method=fmrl_ava_glow_twa seed=42 \
  federated.num_clients=3 \
  dataset.preprocessing.num_clients=3 \
  dataset.preprocessing.iid=true \
  dataset.preprocessing.alpha=1.0 \
  federated.strategy.local_proximal_mu=0.0 \
  tracking.run_id=e1_FMRL_AVA_GLOW_TWA_iid_c3_selective_seed42
```

If FoT still collapses after aggregation, sweep:

```bash
python run.py experiment=exp1 +method=fmrl_ava_glow_twa seed=42 federated.strategy.module_delta_scales.prior_net=0.10 federated.strategy.module_delta_scales.recognition_net=0.00
python run.py experiment=exp1 +method=fmrl_ava_glow_twa seed=42 federated.strategy.module_delta_scales.prior_net=0.50 federated.strategy.module_delta_scales.recognition_net=0.25
```
