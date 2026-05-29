# E6 Ablation Runbook

## Objective

Isolate the contribution of EVT, generator training, client selection, and federated strategy.

## Hydra Config Used

`experiment=exp6` with overlayed toggles such as:

- `training.generator.enabled=false`
- `open_set.evt.enabled=false`
- `federated.strategy.utility_threshold=-1.0`
- `+method=fedavg`
- `+method=fedprox`
- `+method=centralized_osr`
- `+method=centralized_no_osr`

## Override Examples

```bash
python run.py experiment=exp6 +method=fmrl_la seed=42
python run.py experiment=exp6 +method=fmrl_la training.generator.enabled=false seed=42
python run.py experiment=exp6 +method=fedavg open_set.evt.enabled=false seed=42
```

## Execution Commands

```bash
python run.py experiment=exp6 +method=fmrl_la seed=42
python run.py experiment=exp6 +method=fmrl_la open_set.evt.enabled=false experiment.method=No_EVT tracking.run_id=ablation_no_evt_seed42 seed=42
python run.py experiment=exp6 +method=fmrl_la training.generator.enabled=false experiment.method=No_Generator tracking.run_id=ablation_no_generator_seed42 seed=42
python run.py experiment=exp6 +method=fmrl_la federated.strategy.utility_threshold=-1.0 experiment.method=No_Selection tracking.run_id=ablation_no_selection_seed42 seed=42
python run.py experiment=exp6 +method=fedavg open_set.evt.enabled=false tracking.run_id=ablation_fedavg_no_osr_seed42 seed=42
python run.py experiment=exp6 +method=fedprox open_set.evt.enabled=false tracking.run_id=ablation_fedprox_no_osr_seed42 seed=42
python run.py experiment=exp6 +method=centralized_osr tracking.run_id=ablation_central_osr_seed42 seed=42
python run.py experiment=exp6 +method=centralized_no_osr tracking.run_id=ablation_central_no_osr_seed42 seed=42
bash scripts/experiments/e6_ablation.sh
```

## Expected Outputs

- `evaluation_metrics.json`
- `open_set_metrics.json`
- `ablation_metrics.csv` from the suite export
- `communication_metrics.csv`

## Validation

- Confirm each ablation changes only one mechanism at a time.
- Confirm centralized baselines use `method/centralized_*`.
- Confirm the reported comparison table reads saved artifacts only.
