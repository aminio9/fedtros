# E6 Ablation Runbook

## Objective

Isolate the contribution of EVT, generator training, client selection, and federated strategy.

## Hydra Config Used

`experiment=exp6` with overlayed toggles such as:

- `training.generator.enabled=false`
- `open_set.evt.enabled=false`
- `federated.strategy.utility_threshold=-1.0`
- `federated.strategy.alignment_strength=0.0`
- `federated.strategy.critic_blend=0.0`
- `+method=fedavg`
- `+method=fedprox`
- `+method=centralized_osr`
- `+method=centralized_no_osr`

## Override Examples

```bash
python run.py experiment=exp6 +method=fmrl_ava seed=42
python run.py experiment=exp6 +method=fmrl_ava training.generator.enabled=false seed=42
python run.py experiment=exp6 +method=fedavg open_set.evt.enabled=false seed=42
```

## Execution Commands

```bash
python run.py experiment=exp6 +method=fmrl_ava seed=42
python run.py experiment=exp6 +method=fmrl_ava open_set.evt.enabled=false experiment.method=No_EVT tracking.run_id=ablation_no_evt_seed42 seed=42
python run.py experiment=exp6 +method=fmrl_ava training.generator.enabled=false experiment.method=No_Generator tracking.run_id=ablation_no_generator_seed42 seed=42
python run.py experiment=exp6 +method=fmrl_ava federated.strategy.utility_threshold=-1.0 experiment.method=No_Selection tracking.run_id=ablation_no_selection_seed42 seed=42
python run.py experiment=exp6 +method=fmrl_ava federated.strategy.alignment_strength=0.0 experiment.method=No_Vector_Alignment tracking.run_id=ablation_no_alignment_seed42 seed=42
python run.py experiment=exp6 +method=fmrl_ava federated.strategy.critic_blend=0.0 experiment.method=No_Critic_Residual tracking.run_id=ablation_no_critic_seed42 seed=42
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
- Confirm vector-alignment ablations keep the same selected clients, seed, and
  validation split unless the changed aggregation path later changes utilities.
- Confirm centralized baselines use `method/centralized_*`.
- Confirm the reported comparison table reads saved artifacts only.
