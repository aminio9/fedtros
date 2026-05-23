# All Suite Runbook

## Objective

Launch one representative run for each experiment block from a single Hydra config.

## Hydra Config Used

`experiment=all`

## Override Examples

```powershell
python run.py experiment=all
python run.py --multirun experiment=all
python run.py experiment=all seed=42
```

## Execution Commands

```powershell
python run.py experiment=all
python run.py --multirun experiment=all
python scripts/experiments/run_full_suite.ps1
```

## Expected Outputs

- one run directory per child command
- `suite_all_seed<seed>/` if the suite launcher is used directly

## Checkpoints

- child run checkpoints under each child run directory

## Logs

- one `run.log` and `debug.log` per child run
- suite-level tracker output for the launcher run

## Metrics

- the metrics from each child run

## Artifacts

- `resolved_config.yaml` for the launcher and every child run
- `suite_artifacts_manifest.json` when export runs are staged

## Validation

- Confirm the child commands are listed in `experiment.suite_commands`.
- Confirm each child run uses the intended method overlay.

## Troubleshooting

- If a child run recurses into the suite launcher, check the command list.
- If the launcher exits early, inspect the first failing child command.

