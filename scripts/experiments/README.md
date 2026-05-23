# Experiment Execution Scripts

Run from the repository root with PowerShell.

- `validate_configs.ps1`: resolves all experiment configs without running training.
- `run_validation_tiny.ps1`: runs the tiny end-to-end validation path and then plots it.
- `e1_closed_set.ps1` through `e6_efficiency_scalability.ps1`: experiment command files.
- `run_full_suite.ps1`: sequential batch runner for E1-E6.
- `multirun_main_methods.ps1`: Hydra multirun over FedAvg, FedProx, and FMRL-LA for E3.
- `multirun_alpha_seed_sensitivity.ps1`: Hydra multirun over seeds and alpha values for E4.
- `resume_commands.ps1`: resume federated or centralized training from a checkpoint.
- `evaluate_commands.ps1`: regenerate evaluation and plots from a checkpoint.
- `export_results.ps1`: export suite-level CSVs and plots from completed run directories.
