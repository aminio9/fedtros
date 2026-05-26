$ErrorActionPreference = "Stop"

function Invoke-HydraRun {
    param([string[]]$HydraArgs)
    & poetry run python run.py @HydraArgs
}

Invoke-HydraRun @("experiment=ablation", "+method=fmrl_la", "seed=42", "tracking.run_id=ablation_full_seed42")
Invoke-HydraRun @("experiment=ablation", "+method=fmrl_la", "seed=42", "open_set.evt.enabled=false", "experiment.method=No_EVT", "tracking.run_id=ablation_no_evt_seed42")
Invoke-HydraRun @("experiment=ablation", "+method=fmrl_la", "seed=42", "training.generator.enabled=false", "experiment.method=No_Generator", "tracking.run_id=ablation_no_generator_seed42")
Invoke-HydraRun @("experiment=ablation", "+method=fmrl_la", "seed=42", "federated.strategy.utility_threshold=-1.0", "experiment.method=No_Selection", "tracking.run_id=ablation_no_selection_seed42")
Invoke-HydraRun @("experiment=ablation", "+method=fedavg", "seed=42", "open_set.evt.enabled=false", "tracking.run_id=ablation_fedavg_no_osr_seed42")
Invoke-HydraRun @("experiment=ablation", "+method=fedprox", "seed=42", "open_set.evt.enabled=false", "tracking.run_id=ablation_fedprox_no_osr_seed42")
Invoke-HydraRun @("experiment=ablation", "+method=centralized_osr", "seed=42", "tracking.run_id=ablation_central_osr_seed42")
Invoke-HydraRun @("experiment=ablation", "+method=centralized_no_osr", "seed=42", "tracking.run_id=ablation_central_no_osr_seed42")
