$ErrorActionPreference = "Stop"

function Invoke-HydraRun {
    param([string[]]$HydraArgs)
    & poetry run python run.py @HydraArgs
}

Invoke-HydraRun @("experiment=exp2", "+method=fmrl_la", "seed=42")
Invoke-HydraRun @("experiment=exp2", "+method=fmrl_la", "seed=42", "open_set.evt.enabled=false", "experiment.method=ClosedSet_No_EVT", "tracking.run_id=e2_no_evt_seed42")
Invoke-HydraRun @("experiment=exp2", "+method=centralized_osr", "seed=42", "tracking.run_id=e2_central_osr_seed42")
Invoke-HydraRun @("experiment=exp2", "+method=centralized_no_osr", "seed=42", "tracking.run_id=e2_central_no_osr_seed42")

