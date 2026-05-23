$ErrorActionPreference = "Stop"

function Invoke-HydraRun {
    param([string[]]$HydraArgs)
    & poetry run python run.py @HydraArgs
}

Invoke-HydraRun @("experiment=exp1", "+method=fmrl_la", "seed=42")
Invoke-HydraRun @("experiment=exp1", "+method=fedavg", "seed=42")
Invoke-HydraRun @("experiment=exp1", "+method=fedprox", "seed=42")
Invoke-HydraRun @("experiment=exp1", "+method=centralized_no_osr", "seed=42")

