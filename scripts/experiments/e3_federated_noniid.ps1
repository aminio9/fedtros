$ErrorActionPreference = "Stop"

function Invoke-HydraRun {
    param([string[]]$HydraArgs)
    & poetry run python run.py @HydraArgs
}

foreach ($alpha in "0.1", "10.0") {
    Invoke-HydraRun @("experiment=exp3", "+method=fmrl_la", "seed=42", "dataset.preprocessing.alpha=$alpha")
    Invoke-HydraRun @("experiment=exp3", "+method=fedavg", "seed=42", "dataset.preprocessing.alpha=$alpha")
    Invoke-HydraRun @("experiment=exp3", "+method=fedprox", "seed=42", "dataset.preprocessing.alpha=$alpha")
    Invoke-HydraRun @("experiment=exp3", "+method=centralized_no_osr", "seed=42", "dataset.preprocessing.alpha=$alpha", "tracking.run_id=e3_central_alpha${alpha}_seed42")
}
