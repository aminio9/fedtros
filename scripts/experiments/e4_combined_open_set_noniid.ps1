$ErrorActionPreference = "Stop"

function Invoke-HydraRun {
    param([string[]]$HydraArgs)
    & poetry run python run.py @HydraArgs
}

foreach ($alpha in "0.1", "10.0") {
    Invoke-HydraRun @("experiment=exp4", "+method=fmrl_la", "seed=42", "dataset.preprocessing.alpha=$alpha")
    Invoke-HydraRun @("experiment=exp4", "+method=fmrl_la", "seed=42", "dataset.preprocessing.alpha=$alpha", "open_set.evt.enabled=false", "experiment.method=ClosedSet_No_EVT", "tracking.run_id=e4_no_evt_alpha${alpha}_seed42")
    Invoke-HydraRun @("experiment=exp4", "+method=fedavg", "seed=42", "dataset.preprocessing.alpha=$alpha")
    Invoke-HydraRun @("experiment=exp4", "+method=fedprox", "seed=42", "dataset.preprocessing.alpha=$alpha")
    Invoke-HydraRun @("experiment=exp4", "+method=centralized_no_osr", "seed=42", "dataset.preprocessing.alpha=$alpha", "tracking.run_id=e4_central_no_osr_alpha${alpha}_seed42")
    Invoke-HydraRun @("experiment=exp4", "+method=centralized_osr", "seed=42", "dataset.preprocessing.alpha=$alpha", "tracking.run_id=e4_central_osr_alpha${alpha}_seed42")
}
