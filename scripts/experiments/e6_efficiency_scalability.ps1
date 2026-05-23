$ErrorActionPreference = "Stop"

function Invoke-HydraRun {
    param([string[]]$HydraArgs)
    & poetry run python run.py @HydraArgs
}

foreach ($clients in 3, 10, 20, 50, 100) {
    Invoke-HydraRun @("experiment=efficiency", "+method=fmrl_la", "seed=42", "federated.num_clients=$clients")
    Invoke-HydraRun @("experiment=efficiency", "+method=fedavg", "seed=42", "federated.num_clients=$clients")
}

