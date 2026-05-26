$ErrorActionPreference = "Stop"

function Invoke-HydraRun {
    param([string[]]$HydraArgs)
    & poetry run python run.py @HydraArgs
}

foreach ($clients in 3, 10, 20, 50, 100) {
    foreach ($rounds in 50, 100, 200) {
        Invoke-HydraRun @("experiment=efficiency", "+method=fmrl_la", "seed=42", "federated.num_clients=$clients", "federated.num_rounds=$rounds", "tracking.run_id=e6_fmrl_la_clients${clients}_rounds${rounds}_seed42")
        Invoke-HydraRun @("experiment=efficiency", "+method=fedavg", "seed=42", "federated.num_clients=$clients", "federated.num_rounds=$rounds", "tracking.run_id=e6_fedavg_clients${clients}_rounds${rounds}_seed42")
    }
}
