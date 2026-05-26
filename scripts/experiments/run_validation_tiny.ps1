$ErrorActionPreference = "Stop"

function Invoke-HydraRun {
    param([string[]]$HydraArgs)
    & poetry run python run.py @HydraArgs
}

$RunId = "tiny_validation_seed42"
Invoke-HydraRun @("experiment=validation", "runtime=tiny", "seed=42", "tracking.run_id=$RunId")
& poetry run python scripts/plot.py "run_dir=outputs/$RunId" "tracking.run_id=$RunId"
