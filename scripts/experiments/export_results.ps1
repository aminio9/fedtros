param(
    [Parameter(Mandatory = $true)]
    [string[]]$RunDirs,
    [string]$RunId = "suite_export"
)

$ErrorActionPreference = "Stop"

$HydraRuns = "[" + ($RunDirs -join ",") + "]"
& poetry run python run.py "experiment.pipeline=export" "runs=$HydraRuns" "tracking.run_id=$RunId"
& poetry run python run.py "experiment.pipeline=plot" "run_dir=outputs/$RunId" "tracking.run_id=$RunId"

