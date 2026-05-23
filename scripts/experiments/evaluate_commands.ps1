param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$Experiment = "exp4",
    [string]$RunId = "eval_from_checkpoint"
)

$ErrorActionPreference = "Stop"

& poetry run python run.py "experiment=$Experiment" "experiment.pipeline=evaluate" "checkpoint.path=$Checkpoint" "evaluation.checkpoint_path=$Checkpoint" "tracking.run_id=$RunId"
& poetry run python run.py "experiment=$Experiment" "experiment.pipeline=plot" "run_dir=outputs/$RunId" "tracking.run_id=$RunId"

