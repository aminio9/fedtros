param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$RunId = "resumed_fmrl_seed42"
)

$ErrorActionPreference = "Stop"

& poetry run python run.py experiment=exp3 "+method=fmrl_la" "federated.resume_from=$Checkpoint" "tracking.run_id=$RunId"
& poetry run python run.py experiment=ablation "+method=centralized_osr" "training.resume_from=$Checkpoint" "tracking.run_id=${RunId}_central"

