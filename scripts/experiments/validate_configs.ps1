$ErrorActionPreference = "Stop"

$Experiments = @("exp1", "exp2", "exp3", "exp4", "ablation", "efficiency", "validation", "all")
foreach ($Experiment in $Experiments) {
    & poetry run python run.py "experiment=$Experiment" --cfg job --resolve | Out-Null
}

& poetry run python run.py experiment=exp3 "+method=fedavg" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp3 "+method=fedprox" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=ablation runtime=gpu --cfg job --resolve | Out-Null

