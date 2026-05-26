$ErrorActionPreference = "Stop"

$Experiments = @("baseline", "exp1", "exp2", "exp3", "exp4", "ablation", "efficiency", "validation", "all")
foreach ($Experiment in $Experiments) {
    & poetry run python run.py "experiment=$Experiment" --cfg job --resolve | Out-Null
}

& poetry run python run.py experiment=validation runtime=tiny --cfg job --resolve | Out-Null
& poetry run python run.py experiment=smoke runtime=tiny --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp1 "+method=fedprox" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp1 "+method=centralized_no_osr" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp2 "+method=fmrl_la" open_set.evt.enabled=false "experiment.method=ClosedSet_No_EVT" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp2 "+method=centralized_osr" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp2 "+method=centralized_no_osr" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=ablation "+method=fmrl_la" open_set.evt.enabled=false "experiment.method=No_EVT" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=ablation "+method=fmrl_la" training.generator.enabled=false "experiment.method=No_Generator" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=ablation "+method=fmrl_la" federated.strategy.utility_threshold=-1.0 "experiment.method=No_Selection" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=ablation "+method=fedavg" open_set.evt.enabled=false --cfg job --resolve | Out-Null
& poetry run python run.py experiment=ablation "+method=fedprox" open_set.evt.enabled=false --cfg job --resolve | Out-Null
& poetry run python run.py experiment=efficiency federated.num_clients=20 federated.num_rounds=50 --cfg job --resolve | Out-Null
& poetry run python run.py experiment=efficiency federated.num_clients=20 federated.num_rounds=100 --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp3 "+method=fedavg" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp3 "+method=fedprox" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp3 "+method=centralized_no_osr" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp4 "+method=fmrl_la" open_set.evt.enabled=false "experiment.method=ClosedSet_No_EVT" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp4 "+method=centralized_osr" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=exp4 "+method=centralized_no_osr" --cfg job --resolve | Out-Null
& poetry run python run.py experiment=ablation runtime=gpu --cfg job --resolve | Out-Null
& poetry run python run.py experiment=efficiency federated.num_clients=20 federated.num_rounds=200 --cfg job --resolve | Out-Null
