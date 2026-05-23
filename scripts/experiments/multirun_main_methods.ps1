$ErrorActionPreference = "Stop"

& poetry run python run.py --multirun experiment=exp3 "+method=fmrl_la,fedavg,fedprox" seed=42 dataset.preprocessing.alpha=0.1

