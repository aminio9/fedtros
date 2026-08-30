@echo off
poetry run python scripts/run_study.py E1-IID-CS --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py E2-IID-OSR --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py E3-NIID-CS --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py E4-NIID-FOSR --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py E5-NIID-FOSR-LT --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py E6-NIID-FOSR-SLT --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py E7-CROSS --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py E8-LOAO --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py A1-ANCHOR --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py A2-GAMMA --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py A3-CENTER --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py A4-DISTANCE --stage smoke --wandb-mode disabled
poetry run python scripts/run_study.py A5-CONFORMAL --stage smoke --wandb-mode disabled
poetry run python scripts/build_q1_results.py --outputs-dir outputs --target paper_results --stage smoke
