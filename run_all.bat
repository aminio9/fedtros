@echo off
poetry run python scripts/run_study.py E1-IID-CS --stage paper_final --wandb-mode disabled
poetry run python scripts/run_study.py E2-IID-OSR --stage paper_final --wandb-mode disabled
poetry run python scripts/run_study.py E3-NIID-CS --stage paper_final --wandb-mode disabled
poetry run python scripts/run_study.py E4-NIID-FOSR --stage paper_final --wandb-mode disabled
poetry run python scripts/run_study.py E5-NIID-FOSR-LT --stage paper_final --wandb-mode disabled
poetry run python scripts/run_study.py E6-NIID-FOSR-SLT --stage paper_final --wandb-mode disabled
poetry run python scripts/run_study.py E7-CROSS --stage paper_final --wandb-mode disabled
poetry run python scripts/run_study.py E8-LOAO --stage paper_final --wandb-mode disabled
poetry run python scripts/run_study.py A1-A5 --stage ablation --wandb-mode disabled
poetry run python scripts/build_q1_results.py --outputs-dir outputs --target paper_results --stage paper_final ablation
