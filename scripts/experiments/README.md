# Experiment Execution Scripts

Run from the repository root with Bash.

- `validate_configs.sh`: resolves all experiment configs without running training.
- `run_validation_tiny.sh`: runs the tiny end-to-end validation path and then plots it.
- `e1_closed_set.sh` through `e8_labelwise_open_set.sh`: experiment command files.
- `run_full_suite.sh`: sequential batch runner for E1-E8.
- `multirun_main_methods.sh`: legacy shell wrapper for the pre-study CLI; retained for migration reference only. Use `scripts/run_study.py E3-NIID-CS` for the canonical FedAvg-Student, FedProx-Student, and FedTROS-PR matrix.
- `e3_federated_noniid.sh` and `e4_combined_open_set_noniid.sh`: sweep Dirichlet alpha values `0.1`, `0.5`, `1.0`, and `10.0`.
- `multirun_alpha_seed_sensitivity.sh`: Hydra multirun over seeds and alpha values for E4.
- `e5_multi_dataset_open_set_noniid.sh`: external benchmark validation scaffold for B-TAT, ToN-IoT, and CIC-IDS2017; it skips datasets until their final mappings are supplied.
- `e6_ablation.sh`: full ablation matrix, including no-EVT, no-generator, no-selection, FedAvg, and FedProx variants.
- `e7_efficiency_scalability.sh`: client-count and round-budget sweep for FedAvg and FMRL-AVA.
- `e8_labelwise_open_set.sh`: label-wise open-set stress test used for latent-space visual proof.
- `resume_commands.sh`: resume federated training from a checkpoint; centralized resume commands are commented references.
- `evaluate_commands.sh`: regenerate evaluation and plots from a checkpoint.
- `export_results.sh`: export suite-level CSVs and plots from completed run directories.

Centralized `centralized_no_osr` and `centralized_osr` experiment runs are kept as commented reference commands and are skipped by the runnable experiment scripts.
