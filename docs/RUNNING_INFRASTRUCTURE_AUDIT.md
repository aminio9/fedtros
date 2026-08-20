# Running & Logging Infrastructure Audit (Workstream B)

**Project:** FedTROS-PR (Federated Teacher-Regularized Open-Set Recognition with Prototype-Rank Rejection)
**Date:** 2026-08-19
**Auditor:** Workstream B Infrastructure Team
**Scope:** Execution scripts, shell harnesses, launchers, status trackers, logging mechanisms, and command pipelines.

---

## 1. Running Infrastructure Audit (Item B1)

| Path / Target | Type | Classification | Rationale & Action |
|---|---|---|---|
| `run.py` (Root) | Python Entrypoint | `REPLACE` | Monolithic script mixing pipeline dispatch, legacy contract assertions, and subprocess execution. Replace with thin canonical wrapper delegating to `src.infrastructure`. |
| `scripts/run.py` | Python Script | `ACTIVE` / `REPLACE` | Standardize as the single authoritative single-run entrypoint. Implements config resolution, run initialization, logging, tracking, execution, checkpointing, and manifests. |
| `scripts/run_study.py` | Python Script | `ACTIVE` (NEW) | New authoritative study runner replacing shell loops. Implements matrix expansion, seed policy, dry-run, only-missing, and stage controls. |
| `scripts/resume.py` | Python Script | `ACTIVE` (NEW) | Authoritative resume runner for interrupted runs. Checks Schema v2 and config/partition hashes. |
| `scripts/preprocess.py` | Python Script | `REPLACE` | Legacy preprocessing script; functionality integrated into data pipeline and study paired-partition contract. |
| `scripts/train.py` | Python Script | `LEGACY` | Ad-hoc centralized training script. Subsumed by `scripts/run.py` with `pipeline=centralized`. |
| `scripts/federated_train.py` | Python Script | `LEGACY` | Ad-hoc federated simulation runner. Subsumed by `scripts/run.py` with `pipeline=federated`. |
| `scripts/federated_server.py` | Python Script | `LEGACY` | Standalone Flower server script for distributed multi-process runs. Keep as auxiliary legacy utility. |
| `scripts/federated_client.py` | Python Script | `LEGACY` | Standalone Flower client script for distributed multi-process runs. Keep as auxiliary legacy utility. |
| `scripts/evaluate.py` | Python Script | `LEGACY` | Standalone evaluation runner. Subsumed by `scripts/run.py` with `pipeline=evaluate`. |
| `scripts/reproduce_experiment.py` | Python Script | `REPLACE` | Shell wrapper around Hydra. Subsumed by `python scripts/run_study.py <study> --stage reproduction`. |
| `scripts/smoke_test.py` | Python Script | `REPLACE` | Replaced by `python scripts/run_study.py E0_verify --stage smoke`. |
| `scripts/compare_runs.py` | Python Script | `LEGACY` | Comparison utility for multi-run artifacts. Preserved for downstream reporting. |
| `scripts/build_suite_artifacts.py` | Python Script | `LEGACY` | Suite aggregation script. Preserved for legacy compatibility. |
| `scripts/plot.py` | Python Script | `ACTIVE` | Plotting entrypoint (consumed by plotting pipeline). Must NOT be called automatically inside training runs. |
| `scripts/prepare_external_datasets.py` | Python Script | `ACTIVE` | Data utility for downloading/converting BTAT, ToN-IoT, and CIC-IDS2017. |
| `scripts/run_exp5.py` | Python Script | `REPLACE` | Custom runner for multi-dataset Exp5. Replaced by `configs/study/E5_datasetwise.yaml` in `run_study.py`. |
| `scripts/scalability_report.py` | Python Script | `REPLACE` | Custom plotting/metrics script for Exp6/Exp7. Replaced by standardized `timing_round.csv` and `communication_round.csv` outputs. |
| `scripts/experiments/a1_fedtros_ablation.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell loops for ablation. Replaced by `configs/study/A1_teacher.yaml` through `A5_feature_source.yaml`. |
| `scripts/experiments/e1_closed_set.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell script for Exp1. Replaced by `configs/study/E1_iid_closed.yaml`. |
| `scripts/experiments/e2_open_set.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell script for Exp2. Replaced by `configs/study/E2_iid_osr.yaml`. |
| `scripts/experiments/e3_federated_noniid.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell script for Exp3. Replaced by `configs/study/E3_noniid_closed.yaml`. |
| `scripts/experiments/e4_combined_open_set_noniid.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell script for Exp4. Replaced by `configs/study/E4_noniid_fosr.yaml`. |
| `scripts/experiments/e5_multi_dataset_open_set_noniid.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell script for Exp5. Replaced by `configs/study/E5_datasetwise.yaml`. |
| `scripts/experiments/e6_scalability_open_set.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell script for Exp6 containing `tee` and `run_status.txt` appends. Replaced by `configs/study/E6_scalability.yaml`. |
| `scripts/experiments/e7_efficiency_scalability.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell script for Exp7. Replaced by `configs/study/E7_efficiency.yaml`. |
| `scripts/experiments/e8_labelwise_open_set.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Shell script for Exp8 (LOAO). Replaced by `configs/study/E8_leave_one_attack_out.yaml`. |
| `scripts/experiments/evaluate_commands.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Evaluation loop script. Replaced by `scripts/run.py` evaluation pipeline. |
| `scripts/experiments/export_results.sh` | Bash Shell | `LEGACY` | Result export commands. Retained for historical provenance. |
| `scripts/experiments/multirun_alpha_seed_sensitivity.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Sensitivity matrix shell loop. Replaced by `configs/study/S1_sensitivity.yaml`. |
| `scripts/experiments/multirun_main_methods.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Multi-method loop. Replaced by study matrix expansion. |
| `scripts/experiments/resume_commands.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Manual resume commands. Replaced by `scripts/resume.py`. |
| `scripts/experiments/resume_exp8_fedtros.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Ad-hoc Exp8 resume bash script with pipe `tee`. Replaced by `scripts/resume.py`. |
| `scripts/experiments/run_full_suite.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Full suite runner calling bash scripts. Replaced by `run_study.py`. |
| `scripts/experiments/run_validation_tiny.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Tiny validation script. Replaced by `run_study.py E0_verify --stage smoke`. |
| `scripts/experiments/validate_configs.sh` | Bash Shell | `REPLACE` / `DELETE_AFTER_VALIDATION` | Hydra config validation script. Replaced by `test_study_composition` in automated test suite. |
| `run_status.txt` | Text Status Log | `DELETE_AFTER_VALIDATION` | Fragile append-only text file (`DONE <run_id>`, `FAILED <run_id>`). Replaced by structured `status` in `run_manifest.json` and tracker. |
| `Host1` / `Host2` hardcoded strings | Metadata / Config | `DELETE_AFTER_VALIDATION` | Manual host names. Replaced by automatic hardware & hostname capture via `src.infrastructure.hardware`. |
| `tracking.run_id=...` CLI overrides | Config Property | `DELETE_AFTER_VALIDATION` | Manual string typing. Replaced by deterministic config-derived run ID slug generation. |

---

## 2. Logging Audit (Item B2)

Audit of logging, print, and tracking calls across the FedTROS codebase (`src/`, `scripts/`, `run.py`):

| Occurrence Location | Category | Target Migration / Action |
|---|---|---|
| `run.py`: lines 37, 108, 286 (`logger.info(...)`) | `OPERATIONAL_LOG` | Migrated to centralized `src.infrastructure.logging`. |
| `run.py`: line 306 (`print("FedTROS: skipping...")`) | `REMOVE` | Converted to `logger.debug` in centralized logger. |
| `scripts/prepare_external_datasets.py`: line 26 (`print(...)`) | `OPERATIONAL_LOG` | Converted to standard `logger.info`. |
| `scripts/run_exp5.py`: line 83 (`print(...)`) | `OPERATIONAL_LOG` | Subsumed by `run_study.py` study logs. |
| `src/tracking/local.py`: lines 153, 178, 203 (`log_metrics`, CSV rewrite) | `RESULT_ARTIFACT` | Standardized into `src.infrastructure.tracking.local_tracker`. |
| `src/artifacts/manifests.py`: line 109 (`logger.info(...)`) | `OPERATIONAL_LOG` | Standardized in `src.infrastructure.manifests`. |
| `src/artifacts/communication.py`: lines 54, 65, 83 (`logger.warning(...)`) | `OPERATIONAL_LOG` | Centralized logger warning. |
| `src/checkpointing/checkpoints.py`: lines 79, 127 (`logger.info(...)`) | `OPERATIONAL_LOG` | Centralized logger with Schema v2 validation. |
| `src/federated/server.py`: lines 40, 145, 155, 179 (`logger.*`) | `OPERATIONAL_LOG` | Centralized logger. |
| `src/federated/server.py`: round metric dictionary emission | `SCIENTIFIC_METRIC` | Standardized metric namespace (`federated/round`, `closed_set/*`, `open_set/*`). |
| `src/federated/client.py`: lines 47, 62, 101 (`self.logger.*`) | `OPERATIONAL_LOG` | Centralized logger with `Client.{cid}` name. |
| `src/models/bundle.py`: lines 29, 175 (`self.logger.*`) | `OPERATIONAL_LOG` | Centralized logger. |
| `src/models/variational_classifier_teacher.py`: line 27 | `OPERATIONAL_LOG` | Centralized logger with `VCT` name. |
| Legacy EVT/Feature-EVT modules | `REMOVED/ARCHIVED` | Canonical open-set path now uses `prototype_rank` with known-only empirical-rank calibration; no EVT runtime fallback remains. |
| `src/evaluation/openset_eval.py`: AUROC/AUPRC computation | `SCIENTIFIC_METRIC` | Logged to `metrics_final.json`, `osr_scores.parquet`, and tracker under `open_set/*`. |
| `src/evaluation/closed_set.py`: accuracy, macro F1 | `SCIENTIFIC_METRIC` | Logged under `closed_set/accuracy`, `closed_set/macro_f1`. |
| `src/evaluation/run.py`: latent embeddings export | `RESULT_ARTIFACT` | Exported to `artifacts/latent_embeddings.csv`. |
| `src/training/distillation.py`: loss components | `SCIENTIFIC_METRIC` | Standardized under `teacher/*` and `student/*`. |
| `src/utils/utils.py`: line 50 (`logging.info(f"Global seed...")`) | `OPERATIONAL_LOG` | Centralized logging. |
| `scripts/experiments/e6_scalability_open_set.sh`: `tee`, `run_status.txt` | `REMOVE` | Shell pipe tee and text appends completely removed. |

---

## 3. Classification Summary

- **ACTIVE**: 3 modules (`scripts/run.py`, `scripts/run_study.py`, `scripts/resume.py`, `scripts/prepare_external_datasets.py`, `scripts/plot.py`).
- **REPLACE**: 15 legacy scripts and shell harnesses replaced by the unified study runner and study configs.
- **LEGACY**: 5 helper scripts preserved for retrospective tool compatibility without breaking existing data extractions.
- **DELETE_AFTER_VALIDATION**: 12 shell scripts (`*.sh`), `run_status.txt`, and manual host scripts slated for deletion once Workstream B tests pass.
