# FedTROS-PR Refactor Deletion / Archive Manifest

This manifest records code removed from the active execution path during the VCT + W&B + publication-bundle migration. During implementation, potentially useful historical files were moved under `archive/migration_2026/` before final packaging so the migration remains inspectable. They are not imported or used by the canonical pipeline.

| Category | Previous responsibility | Active replacement | Scientific behavior changed? |
|---|---|---|---|
| `src/plotting/` | FedTROS-internal publication plotting | Separate `plots/` repository via versioned publication bundle | No scientific metric calculation intentionally removed |
| `scripts/plot.py` | Direct FedTROS plotting launcher | `plots/scripts/generate_all.py` | No |
| `scripts/scalability_report.py` | Mixed scalability calculations/rendering | canonical result/runtime metrics + external renderer | No; calculations preserved as data |
| `scripts/export_plot_data.py` | Fixed legacy filename adapter | `scripts/export_publication_bundle.py` | No |
| `src/configs/plotting/q1_plots.yaml` | Plot style/config inside training repo | plot-repo style configuration | No |
| old plotting test | tested removed internal plot subsystem | publication-bundle + no-plotting tests | No |
| `CompositeTracker` | multiplexed local + external tracking | W&B tracker + independent ResultStore | No |
| `LocalTracker` | mixed metrics persistence/tracking | `ResultStore` | No |
| `MLflowTracker` | second external tracker | W&B only | No |
| old `src/tracking/` | duplicate local tracker package | `src/infrastructure/tracking/` | No |
| tracking `local.yaml` | local tracker configuration | W&B online/offline/disabled + ResultStore | No |
| old experiment shell scripts | manual Hydra command matrices/status/tee | declarative study YAML + `run_study.py` | No |
| `run_status.txt` pattern | DONE/FAILED append-only state | run manifest lifecycle | No |
| manual Host1/Host2 notes | execution host provenance | automatic run/hardware metadata | No |
| old train/federated launch scripts | overlapping entry points | `run.py` + `run_study.py` | No intended scientific change |
| old exp1…exp8 configs | DQN-era experiment API | E0–E8/A1–A5/S1 study configs | Yes: experiment contract updated intentionally |
| old `method/fedtros.yaml` | legacy method identity | `method/fedtros_pr.yaml` | Naming/config migration |
| old EVT config | inherited legacy detector configuration | explicit Prototype-Rank config | Yes: P0 selector/calibration correction |
| old communication estimator | checkpoint/file-size estimate | actual transmitted NumPy payload accounting | Yes: measurement validity improved |
| old suite builder | manually constructed publication artifacts | canonical analysis + publication bundle | No |
| `selection_utils.py` | RL-era selection utility | no active RL subsystem | Yes: obsolete RL behavior removed |
| duplicate VCT file | redundant teacher implementation | `src/models/variational_teacher.py` | No; one canonical implementation |
| FedGPA experimental baseline | unused baseline implementation | none; archived | No paper claim depends on it |
| stale pre-W&B `poetry.lock` | dependency lock for old plot/MLflow stack | regenerate lock on target Python 3.11/3.12 | No |
| stale tests for removed APIs | old config/OSR/execution expectations | canonical tests | No |

## Active-code deletion rules enforced

The canonical FedTROS source must not depend on:

- Matplotlib/Seaborn/Plotly rendering;
- MLflow/TensorBoard/CompositeTracker/LocalTracker;
- DQN/Q/replay/reward/Bellman training machinery;
- legacy DKD-FedOS / Fed-DiGOS / FedPROTEUS method identities;
- PNPFF as the implemented detector;
- shell-text status tracking.

The archive is migration evidence only. Git history should be the long-term recovery mechanism after the refactor is accepted.

## Final packaging cleanup

The final clean distribution excludes the temporary `archive/migration_2026/` tree and Python cache/generated-test files. Historical recovery is intentionally delegated to the preserved development copy/Git history rather than shipping dead code in the production archive.

Additional active removals completed before packaging:

- `src/openset/evt.py` and EVT/Weibull runtime fallback from Prototype-Rank;
- unused `src/configs/model/openset_qchain.yaml`;
- active EVT configuration keys and `evt_dir` path;
- stale compiled caches and pytest caches.
