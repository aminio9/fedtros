# Source Code Map

This document maps the code that actually executes. It should be read before
using README examples, method configs, or experiment scripts.

## 1. Project Overview

The implemented project trains a CVAE-DQN-style classifier for tabular
blockchain intrusion data, optionally in a federated Flower simulation, then
performs open-set rejection with EVT over generator reconstruction error.

The code also contains conventional open-set scorer utilities. Those
components are tested building blocks, but they are not first-class `run.py`
experiment pipelines yet.

## 2. Entry Points

| Command | Main function | Model | Method | Runtime | Safe? | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `python scripts/cheap_validation.py` | `main()` | CVAE-DQN plus open-set scorer utilities | synthetic preflight | CPU tiny | yes | canonical preflight |
| `python run.py experiment=validation runtime=tiny seed=42` | `run.py:main()` | CVAE-DQN | tiny FMRL-AVA full pipeline | CPU tiny | yes | integration preflight |
| `python run.py experiment=<exp> +method=<method>` | `run.py:main()` | CVAE-DQN | selected method | default/full | no | expensive paper-style run |
| `python run.py experiment=<exp> experiment.pipeline=centralized` | `run_training()` | CVAE-DQN | local replay | selected runtime | maybe | centralized runner |
| `python run.py experiment.pipeline=train` | `run_training()` | CVAE-DQN | local replay | selected runtime | maybe | requires preprocessed tensors |
| `python run.py experiment.pipeline=federated` | `run_federated_simulation()` | CVAE-DQN | Flower simulation | selected runtime | maybe | requires preprocessed tensors |
| `python run.py experiment.pipeline=evaluate evaluation.checkpoint_path=<pt>` | `run_evaluation()` | CVAE-DQN | evaluation only | selected runtime | maybe | requires checkpoint |
| `python run.py experiment=all` | `_run_suite()` | CVAE-DQN | configured suite | full | no | launches child runs |
| `scripts/experiments/*.sh` | shell wrappers | CVAE-DQN | varied | full unless overridden | no | command templates |

## 3. Main Algorithm

1. Hydra composes `src/configs/config.yaml` through `config_fl.yaml`.
2. `src.utils.config.validate_config()` checks required keys and rejects
   unsupported model/scorer/pipeline combinations.
3. `src.utils.entrypoints.prepare_run_context()` resolves device, initializes
   local tracking, snapshots config, and sets seeds.
4. `src.data.preprocessing.run_preprocessing()` reads raw CSV data, creates
   known/unknown splits, scales known data, and writes tensor files.
5. `run.py` synchronizes `model.state_dim` and `model.num_actions` from
   preprocessing metadata for `full`, `centralized`, and `preprocess` modes.
6. Centralized training calls `src.training.centralized.run_training()`.
7. Federated training calls `src.federated.run.run_federated_simulation()`.
8. Local learning uses `src.rl.local_training.run_local_training_round()` and
   `src.agents.agent.Agent.train_step()`.
9. Checkpoints are selected from validation metrics only by
   `src.checkpointing.checkpoints`.
10. Evaluation calls `src.evaluation.run.run_evaluation()`.
11. Closed-set metrics come from `src.evaluation.closed_set` and
    `src.evaluation.metrics`.
12. First-class open-set evaluation uses `src.evaluation.open_set` plus
    `src.openset.evt`.

## 4. Main Model

The main model is the FastTabM-backed CVAE-DQN stack built by
`src.models.cvae_dqn.OpenSetQChainModelFactory`. The implementation lives in
`src/models/models.py`.

The stack is not a single `forward()` module during training:

| Module | File | Input | Output | Used for |
| --- | --- | --- | --- | --- |
| Prior network | `src/models/models.py` | state `s` | `mu_p`, `logvar_p` | prediction features, KL target |
| Recognition network | `src/models/models.py` | state `s`, action `a` | `mu_q`, `logvar_q` | TD latent, reconstruction latent |
| Main Q network | `src/models/models.py` | latent `z`, state `s` | class/Q logits | actions, CE, TD, evaluation |
| Target Q network | `src/models/models.py` | latent `z`, state `s` | target Q logits | Double-DQN bootstrap |
| Generator | `src/models/models.py` | latent `z`, action `a` | reconstructed state | reconstruction training, EVT |

`src.models.interface.CVAEQChainModelAdapter` exposes a read-only dictionary
output contract for evaluation/scorer compatibility:

```python
{
    "logits": Tensor,
    "features": Tensor,
    "q_values": Tensor,
    "mu": Tensor,
    "logvar": Tensor,
    "reconstruction": None,
    "aux": {},
}
```

## 5. Supported Models

| Model config | Current status | Runner support |
| --- | --- | --- |
| `model=openset_qchain` | FastTabM-backed CVAE-DQN | supported |

## 6. Supported Open-Set Scorers

| Config/scorer | Current status | Runner support |
| --- | --- | --- |
| `open_set=evt` / `evt_reconstruction` | main open-set path | supported |
| `msp` | utility baseline | tested, not a runner |
| `energy` | utility baseline | tested, not a runner |
| `prototype_distance` | utility baseline | tested, not a runner |
| `mahalanobis_distance` | utility baseline | tested, not a runner |
| `no_rejection` | utility baseline | tested, not a runner |
| `openmax_evt` | scaffold alias | rejected until real OpenMax exists |

Standalone scorer utilities are built through
`src.openset.scorers.build_open_set_scorer_from_config()`. EVT reconstruction
is handled by `src.evaluation.open_set`, not by the scorer factory.

## 7. Training Pipeline

Centralized call chain:

```text
run.py:main
-> prepare_run_context
-> run_preprocessing
-> run_training
-> BlockchainIntrusionEnv
-> Agent(OpenSetQChainModelFactory)
-> run_local_training_round
-> Agent.train_step
-> evaluate_closed_set
-> save_checkpoint
-> run_evaluation
```

Federated call chain:

```text
run.py:main
-> prepare_run_context
-> run_preprocessing
-> run_federated_simulation
-> FlowerClient
-> Agent(OpenSetQChainModelFactory)
-> run_local_training_round
-> get_strategy
-> Server.fit
-> write federated history
-> run_evaluation
```

The static tabular classification task is cast as a sampled environment with
action-independent transitions. DQN is therefore used as a value-based
classifier with replay and target-network stabilization.

## 8. Evaluation Pipeline

Open-set call chain:

```text
run.py:main
-> run_evaluation
-> build_agent
-> load checkpoint
-> evaluate_closed_set
-> fit_evt_models
-> calibrate_evt_thresholds
-> evaluate_open_set
-> write metrics, scores, curves, confusion matrices
```

The first-class open-set score is:

```text
pred = argmax Q(mu_p(s), s)
mu_q, _ = Recognition(s, pred)
s_hat = Generator(mu_q, pred)
score = mean((s_hat - s)^2) * error_scale_factor
unknown_probability = EVT_class(pred).predict_probability_unknown(score)
```

## 9. Config Hierarchy

Important config groups:

- `experiment/`: pipeline, split, run ID, suite commands.
- `runtime/`: tiny/CPU/GPU/full resource scale.
- `model/`: architecture fields only.
- `method/`: aggregation, auxiliary loss, imbalance, and ablation switches.
- `open_set/`: EVT/scorer/threshold fields.
- `training/`: local RL/replay/generator/loss settings.
- `checkpointing/`: save paths and validation-only monitor metric.

`src.utils.config.validate_config()` is the canonical contract checker.

## 10. Output Hierarchy

Local tracking writes under `outputs/<run_id>/` by default:

```text
config.yaml
resolved_config.yaml
metadata.json
metrics.jsonl
metrics.csv
latest_checkpoint.pt
last_model.pt
best_model.pt
final_model.pt
checkpoint_metadata.json
federated_history.csv
open_set_metrics.json
open_set_scores.csv
open_set_*_curve.csv
before_osr_confusion_matrix.csv
after_osr_confusion_matrix.csv
evt/
plots/
```

## 11. Canonical Files

- Entry routing: `run.py`
- Config validation: `src/utils/config.py`
- Preprocessing: `src/data/preprocessing.py`
- Main model import path: `src/models/cvae_dqn.py`
- Main model implementation: `src/models/models.py`
- Agent update logic: `src/agents/agent.py`
- Local replay loop: `src/rl/local_training.py`
- Federated runner: `src/federated/run.py`
- Strategies: `src/federated/server.py`, `src/federated/class_aware.py`
- Centralized runner: `src/training/centralized.py`
- Closed-set metrics: `src/evaluation/closed_set.py`, `src/evaluation/metrics.py`
- Open-set evaluation: `src/evaluation/open_set.py`
- EVT math/persistence: `src/openset/evt.py`
- Scorer utilities: `src/openset/scorers.py`
- Threshold utilities: `src/openset/thresholding.py`

## 12. Deprecated Or Experimental Files

- `src/evaluation/openset_eval.py`: compatibility shim only.
- Old legacy MLP and scalar-token Transformer model configs/checkpoints are not
  supported by the clean FastTabM-only model stack.
- Non-EVT open-set configs: utility/scaffold configs until a runner is added.
- `open_set=openmax_evt`: scaffold alias, rejected by validation.

## 13. Known Limitations

- MSP, energy, prototype, Mahalanobis, and no-rejection baselines do not have a
  matched experiment evaluator yet.
- OpenMax is not implemented despite the scaffold config name.
- DQN/RL claims must be stated carefully because the environment transition is
  action-independent.
- Full experiment scripts are expensive and assume real data and appropriate
  hardware.
- `black --check .` is not currently clean across the repository.

## 14. Commands To Validate

Safe commands:

```bash
python scripts/cheap_validation.py
python -m pytest
python -m compileall src tests scripts
ruff check .
python run.py experiment=validation runtime=tiny seed=42 tracking.run_id=tiny_validation_seed42
```

Do not run full suites, 100-round experiments, or GPU-scale scripts until the
safe commands pass and the dataset/hardware assumptions are confirmed.
