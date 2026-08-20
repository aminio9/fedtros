# W&B tracking contract

W&B is the **single interactive experiment tracker**. It is not the local scientific source of truth.

## Separation of concerns

```text
scientific code -> structured metrics -> RunServices
                                  |-> ResultStore (required)
                                  `-> WandBTracker (online/offline/disabled)
```

Scientific modules do not import `wandb`. The SDK is isolated under `src/infrastructure/tracking/`.

## Modes

Configuration: `src/configs/tracking/wandb.yaml`

```yaml
backend: wandb
mode: online   # online | offline | disabled
project: FedTROS-PR
```

`disabled` selects `NullTracker`; training/evaluation and publication exports continue to work.

## Stable metric namespaces

- `teacher/*`
- `student/*`
- `federated/*`
- `closed_set/*`
- `open_set/*`
- `prototype_rank/*`
- `communication/*`
- `runtime/*`

Operational events belong in `logs/run.log`, not metric namespaces.

## Grouping

W&B project: `FedTROS-PR`.

The study ID is the default group, for example `E4-NIID-FOSR`. Run names include dataset, method, alpha/IID, unknown condition, client count, and seed.

## Secrets

Never place `WANDB_API_KEY` in committed YAML/source files. Authenticate in the environment or with `wandb login`.

## Artifact policy

Large data stay in the canonical local run directory by default. Upload only useful artifacts/checkpoints according to policy; do not upload every embedding/checkpoint every round.

## Live federated-round metrics

The Flower server emits round events through the generic `RunServices` metrics sink. The server itself does **not** import W&B.

For each federated round, the canonical event stream can include:

- `federated/phase=fit`
- `federated/phase=central_validation`
- `federated/phase=client_evaluate`

Every event carries `federated/round`, which is the scientific x-axis. W&B's internal event step is allowed to increase monotonically across multiple events from the same federated round; do not infer the FL round from the W&B step counter.

The same payload is appended to `metrics/round_metrics.csv` before the external tracker call, so live dashboard failure cannot erase the local scientific trace.
