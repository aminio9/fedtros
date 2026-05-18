# Experiment Plan

The experiment plan is extracted from `Experimentplan/Improved_Experiment_Plan.docx` and reconciled with `Experimentplan/testplot.py`.

## Research Objective

Evaluate a federated intrusion-detection framework, named Proposed/FMRL_LA in the plotting template, for closed-set classification of known IoT traffic and open-set detection of unknown or zero-day attacks.

Primary claims:

- Known-class classification effectiveness.
- Unknown attack rejection quality.
- Robustness to non-IID client distributions and random seeds.
- Scalability and communication efficiency.
- Cross-dataset generalization.

## Datasets

- Primary: B-NAT.
- Generalization: B-TAT, ToN-IoT, CIC-IDS2017.
- Known labels from the plan/config: `Normal`, `BP`, `DoS`, `MitM`.
- Unknown labels are held out from training and encoded as `-1` for open-set evaluation.

## Protocol

- Partitioning: IID or Dirichlet non-IID.
- Dirichlet alpha values: `0.1`, `0.5`, `1.0`, `10`, plus IID.
- Scalability client counts: `3`, `10`, `20`, `50`, `100`.
- Main federated baselines: FedAvg, FedProx, FMRL_LA.
- Open-set baseline: closed-set softmax confidence where available.
- Repetitions: at least five seeds for main comparisons, preferably ten or more for robustness plots.

## Primary Metrics

- Closed-set: accuracy, balanced accuracy, macro precision, macro recall, macro F1, per-class recall.
- Open-set: AUROC, AUPRC, FPR95, unknown-detection recall, unknown F1.
- Federated: final global accuracy, convergence speed, per-client accuracy/loss, selected-client fraction.
- Systems: communication MB, accuracy per MB, runtime/memory when available.
