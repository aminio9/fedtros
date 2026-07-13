# Experiment 5 Literature Context

These are reported paper results, not rows in the same DKD-FedOS leaderboard.
The representations, splits, client construction, and open-set assumptions differ.

## Open-Set Axis: EFC

Souza et al., *Computers & Security* 157 (2025), 104569,
<https://doi.org/10.1016/j.cose.2025.104569>.

- Centralized EFC, not federated learning.
- Its CIC-IDS2017 experiment merges the three Web Attack labels and uses
  leave-one-attack-out evaluation with five-fold cross-validation.
- The separate Zhang protocol uses 256 header features, an 80/20 split, and
  six leave-one-out unknown attacks.
- Reported average: AUROC 0.859 and AUPRC 0.993.
- EFC's 0.5 is a classifier pseudocount, not a Dirichlet alpha.

## Non-IID Axis: FTKD

Zhou et al., *Expert Systems with Applications* 299 (2026), 130144,
<https://doi.org/10.1016/j.eswa.2025.130144>.

- Closed-set federated classification on all ten ToN-IoT classes.
- Uses 5% public data, 76% client training data, and 19% test data.
- Heterogeneity is manually constructed; no Dirichlet alpha is reported.
- Its 30-client comparison reports 97.72% accuracy, 97.72% recall, and 97.71% F1.

## Dataset-Native Closed-Set Axis: Co-CNN/BTAT

The BTAT paper, *IEEE Transactions on Cognitive Communications and Networking*
(2025), <https://doi.org/10.1109/TCCN.2025.3637274>.

- Confirms seven classes and 302,749 samples.
- Converts Bytecode and Value to grayscale images.
- Uses equal partitions across 3, 5, or 10 mining nodes.
- It is closed-set and reports neither a Dirichlet alpha nor a clear train/test ratio.

## Reporting Rule

Report DKD-FedOS in its own alpha-0.5 table. Do not calculate cross-protocol
percentage improvement, significance, or a single winner against these values.
