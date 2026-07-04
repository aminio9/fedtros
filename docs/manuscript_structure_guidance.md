# Manuscript Structure Guidance

This guide describes what each manuscript section should contain, excluding the Proposed Method section, which is already written as the main technical contribution.

## 1. Introduction

Paragraph 1 should introduce the security setting and why blockchain edge traffic requires distributed intrusion detection. It should motivate privacy, decentralization, and operational heterogeneity.

Paragraph 2 should define the open-set problem: models trained only on known classes must not force every future sample into one known label. Explain why unknown attacks such as held-out FoT are realistic.

Paragraph 3 should identify the methodological gap: standard federated aggregation improves privacy but does not directly solve non-IID client drift or unknown-attack rejection.

Paragraph 4 should summarize the proposed contribution in one compact paragraph: CVAE-DQN local learning, generator reconstruction, FMRL-AVA vector-aligned utility aggregation, and EVT calibration.

Paragraph 5 should list contributions as concise claims. Use evidence-oriented wording and avoid overclaiming before results.

Needed material: one contribution list, no figure unless the journal expects a graphical abstract.

## 2. Related Work

Paragraph 1 should cover federated learning for intrusion detection and explain why FedAvg and FedProx are the correct baselines.

Paragraph 2 should cover reinforcement learning and Double-DQN methods for adaptive classification or security decision-making.

Paragraph 3 should cover open-set recognition and EVT-based tail modeling.

Paragraph 4 should position the paper against prior work: most studies handle either federated IDS, RL-based IDS, or open-set recognition separately; this paper combines all three with cooperative aggregation.

Needed material: a comparison table is recommended. Columns should include method family, federated setting, non-IID support, open-set detection, reconstruction signal, and client-selection mechanism.

## 3. Problem Formulation

Paragraph 1 should define the federated clients, local datasets, feature space, known labels, held-out unknown labels, and privacy constraint.

Paragraph 2 should define the local MDP/contextual-bandit formulation: state, action, class-balanced reward option, transition assumption, replay buffer, auxiliary CE stabilizer, and target-network role.

Paragraph 3 should define the federated objective and show the FedAvg/FedProx optimization equations. For FMRL-AVA, explain that the FedAvg sample-count prior is preserved, multiplied by a bounded client utility, and then modulated by an update-vector alignment multiplier, so the server update uses \(a_i=n_i u_i m_i\).

Paragraph 4 should define the open-set decision objective and explain why reconstruction error plus EVT is used instead of closed-set confidence alone.

Needed material: notation table is recommended. Include symbols for clients, datasets, parameters, thresholds, and metrics.

## 5. Experimental Setup

Paragraph 1 should describe the dataset, known/unknown label protocol, and generated tensor artifacts.

Paragraph 2 should describe preprocessing: train-only scaler/encoder fitting, label mapping, closed-set and open-set splits, and client shard generation.

Paragraph 3 should describe federated configuration: client count, rounds, local episodes, client sampling, IID/non-IID settings, and Dirichlet alpha values.

Paragraph 4 should define baselines: centralized, local-only, FedAvg, FedProx, and FMRL-AVA. State that FMRL-AVA uses deterministic audit signals, bounded per-client critic residuals, FedAWA-style update-vector alignment, and a validation-aware mixer target. Clarify that unavailable validation metrics are omitted rather than treated as zero.

Paragraph 5 should describe implementation and reproducibility controls: Hydra configs, fixed seeds, shared partitions, checkpointing, and offline artifacts.

Needed material: an experimental matrix table and a hyperparameter table.

## 6. Evaluation Metrics

Paragraph 1 should define closed-set metrics: accuracy, macro precision/recall/F1, weighted F1, balanced accuracy, and confusion matrices.

Paragraph 2 should define open-set metrics: AUROC, AUPRC, FPR@95%TPR, TPR@5%FPR, unknown F1, unknown rejection rate, and known accuracy after rejection.

Paragraph 3 should define federated metrics: convergence speed, final-10-round mean accuracy, stability, client-wise variance, selected-client fraction, and communication cost.

Paragraph 4 should define efficiency metrics: wall-clock time, transmitted bytes, rounds to convergence, and accuracy per megabyte.

Needed material: a metrics table with definitions and whether higher/lower is better.

## 7. Results and Discussion

Paragraph 1 should report closed-set results and discuss known-class behavior.

Paragraph 2 should report open-set results and discuss the score distribution, rejection threshold, and trade-off between unknown detection and known retention.

Paragraph 3 should report non-IID federated results across Dirichlet alpha values.

Paragraph 4 should report external validation on B-TAT, ToN-IoT, and CIC-IDS2017, emphasizing that each dataset is trained and evaluated independently.

Paragraph 5 should discuss communication and scalability, especially whether FMRL-AVA's two-phase overhead is justified by utility per round.

Paragraph 6 should interpret failure cases using confusion matrices, per-client variance, and score distributions.

Needed material: main comparison table, ROC/PR curves, convergence curves, communication-cost plot, score-distribution plot, and confusion matrices.

## 8. Ablation and Sensitivity

Paragraph 1 should explain the purpose of ablations: isolating EVT, generator reconstruction, FMRL-AVA, FedAvg, FedProx, and client selection.

Paragraph 2 should report module-removal results.

Paragraph 3 should report sensitivity to Dirichlet alpha, EVT tail fraction, utility threshold, vector-alignment strength, random seed, and client count.

Needed material: ablation table and sensitivity plots.

## 9. Limitations and Threats to Validity

Paragraph 1 should discuss dataset scope and generalization.

Paragraph 2 should discuss EVT calibration sensitivity.

Paragraph 3 should discuss FMRL-AVA overhead and additional server-side state, including audit metadata, per-client critics, the centralized mixer, vector-alignment bookkeeping, validation reward bookkeeping, and the communication trade-off created by client selection.

Paragraph 4 should discuss implementation risks: seed variance, client partition fairness, and hyperparameter sensitivity.

Needed material: no figure required.

## 10. Reproducibility

Paragraph 1 should list configuration control, seeds, run commands, and generated artifacts.

Paragraph 2 should explain where checkpoints, metrics, plots, and suite-level CSV files are written.

Paragraph 3 should state that all comparisons reuse the same split, preprocessing contract, seed, and client partitions.

Needed material: artifact table or command listing.

## 11. Conclusion

Paragraph 1 should summarize the technical contribution.

Paragraph 2 should summarize the experimental evidence expected or reported.

Paragraph 3 should state future work: adaptive unknown classes, dataset-specific calibration refinement, and communication-efficient FMRL-AVA variants.

Needed material: no table or figure required.
