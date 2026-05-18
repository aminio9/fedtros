from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotSpec:
    plot_id: str
    title: str
    category: str
    required_data: tuple[str, ...]


REQUIRED_PLOTS: tuple[PlotSpec, ...] = (
    PlotSpec(
        "scalability_nodes_vs_accuracy",
        "1. Scalability: Nodes vs Final Accuracy",
        "federated",
        ("scalability.csv",),
    ),
    PlotSpec(
        "non_iid_data_distribution",
        "2. Non-IID Data Distribution",
        "data",
        ("partition_manifest.jsonl",),
    ),
    PlotSpec(
        "convergence_mild_non_iid",
        "3. Convergence & Variance (Mild Non-IID)",
        "federated",
        ("comparison_metrics.csv",),
    ),
    PlotSpec(
        "convergence_hard_non_iid",
        "4. Convergence & Variance (Hard Non-IID)",
        "federated",
        ("comparison_metrics.csv",),
    ),
    PlotSpec(
        "known_unknown_score_distribution",
        "5. Known vs Unknown Score Distribution",
        "open_set",
        ("open_set_scores.csv",),
    ),
    PlotSpec(
        "openness_vs_auroc",
        "6. Openness vs AUROC Performance",
        "open_set",
        ("openness_metrics.csv",),
    ),
    PlotSpec(
        "unknown_detection_roc",
        "7. ROC Curve for Unknown Zero-Day Attacks",
        "open_set",
        ("open_set_roc_curve.csv",),
    ),
    PlotSpec(
        "cross_dataset_generalization",
        "8. Cross-Dataset Generalization",
        "generalization",
        ("cross_dataset_metrics.csv",),
    ),
    PlotSpec(
        "confusion_matrix_before_osr",
        "9. BEFORE OSR: Unknowns Misclassified",
        "evaluation",
        ("before_osr_confusion_matrix.csv",),
    ),
    PlotSpec(
        "confusion_matrix_after_osr",
        "10. AFTER OSR: Unknowns Safely Detected",
        "evaluation",
        ("after_osr_confusion_matrix.csv",),
    ),
    PlotSpec(
        "seed_robustness_boxplot",
        "11. Robustness: Variance Across Random Seeds",
        "robustness",
        ("seed_robustness.csv",),
    ),
    PlotSpec(
        "latent_space_separation",
        "12. t-SNE/UMAP Latent Space Separation",
        "representation",
        ("latent_embeddings.csv",),
    ),
    PlotSpec(
        "communication_efficiency",
        "13. Communication Efficiency",
        "systems",
        ("communication_metrics.csv",),
    ),
    PlotSpec(
        "architectural_ablation",
        "14. Ablation Study: Impact of Modules",
        "ablation",
        ("ablation_metrics.csv",),
    ),
)


def plot_specs_by_id() -> dict[str, PlotSpec]:
    return {spec.plot_id: spec for spec in REQUIRED_PLOTS}
