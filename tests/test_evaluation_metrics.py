import numpy as np

from src.evaluation.metrics import compute_classification_metrics


def test_classification_metrics_use_stable_label_order_and_names():
    metrics = compute_classification_metrics(
        np.array([0, 1, 1, 2]),
        np.array([0, 1, 2, 2]),
        label_ids=[0, 1, 2],
        class_names={0: "normal", 1: "dos", 2: "mitm"},
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"].shape == (3, 3)
    assert metrics["target_names"] == ["normal", "dos", "mitm"]
    assert metrics["per_class_recall_by_name"]["dos"] == 0.5
