from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    *,
    label_ids: list[int],
    class_names: dict[int, str],
) -> dict[str, Any]:
    """Compute closed-set classification metrics with one label ordering."""
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    labels = [int(label) for label in label_ids]
    target_names = [str(class_names[label]) for label in labels]
    if true.size == 0:
        zero_by_id = {label: 0.0 for label in labels}
        zero_by_name = {class_names[label]: 0.0 for label in labels}
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_recall_by_id": zero_by_id,
            "per_class_recall_by_name": zero_by_name,
            "per_class_f1_by_id": zero_by_id,
            "classification_report": {},
            "classification_report_text": "",
            "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=np.int64),
            "target_names": target_names,
        }

    per_class_recall = recall_score(
        true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    report = classification_report(
        true,
        pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    report_text = classification_report(
        true,
        pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "per_class_recall_by_id": {
            int(label): float(per_class_recall[pos]) for pos, label in enumerate(labels)
        },
        "per_class_recall_by_name": {
            class_names[label]: float(per_class_recall[pos]) for pos, label in enumerate(labels)
        },
        "per_class_f1_by_id": {
            int(label): float(per_class_f1[pos]) for pos, label in enumerate(labels)
        },
        "classification_report": report,
        "classification_report_text": report_text,
        "confusion_matrix": confusion_matrix(true, pred, labels=labels),
        "target_names": target_names,
    }
