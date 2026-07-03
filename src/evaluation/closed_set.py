from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
<<<<<<< HEAD
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.agents.agent import Agent
=======
from torch.utils.data import DataLoader, TensorDataset

from src.agents.agent import Agent
from src.evaluation.metrics import compute_classification_metrics
>>>>>>> ea28efe (Initial commit with updated source code)

logger = logging.getLogger(__name__)


def load_class_names(path: str | Path, num_actions: int) -> dict[int, str]:
    class_path = Path(path)
    if not class_path.exists():
        raise FileNotFoundError(f"Class-name mapping not found: {class_path}")
    raw = json.loads(class_path.read_text(encoding="utf-8"))
    class_names = {int(k): str(v) for k, v in raw.items()}
    missing = [idx for idx in range(num_actions) if idx not in class_names]
    if missing:
        raise ValueError(f"Class-name mapping is missing class ids: {missing}")
    return class_names


def evaluate_closed_set(
    agent: Agent,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    class_names: dict[int, str],
    output_dir: str | Path,
    prefix: str = "test",
    save_predictions: bool = True,
) -> dict[str, Any]:
    loader = DataLoader(
        TensorDataset(features.float(), labels.long()), batch_size=batch_size, shuffle=False
    )
<<<<<<< HEAD
    agent.prior_net.eval()
    agent.value_net_main.eval()

=======
>>>>>>> ea28efe (Initial commit with updated source code)
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
<<<<<<< HEAD
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            mu, _ = agent.prior_net(batch_features)
            logits = agent.value_net_main(mu, batch_features)
            loss = F.cross_entropy(logits, batch_labels, reduction="sum")
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item())
            total += int(batch_labels.numel())
            y_true.extend(batch_labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    label_ids = sorted(class_names)
    target_names = [class_names[idx] for idx in label_ids]
    loss_value = total_loss / max(total, 1)
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    balanced = balanced_accuracy_score(y_true, y_pred) if y_true else 0.0
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
=======
    prior_was_training = agent.prior_net.training
    q_was_training = agent.value_net_main.training
    try:
        agent.prior_net.eval()
        agent.value_net_main.eval()
        with torch.no_grad():
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                mu, _ = agent.prior_net(batch_features)
                logits = agent.value_net_main(mu, batch_features)
                loss = F.cross_entropy(logits, batch_labels, reduction="sum")
                preds = logits.argmax(dim=1)
                total_loss += float(loss.item())
                total += int(batch_labels.numel())
                y_true.extend(batch_labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
    finally:
        agent.prior_net.train(prior_was_training)
        agent.value_net_main.train(q_was_training)

    label_ids = sorted(class_names)
    loss_value = total_loss / max(total, 1)
    metric_bundle = compute_classification_metrics(
        y_true,
        y_pred,
        label_ids=label_ids,
        class_names=class_names,
    )
    target_names = metric_bundle["target_names"]
>>>>>>> ea28efe (Initial commit with updated source code)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = {
        f"{prefix}/loss": float(loss_value),
<<<<<<< HEAD
        f"{prefix}/accuracy": float(accuracy),
        f"{prefix}/balanced_accuracy": float(balanced),
        f"{prefix}/macro_precision": float(macro_precision),
        f"{prefix}/macro_recall": float(macro_recall),
        f"{prefix}/macro_f1": float(macro_f1),
        "num_examples": int(total),
        "per_class_accuracy": {
            class_names[idx]: float(per_class_recall[pos]) for pos, idx in enumerate(label_ids)
        },
    }
    if prefix == "test":
        metrics["test/loss"] = float(loss_value)
        metrics["test/accuracy"] = float(accuracy)
=======
        f"{prefix}/accuracy": float(metric_bundle["accuracy"]),
        f"{prefix}/balanced_accuracy": float(metric_bundle["balanced_accuracy"]),
        f"{prefix}/macro_precision": float(metric_bundle["macro_precision"]),
        f"{prefix}/macro_recall": float(metric_bundle["macro_recall"]),
        f"{prefix}/macro_f1": float(metric_bundle["macro_f1"]),
        f"{prefix}/weighted_f1": float(metric_bundle["weighted_f1"]),
        "num_examples": int(total),
        "per_class_accuracy": metric_bundle["per_class_recall_by_name"],
        f"{prefix}/per_class_recall": metric_bundle["per_class_recall_by_name"],
    }
    if prefix == "test":
        metrics["test/loss"] = float(loss_value)
        metrics["test/accuracy"] = float(metric_bundle["accuracy"])
>>>>>>> ea28efe (Initial commit with updated source code)
    (output_path / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / f"{prefix}_classification_report.json").write_text(
<<<<<<< HEAD
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    confusion_df = pd.DataFrame(cm, index=target_names, columns=target_names)
=======
        json.dumps(metric_bundle["classification_report"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    confusion_df = pd.DataFrame(
        metric_bundle["confusion_matrix"],
        index=target_names,
        columns=target_names,
    )
>>>>>>> ea28efe (Initial commit with updated source code)
    confusion_df.to_csv(output_path / f"{prefix}_confusion_matrix.csv")
    if save_predictions:
        pred_records = [
            {"y_true": int(t), "y_pred": int(p)} for t, p in zip(y_true, y_pred, strict=True)
        ]
        (output_path / f"{prefix}_predictions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in pred_records),
            encoding="utf-8",
        )

    logger.info(
        "Closed-set %s metrics | loss=%.6f | accuracy=%.4f | macro_f1=%.4f",
        prefix,
        loss_value,
<<<<<<< HEAD
        accuracy,
        macro_f1,
=======
        metric_bundle["accuracy"],
        metric_bundle["macro_f1"],
>>>>>>> ea28efe (Initial commit with updated source code)
    )
    return metrics
