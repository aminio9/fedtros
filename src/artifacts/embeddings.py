"""Project and export latent embeddings for visual inspection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        parameter = next(module.parameters())
        return parameter.device
    except StopIteration:
        return torch.device("cpu")


def _label_name(label: int, class_names: dict[int, str]) -> str:
    if label < 0:
        return "Unknown"
    return str(class_names.get(label, f"class_{label}"))


def _project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if embeddings.shape[1] == 1:
        return np.column_stack([embeddings[:, 0], np.zeros(len(embeddings), dtype=np.float32)])
    if embeddings.shape[1] == 2:
        return embeddings.astype(np.float32)
    return PCA(n_components=2, random_state=0).fit_transform(embeddings).astype(np.float32)


def export_latent_embeddings(
    *,
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: dict[int, str],
    output_path: Path,
    batch_size: int,
    max_points: int = 5000,
    source: str | None = None,
) -> pd.DataFrame:
    """Project latent representations to 2D and persist them as a plotting CSV."""
    target_module = model
    if target_module is None:
        raise ValueError("model must be provided.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = TensorDataset(features.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = _module_device(target_module)
    latent_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    with torch.no_grad():
        target_module.eval()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            if hasattr(target_module, "distill_forward"):
                # VariationalClassifierTeacher
                _, mu, _ = target_module.distill_forward(batch_features)
                rep = mu
            elif hasattr(target_module, "classifier_features"):
                # StudentIDSModel
                rep = target_module.classifier_features(batch_features)
            elif callable(target_module):
                out = target_module(batch_features)
                rep = out[0] if isinstance(out, (tuple, list)) else out
            else:
                rep = batch_features
            latent_batches.append(rep.detach().cpu().numpy())
            label_batches.append(batch_labels.cpu().numpy())

    if latent_batches:
        latent = np.concatenate(latent_batches, axis=0)
        latent_labels = np.concatenate(label_batches, axis=0)
    else:
        latent = np.empty((0, 2), dtype=np.float32)
        latent_labels = np.empty((0,), dtype=np.int64)

    if max_points > 0 and len(latent) > max_points:
        indices = np.linspace(0, len(latent) - 1, num=max_points, dtype=int)
        latent = latent[indices]
        latent_labels = latent_labels[indices]

    projected = _project_embeddings(latent)
    frame = pd.DataFrame(
        {
            "x": projected[:, 0] if len(projected) else [],
            "y": projected[:, 1] if len(projected) else [],
            "label": [_label_name(int(label), class_names) for label in latent_labels],
        }
    )
    if source is not None:
        frame["source"] = source
    frame.to_csv(output_path, index=False)

    meta: dict[str, Any] = {
        "batch_size": int(batch_size),
        "max_points": int(max_points),
        "num_points": int(len(frame)),
        "latent_dim": int(latent.shape[1] if latent.ndim == 2 else 0),
        "projection": "pca" if latent.shape[1] > 2 else "identity",
    }
    if source is not None:
        meta["source"] = source
    output_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Saved latent embeddings to %s (%d rows)", output_path, len(frame))
    return frame
