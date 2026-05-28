from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.evaluation.run import _latent_export_tensor


def test_latent_export_tensor_prefers_open_set_tensor_when_available(tmp_path):
    closed_features = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    closed_labels = torch.tensor([0, 1])
    open_features = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    open_labels = torch.tensor([1, -1])

    torch.save({"features": closed_features, "labels": closed_labels}, tmp_path / "closed.pt")
    torch.save({"features": open_features, "labels": open_labels}, tmp_path / "open.pt")

    cfg = OmegaConf.create(
        {
            "evaluation": {
                "mode": "open_set",
                "open_set_data": str(tmp_path / "open.pt"),
            }
        }
    )

    features, labels, source = _latent_export_tensor(
        cfg,
        project_root=Path(tmp_path),
        closed_features=closed_features,
        closed_labels=closed_labels,
    )

    assert source == "open_set"
    assert torch.equal(features, open_features)
    assert torch.equal(labels, open_labels)


def test_latent_export_tensor_falls_back_to_closed_tensor_in_closed_mode(tmp_path):
    closed_features = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    closed_labels = torch.tensor([0, 1])
    open_features = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    open_labels = torch.tensor([1, -1])

    torch.save({"features": closed_features, "labels": closed_labels}, tmp_path / "closed.pt")
    torch.save({"features": open_features, "labels": open_labels}, tmp_path / "open.pt")

    cfg = OmegaConf.create(
        {
            "evaluation": {
                "mode": "closed_set",
                "open_set_data": str(tmp_path / "open.pt"),
            }
        }
    )

    features, labels, source = _latent_export_tensor(
        cfg,
        project_root=Path(tmp_path),
        closed_features=closed_features,
        closed_labels=closed_labels,
    )

    assert source == "closed_set"
    assert torch.equal(features, closed_features)
    assert torch.equal(labels, closed_labels)
