from pathlib import Path

import torch


def load_tensor_dataset(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a project tensor dataset saved as {'features': Tensor, 'labels': Tensor}."""
    data = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(data, dict) or "features" not in data or "labels" not in data:
        raise ValueError("Expected a tensor dataset with 'features' and 'labels' keys.")
    features = data["features"].float()
    labels = data["labels"].long()
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [N, D], got {tuple(features.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D [N], got {tuple(labels.shape)}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels have different numbers of samples.")
    return features, labels
