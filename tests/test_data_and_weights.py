import torch

from src.data.io import load_tensor_dataset
from src.training.class_balance import effective_number_class_weights


def test_tensor_dataset_loader_validates_shapes(tmp_path):
    path = tmp_path / "client.pt"
    torch.save(
        {
            "features": torch.randn(4, 3),
            "labels": torch.tensor([0, 2, 2, 0]),
        },
        path,
    )

    features, labels = load_tensor_dataset(path)

    assert features.shape == (4, 3)
    assert labels.tolist() == [0, 2, 2, 0]


def test_effective_number_class_weights_imbalance():
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    weights = effective_number_class_weights(labels, num_classes=2, beta=0.999)

    assert weights.shape[0] == 2
    # Class 1 (minority) should have higher weight than Class 0 (majority)
    assert weights[1] > weights[0]
    assert torch.isfinite(weights).all()
