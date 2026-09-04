import torch

from src.data.io import load_tensor_dataset


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
