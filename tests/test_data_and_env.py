import torch

from src.data.io import load_tensor_dataset
from src.rl.environment import BlockchainIntrusionEnv


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


def test_environment_uses_max_label_for_non_contiguous_local_labels(tmp_path):
    path = tmp_path / "client.pt"
    torch.save(
        {
            "features": torch.randn(6, 5),
            "labels": torch.tensor([0, 2, 2, 0, 2, 0]),
        },
        path,
    )

    env = BlockchainIntrusionEnv(str(path), steps_per_episode=2)
    obs, info = env.reset(seed=123)
    next_obs, reward, terminated, truncated, step_info = env.step(2)

    assert env.num_actions_nt == 3
    assert obs.shape == (5,)
    assert next_obs.shape == (5,)
    assert reward in {-1.0, 1.0}
    assert not truncated
    assert "true_label" in info
    assert "true_label" in step_info
    assert isinstance(terminated, bool)
