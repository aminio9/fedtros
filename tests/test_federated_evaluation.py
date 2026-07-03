import json
import logging
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.federated.client import FlowerClient
from src.federated.server import SaveModelFedAvg


def _write_tensor_dataset(path, features, labels):
    torch.save({"features": features, "labels": labels}, path)


def test_closed_set_eval_skips_empty_validation_and_uses_test_split(tmp_path):
    validation_path = tmp_path / "validation.pt"
    test_path = tmp_path / "shared_closed_set_test.pt"
    class_names_path = tmp_path / "class_names.json"

    _write_tensor_dataset(
        validation_path,
        torch.empty((0, 3), dtype=torch.float32),
        torch.empty((0,), dtype=torch.long),
    )
    _write_tensor_dataset(
        test_path,
        torch.randn(4, 3),
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )
    class_names_path.write_text(json.dumps({"0": "Normal", "1": "Attack"}), encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "evaluation": {"validation_data": str(validation_path)},
            "paths": {
                "closed_set_test_data": str(test_path),
                "class_names": str(class_names_path),
            },
            "training": {"batch_size": 2},
            "model": {"num_actions": 2},
        }
    )

    client = object.__new__(FlowerClient)
    client.cid = "1"
    client.cfg = cfg
    client.device = torch.device("cpu")
    client._move_data_to_device = False
    client.logger = logging.getLogger("test.client")
    client.eval_enabled = False
    client.eval_loader = None
    client.eval_class_names = []
    client.closed_set_data_path = None
    client.eval_dataset_role = "none"

    client._init_closed_set_evaluation()

    assert client.eval_enabled is True
    assert client.eval_dataset_role == "test"
    assert client.closed_set_data_path == test_path
    assert len(client.eval_loader.dataset) == 4


def test_fedavg_evaluate_aggregation_skips_all_zero_example_results():
    strategy = SaveModelFedAvg(cfg=OmegaConf.create({}))
    result = (
        SimpleNamespace(cid="1"),
        SimpleNamespace(num_examples=0, loss=0.0, metrics={}),
    )

    loss, metrics = strategy.aggregate_evaluate(server_round=1, results=[result], failures=[])

    assert loss is None
    assert metrics == {}
