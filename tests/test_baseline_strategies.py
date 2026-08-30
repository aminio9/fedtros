import torch
from omegaconf import OmegaConf

from src.training.local_only_student import run_local_only_training
from src.training.scaffold_student import decode_control_variate, encode_control_variate


def test_scaffold_control_variate_roundtrip():
    source = {"layer.weight": torch.tensor([[1.0, -2.0]]), "layer.bias": torch.tensor([0.5])}
    restored = decode_control_variate(encode_control_variate(source))
    assert set(restored) == set(source)
    for key in source:
        assert torch.equal(restored[key], source[key])


def test_local_only_training_executes_one_configured_local_round(monkeypatch):
    calls = []

    def fake_round(**kwargs):
        calls.append(kwargs)
        return 3, {"train_loss": 0.25}

    monkeypatch.setattr("src.training.local_only_student.run_local_training_round", fake_round)
    cfg = OmegaConf.create({"local_epochs": 50})
    metrics = run_local_only_training(
        agent=object(),
        features=torch.zeros(2, 3),
        labels=torch.zeros(2, dtype=torch.long),
        cfg_training=cfg,
        device=torch.device("cpu"),
        client_id="c0",
    )
    assert len(calls) == 1
    assert metrics["global_step"] == 3
    assert metrics["is_local_only_baseline"] == 1.0