import pytest
import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.student import StudentIDSModel
from src.federated.server import FedTROSStrategy

def test_canonical_prototype_representation():
    model = StudentIDSModel(
        input_dim=10,
        num_classes=5,
        hidden_dims=[512, 256, 128],
        osr_enabled=False
    )
    assert not model.osr_enabled
    assert model.osr_encoder is None
    
    x = torch.randn(4, 10)
    features, logits = model(x)
    assert features.shape == (4, 128)
    assert logits.shape == (4, 5)

def test_canonical_aggregation_weights():
    cfg = OmegaConf.create({
        "strategy": {"support_min_weight": 0.01},
    })
    
    class MockServer:
        def __init__(self, cfg):
            self.cfg = cfg
            
        def _client_support_weight(self, record, max_examples):
            num_examples = max(float(record.get("num_examples", 0.0)), 0.0)
            sample_factor = float(np.sqrt(num_examples / max(float(max_examples), 1.0)))
            min_weight = float(OmegaConf.select(self.cfg, "strategy.support_min_weight", default=0.01))
            return float(np.clip(sample_factor, min_weight, 1.0))
            
    server = MockServer(cfg)
    records = [
        {"num_examples": 100},
        {"num_examples": 400},
        {"num_examples": 1000}
    ]
    max_examples = max(r["num_examples"] for r in records)
    weights = [server._client_support_weight(r, max_examples) for r in records]
    
    assert np.isclose(weights[0], np.sqrt(0.1))
    assert np.isclose(weights[1], np.sqrt(0.4))
    assert np.isclose(weights[2], 1.0)
    assert server._client_support_weight({"num_examples": 0}, 1000) == 0.01

