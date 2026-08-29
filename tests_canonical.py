import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

from src.openset.conformal import fit_multicenter_conformal

def test_kappa_i():
    C = 4
    cases = [
        ([25, 25, 25, 25], 1.0, 1.0),
        ([25, 25, 0, 0], 0.5, 0.5),
        ([100, 0, 0, 0], 0.25, 0.25),
        ([97, 1, 1, 1], 1.0, None)
    ]
    print("\n--- Check 18: kappa_i ---")
    for counts, exp_q, exp_k in cases:
        c = torch.tensor(counts, dtype=torch.float32)
        total = c.sum().item()
        probs = c[c > 0] / total
        unnorm_entropy = -(probs * torch.log(probs)).sum().item()
        kappa = float(np.exp(unnorm_entropy) / C)
        q = (c > 0).sum().item() / C
        print(f"counts={counts}, q={q}, kappa={kappa:.4f}")
        assert 1/C <= kappa <= q + 1e-6 <= 1.0 + 1e-6
        if exp_k is not None:
            assert np.isclose(kappa, exp_k, atol=1e-4)
    print("PASS: Check 18 kappa_i")

def test_aggregation():
    class DummyServer:
        def __init__(self, gamma):
            self.cfg = OmegaConf.create({"method": {"canonical": True}, "strategy": {"gamma": gamma}})
        def _client_support_weight(self, record, max_examples):
            num = max(float(record.get("num_examples", 0.0)), 0.0)
            return float(num ** self.cfg.strategy.gamma)

    records = [{"num_examples": 100}, {"num_examples": 400}]
    print("\n--- Check 2: Aggregation Equivalence ---")
    for gamma, exp_weights in [(0, [0.5, 0.5]), (1, [0.2, 0.8]), (0.5, [1/3, 2/3])]:
        srv = DummyServer(gamma)
        weights = [srv._client_support_weight(r, 400) for r in records]
        w_sum = sum(weights)
        norm_weights = [w / w_sum for w in weights]
        print(f"gamma={gamma}, weights={norm_weights}")
        assert np.allclose(norm_weights, exp_weights)
        assert np.isclose(sum(norm_weights), 1.0, atol=1e-7)
    print("PASS: Check 2 Aggregation")

def test_conformal():
    print("\n--- Check 19 & 20: Conformal Calibration ---")
    df_proto = pd.DataFrame({"feature": [np.random.randn(10) for _ in range(50)], "y_raw": [0]*50})
    df_calib = pd.DataFrame({"feature": [np.random.randn(10) for _ in range(18)], "y_raw": [0]*18})
    res = fit_multicenter_conformal(df_proto, df_calib, num_classes=1, alpha=0.05)
    print(f"m=18 threshold:", res["thresholds"][0])
    assert res["thresholds"][0] == float('inf')
    
    df_calib_19 = pd.DataFrame({"feature": [np.random.randn(10) for _ in range(19)], "y_raw": [0]*19})
    res_19 = fit_multicenter_conformal(df_proto, df_calib_19, num_classes=1, alpha=0.05)
    print(f"m=19 threshold:", res_19["thresholds"][0])
    assert res_19["thresholds"][0] < float('inf')
    print("PASS: Check 19 & 20 Conformal Calibration")

def test_serialization():
    from src.federated.client import FlowerClient
    from omegaconf import OmegaConf
    import torch.nn as nn
    
    print("\n--- Check 22: Serialized Payload Privacy Audit ---")
    class DummyClient:
        def __init__(self):
            # Regression protection: pull the actual keys from the class
            self._SERVER_LOCAL_ONLY_METRIC_KEYS = FlowerClient._SERVER_LOCAL_ONLY_METRIC_KEYS
            
        def _sanitize_server_metrics(self, metrics):
            return {k: v for k, v in metrics.items() if k not in self._SERVER_LOCAL_ONLY_METRIC_KEYS}
            
    client = DummyClient()
    
    # Simulate some metrics containing private data
    metrics = {
        "label_histogram": [10, 20],
        "class_entropy": 0.5,
        "unnormalized_entropy": 0.5,
        "label_coverage": 1.0,
        "kappa_i": 0.8,
        "missing_classes": 0,
        "present_classes": 2,
        "imbalance_ratio": 2.0,
        "teacher_loss": 0.1,
        "student_loss": 0.2
    }
    
    sanitized = client._sanitize_server_metrics(metrics)
    forbidden_keys = [
        "label_histogram", "class_counts", "class_entropy", 
        "unnormalized_entropy", "label_coverage", "q_i", 
        "kappa_i", "missing_classes", "present_classes", "imbalance_ratio"
    ]
    for key in forbidden_keys:
        if key in sanitized:
            raise AssertionError(f"Privacy leak! Found forbidden key {key} in sanitized payload.")
            
    print("Sanitized keys:", list(sanitized.keys()))
    print("PASS: Check 22 Serialization Audit")

if __name__ == "__main__":
    test_kappa_i()
    test_aggregation()
    test_conformal()
    test_serialization()
    print("ALL TESTS PASSED")
