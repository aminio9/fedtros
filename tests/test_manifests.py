"""Test provenance and run manifest generation for FedTROS-PR."""

import json
from pathlib import Path
from omegaconf import OmegaConf

from src.artifacts.manifests import (
    compute_dict_hash,
    compute_file_hash,
    create_run_manifest,
    get_hardware_info,
)


def test_compute_hashes(tmp_path):
    sample_file = tmp_path / "test.txt"
    sample_file.write_text("FedTROS-PR test content\n", encoding="utf-8")

    file_hash = compute_file_hash(sample_file)
    assert len(file_hash) == 64

    data = {"method": "FedTROS-PR", "seed": 42}
    dict_hash = compute_dict_hash(data)
    assert len(dict_hash) == 64


def test_create_run_manifest(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {"method": "FedTROS-PR", "name": "exp1"},
            "dataset": {"name": "b_nat", "preprocessing": {"known_labels": ["Normal", "DoS"], "unknown_labels": ["MitM"], "alpha": 0.1}},
            "federated": {"num_clients": 5},
            "seed": 42,
        }
    )

    manifest = create_run_manifest(
        cfg,
        output_dir=tmp_path,
        project_root=tmp_path,
        metrics={"test/accuracy": 0.95},
    )

    manifest_file = tmp_path / "run_manifest.json"
    assert manifest_file.exists()

    loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert loaded["method"] == "FedTROS-PR"
    assert loaded["schema_version"] == 2
    assert loaded["num_clients"] == 5
    assert "hardware" in loaded
    assert "timestamp_utc" in loaded
