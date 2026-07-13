import json

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from src.data.preprocessing import run_preprocessing


def _write_raw_dataset(path):
    labels = ["Normal", "BP", "DoS", "MitM"]
    rows = []
    for class_index, label in enumerate(labels):
        for sample_index in range(32):
            rows.append(
                {
                    "feature_a": class_index * 10 + sample_index,
                    "feature_b": sample_index / 10,
                    "service": f"svc_{sample_index % 3}",
                    "label": label,
                }
            )
    for sample_index in range(8):
        rows.append(
            {
                "feature_a": 100 + sample_index,
                "feature_b": sample_index / 5,
                "service": "fot_svc",
                "label": "FoT",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _cfg(raw_path, output_dir, num_clients):
    return OmegaConf.create(
        {
            "seed": 123,
            "dataset": {
                "name": "synthetic",
                "preprocessing": {
                    "output_dir": str(output_dir),
                    "raw_file": str(raw_path),
                    "label_column": "label",
                    "known_labels": ["Normal", "BP", "DoS", "MitM"],
                    "numerical_cols": None,
                    "categorical_cols": None,
                    "numeric_threshold": 0.9,
                    "validation_split": 0.1,
                    "closed_set_test_size": 0.2,
                    "num_clients": num_clients,
                    "alpha": 0.5,
                    "iid": False,
                    "unknown_label_id": -1,
                },
            },
        }
    )


@pytest.mark.parametrize("num_clients", [3, 10, 20])
def test_preprocessing_writes_non_empty_tensor_for_each_client(tmp_path, num_clients):
    raw_path = tmp_path / "raw.csv"
    output_dir = tmp_path / "processed"
    _write_raw_dataset(raw_path)

    metadata = run_preprocessing(
        _cfg(raw_path, output_dir, num_clients),
        project_root=tmp_path,
    )

    assert metadata["num_clients"] == num_clients
    assert metadata["known_labels"] == ["Normal", "BP", "DoS", "MitM"]
    assert metadata["num_actions"] == 4
    assert metadata["state_dim"] > 0
    assert not (output_dir / f"client_{num_clients + 1}_train.pt").exists()

    for client_id in range(1, num_clients + 1):
        client_path = output_dir / f"client_{client_id}_train.pt"
        assert client_path.exists()
        data = torch.load(client_path, map_location="cpu", weights_only=True)
        assert data["features"].ndim == 2
        assert data["labels"].ndim == 1
        assert data["labels"].numel() > 0

    distribution = pd.read_csv(output_dir / "client_class_distribution.csv")
    assert distribution["client_id"].tolist() == list(range(1, num_clients + 1))

    manifest_client_ids = {
        json.loads(line)["client_id"]
        for line in (output_dir / "partition_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert manifest_client_ids == set(range(1, num_clients + 1))


def test_preprocessing_can_skip_validation_split(tmp_path):
    raw_path = tmp_path / "raw.csv"
    output_dir = tmp_path / "processed"
    _write_raw_dataset(raw_path)
    cfg = _cfg(raw_path, output_dir, num_clients=3)
    cfg.dataset.preprocessing.closed_set_test_size = 0.1
    cfg.dataset.preprocessing.validation_split = 0.0

    metadata = run_preprocessing(cfg, project_root=tmp_path)

    known_train = torch.load(output_dir / "known_train.pt", map_location="cpu", weights_only=True)
    validation = torch.load(output_dir / "validation.pt", map_location="cpu", weights_only=True)
    closed_test = torch.load(
        output_dir / "closed_set_test.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert known_train["labels"].numel() == metadata["num_train_samples"] == 115
    assert validation["labels"].numel() == metadata["num_validation_samples"] == 0
    assert closed_test["labels"].numel() == metadata["num_closed_test_samples"] == 13


def test_external_protocol_caps_classes_balances_unknowns_and_writes_manifests(tmp_path):
    raw_path = tmp_path / "external.csv"
    output_dir = tmp_path / "processed"
    rows = []
    source_labels = ["KnownA", "KnownB", "KnownC", "KnownD", "UnknownA", "UnknownB"]
    for class_index, label in enumerate(source_labels):
        for sample_index in range(60):
            rows.append(
                {
                    "numeric": (
                        float("inf")
                        if sample_index == 0
                        else (None if sample_index == 1 else class_index + sample_index / 100)
                    ),
                    "service": "unknown_only" if label.startswith("Unknown") else "known",
                    "label": label,
                }
            )
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "dataset": {
                "name": "external-fixture",
                "preprocessing": {
                    "output_dir": str(output_dir),
                    "raw_file": str(raw_path),
                    "label_column": "label",
                    "source_labels": source_labels,
                    "known_labels": source_labels[:4],
                    "unknown_labels": source_labels[4:],
                    "require_all_source_labels": True,
                    "drop_duplicates": True,
                    "max_samples_per_class": 40,
                    "numerical_cols": None,
                    "categorical_cols": None,
                    "numeric_threshold": 0.9,
                    "categorical_schema_scope": "known_train",
                    "validation_split": 0.125,
                    "closed_set_test_size": 0.2,
                    "num_clients": 4,
                    "min_samples_per_client": 10,
                    "partition_max_attempts": 200,
                    "max_unknown_test_ratio": 1.0,
                    "alpha": 0.5,
                    "iid": False,
                    "unknown_label_id": -1,
                },
            },
        }
    )

    metadata = run_preprocessing(cfg, project_root=tmp_path)

    assert metadata["source_class_counts"] == {label: 60 for label in source_labels}
    assert metadata["experiment_class_counts"] == {label: 40 for label in source_labels}
    assert metadata["num_train_samples"] == 112
    assert metadata["num_validation_samples"] == 16
    assert metadata["num_closed_test_samples"] == 32
    assert metadata["num_unknown_samples"] == 32
    assert metadata["num_open_test_samples"] == 64
    assert "__UNK__" in metadata["categorical_categories"][0]

    split_manifest = pd.read_csv(output_dir / "split_manifest.csv")
    assert set(split_manifest["split"]) == {
        "known_train",
        "validation",
        "known_test",
        "unknown_test",
    }
    assert split_manifest["source_index"].is_unique
    assert (output_dir / "class_support.csv").exists()
    assert (output_dir / "feature_schema.json").exists()
    assert (output_dir / "numeric_imputer.joblib").exists()

    open_test = torch.load(output_dir / "open_set_test.pt", map_location="cpu", weights_only=True)
    assert torch.isfinite(open_test["features"]).all()
    assert int((open_test["labels"] == -1).sum()) == 32
    for client_id in range(1, 5):
        client = torch.load(
            output_dir / f"client_{client_id}_train.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert client["labels"].numel() >= 10
