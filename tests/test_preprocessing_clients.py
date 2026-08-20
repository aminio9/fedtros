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


def test_smoke_preprocessing_has_a_deterministic_per_class_cap(tmp_path):
    raw_path = tmp_path / "smoke.csv"
    output_dir = tmp_path / "processed"
    rows = [
        {"feature": sample, "label": label}
        for label in ("Normal", "BP", "DoS", "MitM", "FoT")
        for sample in range(120)
    ]
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    cfg = OmegaConf.create({
        "seed": 42,
        "dataset": {
            "name": "smoke-fixture",
            "preprocessing": {
                "output_dir": str(output_dir),
                "raw_file": str(raw_path),
                "label_column": "label",
                "source_labels": ["Normal", "BP", "DoS", "MitM", "FoT"],
                "known_labels": ["Normal", "BP", "DoS", "MitM"],
                "unknown_labels": ["FoT"],
                "numerical_cols": ["feature"],
                "categorical_cols": [],
                "numeric_threshold": 0.9,
                "categorical_schema_scope": "known_train",
                "validation_split": 0.1,
                "closed_set_test_size": 0.2,
                "num_clients": 2,
                "alpha": 0.5,
                "iid": False,
                "unknown_label_id": -1,
                "smoke": True,
            },
        },
    })

    metadata = run_preprocessing(cfg, project_root=tmp_path)

    assert metadata["experiment_class_counts"] == {
        "BP": 96,
        "DoS": 96,
        "FoT": 96,
        "MitM": 96,
        "Normal": 96,
    }
    assert metadata["num_known_samples"] == 384
    assert metadata["num_unknown_samples"] == 96
    assert metadata["min_samples_per_client"] == 8


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
    assert metadata["num_classes"] == 4
    assert metadata["feature_dim"] > 0
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


def test_paired_partition_is_reused_across_independent_run_output_dirs(tmp_path):
    """Matched methods must reuse one immutable client partition.

    The generated tensors live in different run-local directories, while the
    persisted paired-partition file is shared by scientific condition.  The
    order of manifest rows is not part of the contract, so compare the exact
    sample -> client assignment after sorting.
    """
    raw_path = tmp_path / "raw.csv"
    _write_raw_dataset(raw_path)
    paired_partition = tmp_path / "partitions" / "synthetic_a05_fot_seed123.json"

    first_cfg = _cfg(raw_path, tmp_path / "run_a" / "data", num_clients=4)
    first_cfg.dataset.preprocessing.unknown_labels = ["FoT"]
    first_cfg.dataset.partition_file = str(paired_partition)
    run_preprocessing(first_cfg, project_root=tmp_path)

    second_cfg = _cfg(raw_path, tmp_path / "run_b" / "data", num_clients=4)
    second_cfg.dataset.preprocessing.unknown_labels = ["FoT"]
    second_cfg.dataset.partition_file = str(paired_partition)
    run_preprocessing(second_cfg, project_root=tmp_path)

    assert paired_partition.exists()
    payload = json.loads(paired_partition.read_text(encoding="utf-8"))
    assert payload["schema_name"] == "fedtros_paired_partition"
    assert payload["schema_version"] == 1
    assert payload["unknown_labels"] == ["FoT"]

    first = pd.read_json(tmp_path / "run_a" / "data" / "partition_manifest.jsonl", lines=True)
    second = pd.read_json(tmp_path / "run_b" / "data" / "partition_manifest.jsonl", lines=True)
    cols = ["sample_index", "client_id", "label", "label_name"]
    first = first[cols].sort_values(["sample_index", "client_id"]).reset_index(drop=True)
    second = second[cols].sort_values(["sample_index", "client_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_paired_partition_rejects_unknown_protocol_mismatch(tmp_path):
    """The same partition file cannot be silently reused for another OSR protocol."""
    raw_path = tmp_path / "raw.csv"
    _write_raw_dataset(raw_path)
    paired_partition = tmp_path / "partitions" / "shared.json"

    open_cfg = _cfg(raw_path, tmp_path / "open" / "data", num_clients=4)
    open_cfg.dataset.preprocessing.unknown_labels = ["FoT"]
    open_cfg.dataset.partition_file = str(paired_partition)
    run_preprocessing(open_cfg, project_root=tmp_path)

    closed_cfg = _cfg(raw_path, tmp_path / "closed" / "data", num_clients=4)
    closed_cfg.dataset.preprocessing.unknown_labels = []
    closed_cfg.dataset.partition_file = str(paired_partition)
    with pytest.raises(ValueError, match="Paired partition mismatch for unknown_labels"):
        run_preprocessing(closed_cfg, project_root=tmp_path)
