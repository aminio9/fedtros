from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir


def _config_dir() -> str:
    return str((Path(__file__).resolve().parents[1] / "src" / "configs").resolve())


def test_bnat_config_points_to_canonical_csv():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl")

    assert cfg.dataset.name == "B-NAT"
    assert cfg.dataset.raw_path == "data/raw/BNaT.csv"
    assert cfg.dataset.source_url == "https://github.com/avitech-vnu/BNaT"
    assert cfg.dataset.source_labels == ["Normal", "BP", "DoS", "MitM", "FoT"]
    assert cfg.dataset.known_labels == ["Normal", "BP", "DoS", "MitM"]
    assert cfg.model.num_classes == 4
    assert cfg.model.feature_dim == 31


def test_bnat_csv_contains_expected_source_labels():
    csv_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "BNaT.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert df.shape == (210000, 22)
    assert set(df["label"].unique()) == {"Normal", "BP", "DoS", "MitM", "FoT"}
    counts = df["label"].value_counts().to_dict()
    assert counts == {
        "Normal": 150000,
        "MitM": 15000,
        "BP": 15000,
        "DoS": 15000,
        "FoT": 15000,
    }


def test_exp5_config_is_external_validation_ready():
    from src.infrastructure.study import load_study_config
    cfg = load_study_config("E5-DATASET", Path(__file__).resolve().parents[1])

    assert cfg["study_id"] == "E5-DATASET"
    assert cfg["name"] == "E5_datasetwise"
    assert cfg["num_clients"] == 10
    assert cfg["base_overrides"]["federated.num_rounds"] == 100
    assert cfg["base_overrides"]["evaluation.mode"] == "open_set"
    assert "bnat" in cfg["datasets"]
    assert "btat" in cfg["datasets"]


def test_external_dataset_aliases_normalize_to_registry_ids():
    from src.utils.config import _canonical_dataset_id

    assert _canonical_dataset_id("B-NAT") == "bnat"
    assert _canonical_dataset_id("B-TAT") == "btat"
    assert _canonical_dataset_id("CIC-IDS2017") == "cicids2017"
    assert _canonical_dataset_id("ToN-IoT") == "toniot"


def test_exp8_config_is_labelwise_open_set():
    from src.infrastructure.study import load_study_config
    cfg = load_study_config("E8-LOAO", Path(__file__).resolve().parents[1])

    assert cfg["study_id"] == "E8-LOAO"
    assert cfg["name"] == "E8_leave_one_attack_out"
    assert cfg["num_clients"] == 10
    assert "Normal" in cfg["known_labels_by_unknown"]["FoT"]
    assert cfg["base_overrides"]["evaluation.mode"] == "open_set"
