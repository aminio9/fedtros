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
    assert cfg.model.num_actions == 4
    assert cfg.model.state_dim == 31


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
