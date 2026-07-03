from pathlib import Path

from hydra import compose, initialize_config_dir

from src.federated.class_aware import (
    class_aware_aggregation_records,
    class_rarity_vector,
    parse_class_vector,
)
from src.federated.server import get_effective_num_rounds


def _config_dir() -> str:
    return str((Path(__file__).resolve().parents[1] / "src" / "configs").resolve())


def test_class_vector_parser_accepts_json_dicts():
    parsed = parse_class_vector('{"0": 5, "2": 7, "99": 1}', num_classes=4)

    assert parsed.tolist() == [5.0, 0.0, 7.0, 0.0]


def test_class_rarity_vector_emphasizes_minority_classes():
    rarity = class_rarity_vector(parse_class_vector('{"0": 1000, "1": 100, "2": 10}', 3), smoothing=1.0)

    assert rarity[2] > rarity[1] > rarity[0]


def test_fedmade_weights_boost_rare_class_clients():
    records = [
        {
            "cid": "majority",
            "num_examples": 1000,
            "quality": 0.8,
            "label_histogram": '{"0": 1000, "1": 0, "2": 0}',
        },
        {
            "cid": "minority",
            "num_examples": 100,
            "quality": 0.8,
            "label_histogram": '{"0": 0, "1": 90, "2": 10}',
        },
    ]

    weighted = class_aware_aggregation_records(
        records,
        num_classes=3,
        rare_class_strength=1.25,
        quality_weight_blend=0.0,
        cluster_balance_strength=0.0,
        min_multiplier=0.25,
        max_multiplier=3.0,
    )

    by_client = {record["cid"]: record for record in weighted}
    assert by_client["minority"]["class_multiplier"] > by_client["majority"]["class_multiplier"]
    assert by_client["minority"]["aggregation_weight"] > 0.0
    assert by_client["majority"]["aggregation_weight"] > 0.0


def test_fedmade_method_overlay_uses_single_round_strategy():
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config_fl", overrides=["experiment=exp3", "+method=fedmade"])

    assert cfg.experiment.method == "FedMADE"
    assert cfg.federated.strategy.name == "fedmade"
    assert cfg.federated.strategy.monitor_path.endswith("fedmade_monitoring.jsonl")
    assert get_effective_num_rounds(cfg) == cfg.federated.num_rounds
