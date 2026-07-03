from omegaconf import OmegaConf

from src.federated.server import FMRLAdaptiveVectorAlignedAggregationStrategy


def _strategy():
    strategy = object.__new__(FMRLAdaptiveVectorAlignedAggregationStrategy)
    strategy.cfg = OmegaConf.create({"model": {"num_actions": 5}})
    strategy.min_selected_clients = 9
    return strategy


def test_coverage_safe_select_keeps_at_least_ninety_percent_clients():
    strategy = _strategy()
    records = [
        {
            "cid": str(i),
            "utility": 1.0 - 0.01 * i,
            "td_error": 0.1 * i,
            "selected": False,
            "label_histogram": '{"0": 10, "1": 10, "2": 10, "3": 10, "4": 10}',
        }
        for i in range(10)
    ]

    selected = strategy._coverage_safe_select(records)

    assert len(selected) == 9
    assert sum(bool(record["selected"]) for record in records) == 9


def test_coverage_safe_select_does_not_drop_only_class_supporter():
    strategy = _strategy()
    records = []
    for i in range(9):
        records.append(
            {
                "cid": str(i),
                "utility": 1.0,
                "td_error": 0.0,
                "selected": False,
                "label_histogram": '{"0": 10, "1": 10, "2": 10, "3": 10, "4": 0}',
            }
        )
    records.append(
        {
            "cid": "only_class_4",
            "utility": 0.0,
            "td_error": 99.0,
            "selected": False,
            "label_histogram": '{"0": 0, "1": 0, "2": 0, "3": 0, "4": 2}',
        }
    )

    selected = strategy._coverage_safe_select(records)

    assert len(selected) >= 9
    assert "only_class_4" in {record["cid"] for record in selected}
    assert next(record for record in records if record["cid"] == "only_class_4")["selected"] is True
