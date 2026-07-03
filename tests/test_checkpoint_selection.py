from omegaconf import OmegaConf

from src.checkpointing.checkpoints import (
    CheckpointState,
    build_checkpoint_metadata,
    metric_improved,
    select_checkpoint_metric,
)


def test_checkpoint_metric_selection_accepts_validation_metrics_only():
    assert (
        select_checkpoint_metric({"train/accuracy": 0.99}, monitor_metric="train/accuracy")
        is None
    )
    assert (
        select_checkpoint_metric({"test/macro_f1": 0.99}, monitor_metric="test/macro_f1")
        is None
    )

    selected = select_checkpoint_metric({"val/macro_f1": 0.41}, monitor_metric="val/macro_f1")

    assert selected == ("val/macro_f1", 0.41)


def test_checkpoint_metric_history_selects_best_validation_epoch():
    best = None
    best_epoch = None
    for epoch, value in enumerate([0.20, 0.35, 0.31], start=1):
        selected = select_checkpoint_metric(
            {"val/macro_f1": value},
            monitor_metric="val/macro_f1",
        )
        assert selected is not None
        if metric_improved(selected[1], best, mode="max"):
            best = selected[1]
            best_epoch = epoch

    assert best == 0.35
    assert best_epoch == 2


def test_combined_validation_score_uses_only_allowed_components():
    selected = select_checkpoint_metric(
        {
            "val/macro_f1": 0.25,
            "val/balanced_accuracy": 0.75,
            "train/accuracy": 1.0,
        },
        monitor_metric="combined_validation_score",
    )

    assert selected == ("combined_validation_score", 0.5)


def test_checkpoint_metadata_contains_reproducibility_fields(tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 123,
            "dataset": {
                "source_labels": ["Normal", "Attack"],
                "preprocessing": {"known_labels": ["Normal"]},
            },
        }
    )
    metadata = build_checkpoint_metadata(
        cfg,
        CheckpointState(epoch=2, global_step=10, metrics={"val/macro_f1": 0.4}),
        checkpoint_path=tmp_path / "best_model.pt",
        selected_metric_name="val/macro_f1",
        selected_metric_value=0.4,
    )

    assert metadata["epoch"] == 2
    assert metadata["selected_metric_name"] == "val/macro_f1"
    assert metadata["seed"] == 123
    assert metadata["known_labels"] == ["Normal"]
    assert metadata["unknown_labels"] == ["Attack"]
    assert len(metadata["config_hash"]) == 64
