from pathlib import Path

from omegaconf import OmegaConf

from src.evaluation.run import _resolve_evaluation_checkpoint


def test_evaluation_prefers_best_checkpoint_when_enabled(tmp_path):
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    latest.write_text("latest")
    best.write_text("best")
    cfg = OmegaConf.create(
        {
            "evaluation": {"checkpoint_path": str(latest), "use_best_checkpoint": True},
            "checkpointing": {"best_model_path": str(best)},
        }
    )

    assert _resolve_evaluation_checkpoint(cfg, Path(".")) == best


def test_evaluation_can_use_latest_when_best_disabled(tmp_path):
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    latest.write_text("latest")
    best.write_text("best")
    cfg = OmegaConf.create(
        {
            "evaluation": {"checkpoint_path": str(latest), "use_best_checkpoint": False},
            "checkpointing": {"best_model_path": str(best)},
        }
    )

    assert _resolve_evaluation_checkpoint(cfg, Path(".")) == latest
