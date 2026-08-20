"""Tests for Schema v2 checkpointing and legacy rejection in FedTROS-PR."""

from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf

from src.models.bundle import FedTROSModelBundle as Agent
from src.checkpointing.checkpoints import (
    CheckpointState,
    IncompatibleCheckpointError,
    load_agent_checkpoint,
    save_agent_checkpoint,
)
from src.models.models import ModelFactory


@pytest.fixture
def agent():
    model_cfg = OmegaConf.create({"feature_dim": 10, "latent_dim": 8, "num_classes": 4})
    train_cfg = OmegaConf.create(
        {
            "teacher_lr": 1e-3,
            "student_lr": 1e-3,
            "teacher_beta_kl": 0.01,
            "lambda_kd_init": 0.20,
            "lambda_align_init": 0.08,
            "fedtros_global_anchor_weight": 2.0,
        }
    )
    return Agent(ModelFactory(model_cfg), train_cfg, device=torch.device("cpu"))


def test_schema_v2_save_and_load(tmp_path, agent):
    cfg = OmegaConf.create(
        {
            "checkpointing": {"include_rng_state": False},
            "model": {"feature_dim": 10, "latent_dim": 8, "num_classes": 4},
        }
    )
    state = CheckpointState(epoch=3, global_step=150, metrics={"accuracy": 0.85}, best_metric=0.85)
    ckpt_path = tmp_path / "model_schema_v2.pt"

    save_agent_checkpoint(agent, cfg, ckpt_path, state)
    assert ckpt_path.exists()

    loaded = load_agent_checkpoint(agent, ckpt_path, device=torch.device("cpu"))
    assert loaded["schema_version"] == 2
    assert loaded["method"] == "FedTROS-PR"
    assert loaded["teacher_type"] == "variational_classifier"
    assert "student_model" in loaded


def test_legacy_dqn_checkpoint_raises_error(tmp_path, agent):
    legacy_ckpt_path = tmp_path / "legacy_dqn_model.pt"
    legacy_payload = {
        "prior_net": {},
        "recognition_net": {},
        "value_net_main": {},
        "epoch": 5,
    }
    torch.save(legacy_payload, legacy_ckpt_path)

    with pytest.raises(IncompatibleCheckpointError) as exc_info:
        load_agent_checkpoint(agent, legacy_ckpt_path, device=torch.device("cpu"))

    assert "legacy DQN/RL checkpoint" in str(exc_info.value)
