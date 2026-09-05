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


def test_save_global_model_round_gating(tmp_path, agent):
    import src.federated.server as server_mod

    server_mod.GLOBAL_AGENT_REF = agent
    ckpt_dir = tmp_path / "checkpoints"
    params = agent.get_federated_parameters()

    cfg = OmegaConf.create(
        {
            "checkpointing": {
                "dir": str(ckpt_dir),
                "latest_checkpoint_path": str(ckpt_dir / "latest.pt"),
                "best_model_path": str(ckpt_dir / "best_model.pt"),
                "final_model_path": str(ckpt_dir / "final_model.pt"),
                "save_latest": True,
                "save_best": True,
                "save_final": True,
                "save_round_checkpoints": False,
                "checkpoint_interval": 0,
                "include_rng_state": False,
                "monitor_metric": "val/accuracy",
            },
            "federated": {"num_rounds": 2, "central_evaluate": {"enabled": False}},
            "model": {"feature_dim": 10, "latent_dim": 8, "num_classes": 4},
        }
    )

    # Round 1 with save_round_checkpoints=False:
    server_mod.save_global_model(params, 1, cfg, metrics={"val/accuracy": 0.8})
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "best_model.pt").exists()
    assert not (ckpt_dir / "global_model_round_0001.pt").exists()
    assert not (ckpt_dir / "final_model.pt").exists()

    # Round 2 (final round): final_model.pt should be saved
    server_mod.save_global_model(params, 2, cfg, metrics={"val/accuracy": 0.85})
    assert (ckpt_dir / "final_model.pt").exists()
    assert not (ckpt_dir / "global_model_round_0002.pt").exists()

    # With save_round_checkpoints=True: round file should be saved
    cfg.checkpointing.save_round_checkpoints = True
    server_mod.save_global_model(params, 3, cfg, metrics={"val/accuracy": 0.86})
    assert (ckpt_dir / "global_model_round_0003.pt").exists()

