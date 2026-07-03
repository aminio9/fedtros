import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.checkpointing.checkpoints import (
    CheckpointState,
    load_agent_checkpoint,
    save_agent_checkpoint,
)
from src.models.cvae_dqn import OpenSetQChainModelFactory
from src.rl.environment import BlockchainIntrusionEnv


def _model_cfg():
    return OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 2})


def _training_cfg():
    return OmegaConf.create(
        {
            "gamma": 0.7,
            "use_double_dqn": True,
            "lr_prior": 1e-3,
            "lr_q_rl": 1e-3,
            "prior_grad_clip_norm": 1.0,
            "q_grad_clip_norm": 1.0,
            "prior_kl_raw": False,
            "loss_weights": {
                "prior_kl": 1.0,
                "q_td": 1.0,
                "classification": 1.0,
                "generator_reconstruction": 1.0,
                "proximal": 1.0,
            },
            "classification_loss": {
                "name": "focal",
                "focal_gamma": 2.0,
                "use_class_weights": True,
            },
            "kl": {
                "free_nats": 0.25,
                "warmup_steps": 10,
            },
            "optimizer_name": "adamw",
            "optimizer_weight_decay": 1e-4,
            "optimizer_betas": [0.9, 0.95],
            "optimizer_eps": 1e-8,
        }
    )


def _batch(batch_size: int = 4):
    return (
        torch.randn(batch_size, 5),
        torch.tensor([[0], [1], [0], [1]], dtype=torch.long)[:batch_size],
        torch.ones(batch_size, 1),
        torch.randn(batch_size, 5),
        torch.zeros(batch_size, 1),
        torch.tensor([[0], [1], [0], [1]], dtype=torch.long)[:batch_size],
    )


def test_train_step_returns_explicit_loss_dictionary_and_covers_optimizers():
    agent = Agent(OpenSetQChainModelFactory(_model_cfg()), _training_cfg(), torch.device("cpu"))

    prior_param_ids = {
        id(param) for group in agent.optimizer_prior.param_groups for param in group["params"]
    }
    q_param_ids = {
        id(param) for group in agent.optimizer_q_rl.param_groups for param in group["params"]
    }

    assert prior_param_ids == {id(param) for param in agent.prior_net.parameters()}
    assert q_param_ids == {
        id(param)
        for module in (agent.recognition_net, agent.value_net_main)
        for param in module.parameters()
    }
    assert not q_param_ids.intersection({id(param) for param in agent.value_net_target.parameters()})
    assert isinstance(agent.optimizer_prior, torch.optim.AdamW)
    assert isinstance(agent.optimizer_q_rl, torch.optim.AdamW)

    metrics = agent.train_step(_batch())

    for key in (
        "loss/total",
        "loss/prior_kl",
        "loss/q_td",
        "loss/classification",
        "loss/prior_kl_raw",
        "loss/prior_kl_warmup",
        "gradient/prior_norm",
        "gradient/q_norm",
        "lr/prior",
        "lr/q_rl",
    ):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))
    assert metrics["loss/prior_kl_warmup"] == 0.1


class _AlwaysZeroQ(nn.Module):
    def forward(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(states.size(0), 2, device=states.device)
        logits[:, 0] = 10.0
        logits[:, 1] = -10.0
        return logits


def test_generator_training_uses_smooth_l1_contract():
    agent = Agent(OpenSetQChainModelFactory(_model_cfg()), _training_cfg(), torch.device("cpu"))
    agent.value_net_main = _AlwaysZeroQ()
    features = torch.randn(8, 5)
    labels = torch.zeros(8, dtype=torch.long)
    generator_cfg = OmegaConf.create(
        {
            "batch_size": 4,
            "lr": 1e-3,
            "rounds": 1,
            "epochs_per_round": 1,
            "min_correct_samples": 1,
            "reconstruction_loss": "smooth_l1",
            "reconstruction_beta": 1.0,
            "grad_clip_norm": 1.0,
        }
    )

    metrics = agent.train_generation_network(features, labels, generator_cfg)

    assert metrics["generator_samples"] == 8.0
    assert torch.isfinite(torch.tensor(metrics["generator_reconstruction_loss"]))
    assert torch.isfinite(torch.tensor(metrics["generator_grad_norm"]))


def test_soft_target_update_moves_target_toward_main():
    agent = Agent(OpenSetQChainModelFactory(_model_cfg()), _training_cfg(), torch.device("cpu"))
    target_before = [param.detach().clone() for param in agent.value_net_target.parameters()]

    with torch.no_grad():
        for param in agent.value_net_main.parameters():
            param.add_(0.25)

    agent.update_target_network(tau=0.5)

    assert any(
        not torch.allclose(before, after)
        for before, after in zip(target_before, agent.value_net_target.parameters(), strict=True)
    )


def test_checkpoint_save_load_preserves_core_networks(tmp_path):
    agent = Agent(OpenSetQChainModelFactory(_model_cfg()), _training_cfg(), torch.device("cpu"))
    agent.train_step(_batch())
    path = tmp_path / "checkpoint.pt"
    cfg = OmegaConf.create({"checkpointing": {"include_rng_state": False}})
    save_agent_checkpoint(
        agent,
        cfg,
        path,
        CheckpointState(epoch=1, global_step=1, metrics={"loss/total": 1.0}),
    )

    reloaded = Agent(OpenSetQChainModelFactory(_model_cfg()), _training_cfg(), torch.device("cpu"))
    load_agent_checkpoint(reloaded, path, torch.device("cpu"), load_optimizers=False)

    for original, restored in zip(
        agent.value_net_main.parameters(),
        reloaded.value_net_main.parameters(),
        strict=True,
    ):
        assert torch.allclose(original, restored)


def test_class_balanced_reward_increases_minority_correct_reward(tmp_path):
    path = tmp_path / "imbalanced.pt"
    torch.save(
        {
            "features": torch.randn(11, 5),
            "labels": torch.tensor([0] * 10 + [1], dtype=torch.long),
        },
        path,
    )
    env = BlockchainIntrusionEnv(
        str(path),
        steps_per_episode=1,
        device=torch.device("cpu"),
        global_num_actions=2,
        class_balanced_rewards=True,
    )

    assert env._class_reward_weights[1] > env._class_reward_weights[0]
