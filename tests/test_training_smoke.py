import torch
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.agents.policy import EpsilonGreedyPolicy, EpsilonScheduler
<<<<<<< HEAD
from src.models.models import OpenSetQChainModelFactory
=======
from src.models import CVAEQChainModelAdapter, validate_tabular_output
from src.models.cvae_dqn import OpenSetQChainModelFactory
>>>>>>> ea28efe (Initial commit with updated source code)
from src.rl.environment import BlockchainIntrusionEnv
from src.rl.local_training import run_local_training_round
from src.rl.replay_buffer import ExperienceReplayBuffer


def test_minimal_local_training_smoke(tmp_path):
    path = tmp_path / "client.pt"
    torch.save(
        {
            "features": torch.randn(8, 5),
            "labels": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        },
        path,
    )
<<<<<<< HEAD
    model_cfg = OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 2})
=======
    model_cfg = OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 2, "backbone": {"hidden_dim": 8, "depth": 1, "ensemble_size": 1, "dropout": 0.0, "expansion": 1, "q_hidden_dim": 8, "q_depth": 1, "q_ensemble_size": 1}, "generator": {"hidden_dim": 8, "depth": 1, "dropout": 0.0, "expansion": 1}})
>>>>>>> ea28efe (Initial commit with updated source code)
    training_cfg = OmegaConf.create(
        {
            "seed": 7,
            "local_episodes_per_round": 1,
            "steps_per_episode": 3,
            "replay_buffer_size": 20,
            "min_buffer_size": 2,
            "batch_size": 2,
            "gamma": 0.7,
            "lr_prior": 1e-3,
            "lr_q_rl": 1e-3,
            "tau": 0.01,
            "target_update_freq": 2,
            "use_double_dqn": True,
            "prior_grad_clip_norm": 1.0,
<<<<<<< HEAD
            "prior_kl_raw": False,
=======
            "q_grad_clip_norm": 1.0,
            "prior_kl_raw": False,
            "loss_weights": {
                "prior_kl": 1.0,
                "q_td": 1.0,
                "classification": 1.0,
                "generator_reconstruction": 1.0,
                "proximal": 1.0,
            },
>>>>>>> ea28efe (Initial commit with updated source code)
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_rate": 1.0,
        }
    )

    device = torch.device("cpu")
    agent = Agent(OpenSetQChainModelFactory(model_cfg), training_cfg, device=device)
    env = BlockchainIntrusionEnv(str(path), 3, device=device, global_num_actions=2)
    buffer = ExperienceReplayBuffer(20)
    policy = EpsilonGreedyPolicy(agent.prior_net, agent.value_net_main, 2, device)
    scheduler = EpsilonScheduler(training_cfg)

    steps, metrics = run_local_training_round(
        agent=agent,
        env=env,
        buffer=buffer,
        policy=policy,
        epsilon_scheduler=scheduler,
        cfg_training=training_cfg,
        device=device,
    )

    assert steps == 3
    assert "avg_td_loss" in metrics
<<<<<<< HEAD
=======
    assert "avg_total_loss" in metrics
    assert "avg_classification_loss" in metrics
    assert "gradient_norm_q" in metrics
    assert "learning_rate_q_rl" in metrics
    assert "reward_mean" in metrics


def test_cvae_dqn_adapter_exposes_shared_output_contract():
    model_cfg = OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 2, "backbone": {"hidden_dim": 8, "depth": 1, "ensemble_size": 1, "dropout": 0.0, "expansion": 1, "q_hidden_dim": 8, "q_depth": 1, "q_ensemble_size": 1}, "generator": {"hidden_dim": 8, "depth": 1, "dropout": 0.0, "expansion": 1}})
    training_cfg = OmegaConf.create(
        {
            "seed": 7,
            "gamma": 0.7,
            "lr_prior": 1e-3,
            "lr_q_rl": 1e-3,
            "tau": 0.01,
            "use_double_dqn": True,
            "prior_kl_raw": False,
            "prior_grad_clip_norm": 1.0,
            "q_grad_clip_norm": 1.0,
            "loss_weights": {
                "prior_kl": 1.0,
                "q_td": 1.0,
                "classification": 1.0,
                "generator_reconstruction": 1.0,
                "proximal": 1.0,
            },
        }
    )
    device = torch.device("cpu")
    agent = Agent(OpenSetQChainModelFactory(model_cfg), training_cfg, device=device)
    adapter = CVAEQChainModelAdapter(
        prior_net=agent.prior_net,
        value_net_main=agent.value_net_main,
        recognition_net=agent.recognition_net,
        generation_net=agent.generation_net,
    )

    output = adapter(torch.randn(4, 5))

    validate_tabular_output(output, batch_size=4, num_classes=2)
    assert output["q_values"] is output["logits"]
    assert output["mu"] is output["features"]
>>>>>>> ea28efe (Initial commit with updated source code)
