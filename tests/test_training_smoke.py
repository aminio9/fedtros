import torch
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.agents.policy import EpsilonGreedyPolicy, EpsilonScheduler
from src.models.models import OpenSetQChainModelFactory
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
    model_cfg = OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 2})
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
            "prior_kl_raw": False,
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
