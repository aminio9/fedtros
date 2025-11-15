import logging
from typing import Any, Dict, List, Tuple, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import Parameters, parameters_to_ndarrays
from omegaconf import DictConfig

# get_device helper
try:
    from .utils import get_device
except ImportError:  # pragma: no cover - standalone usage
    from utils import get_device

try:
    from .agent import Agent
    from .environment import BlockchainIntrusionEnv
    from .exceptions import ConfigMismatchError
    from .local_training import run_local_training_round
    from .models import OpenSetQChainModelFactory
    from .policy import EpsilonGreedyPolicy, EpsilonScheduler
    from .replay_buffer import ExperienceReplayBuffer
except ImportError:  # pragma: no cover - standalone usage
    from agent import Agent
    from environment import BlockchainIntrusionEnv
    from exceptions import ConfigMismatchError
    from local_training import run_local_training_round
    from models import OpenSetQChainModelFactory
    from policy import EpsilonGreedyPolicy, EpsilonScheduler
    from replay_buffer import ExperienceReplayBuffer

logger = logging.getLogger("Client")


class FlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient implementing the Fed-Per agent."""

    def __init__(self, cid: str, cfg: DictConfig, data_path: str):
        self.cid = cid
        self.cfg = cfg
        self.data_path = data_path
        self.device = get_device()

        logger.info("Client %s: Initializing...", cid)

        self.model_factory = OpenSetQChainModelFactory(cfg.model)
        self.env = BlockchainIntrusionEnv(
            processed_data_path=self.data_path,
            steps_per_episode=cfg.training.steps_per_episode,
        )

        if (
            cfg.model.state_dim != self.env.feature_dim
            or cfg.model.num_actions != self.env.num_actions_nt
        ):
            raise ConfigMismatchError(
                f"Config/Env mismatch on client {cid}. "
                f"Config (s:{cfg.model.state_dim}, a:{cfg.model.num_actions}), "
                f"Env (s:{self.env.feature_dim}, a:{self.env.num_actions_nt}). "
            "Ensure 'env_metadata' in 'config_fl.yaml' matches your processed data."
            )

        self.agent = Agent(self.model_factory, cfg.training, self.device)
        self.buffer = ExperienceReplayBuffer(cfg.training.replay_buffer_size)
        self.policy = EpsilonGreedyPolicy(
            self.agent.prior_net, self.agent.value_net_main, cfg.model.num_actions, self.device
        )
        self.epsilon_scheduler = EpsilonScheduler(cfg.training)

        logger.info("Client %s: Initialization complete.", cid)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        logger.debug("Client %s: get_parameters called", self.cid)
        return self.agent.get_federated_parameters()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        logger.debug("Client %s: set_parameters called", self.cid)
        self.agent.set_federated_parameters(parameters, hard_target_update=True)

    def fit(
        self, parameters: Union[List[np.ndarray], Parameters], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        round_num = config.get("server_round", "?")
        logger.info("Client %s: fit() called for round %s", self.cid, round_num)

        param_list = (
            parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
        )
        self.set_parameters(param_list)

        num_steps_trained, metrics = run_local_training_round(
            agent=self.agent,
            env=self.env,
            buffer=self.buffer,
            policy=self.policy,
            epsilon_scheduler=self.epsilon_scheduler,
            cfg_training=self.cfg.training,
            device=self.device,
        )

        updated_parameters = self.get_parameters(config={})
        return updated_parameters, num_steps_trained, metrics

    def evaluate(
        self, parameters: Union[List[np.ndarray], Parameters], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        round_num = config.get("server_round", "?")
        logger.info("Client %s: evaluate() called for round %s", self.cid, round_num)
        logger.warning("Evaluate() is not implemented. Returning placeholder values.")

        param_list = (
            parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
        )
        self.set_parameters(param_list)

        loss = 0.0
        accuracy = 0.0
        num_examples = 1
        return loss, num_examples, {"accuracy": accuracy}
