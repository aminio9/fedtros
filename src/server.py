import logging
import json
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import flwr as fl
import matplotlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flwr.server.strategy import FedAvg, FedProx, Strategy
from flwr.common import (
    Parameters,
    FitIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from .utils import get_device

logger = logging.getLogger("Server")

REWARD_HISTORY: List[Tuple[int, float]] = []


def _project_root() -> Path:
    try:
        return Path(get_original_cwd())
    except ValueError:
        return Path(os.getcwd())


def _resolve_path(path_like) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (_project_root() / path)


def reset_reward_history() -> None:
    """Clear stored reward metrics before a new training session."""
    REWARD_HISTORY.clear()


def get_reward_history() -> List[Tuple[int, float]]:
    """Return a shallow copy of the reward history."""
    return list(REWARD_HISTORY)


def fit_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
    """
    Pass configuration to the client's fit method.
    CRITICAL FIX: We sets 'phase' to 'standard' so the client knows
    it is NOT in the special FMRL two-phase mode.
    """
    return {"server_round": server_round, "phase": "standard"}

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import logging
from flwr.common import (
    Parameters,
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

# Set up logger
logger = logging.getLogger(__name__)

# Mock get_device function if not imported
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FMRL_LA_Strategy(FedAvg):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- LAZY IMPORT ---
        # Prevents circular imports and ensures we use the updated models
        try:
            from .server_models import AsyncCritic, CentralizedAggregator
        except ImportError:
            try:
                from server_models import AsyncCritic, CentralizedAggregator
            except ImportError:
                logger.error("Could not import server_models. FMRL-LA strategy will fail.")
                raise

        self.cfg = cfg
        self.device = get_device()

        # Global State is aggregation of all client latents (conceptually)
        # We define state_dim based on max_agents to allow padding
        self.state_dim = int(cfg.model.latent_dim * cfg.strategy.max_agents)
        self.critics = {}

        # Initialize Server Aggregator (The "Judge")
        self.aggregator = CentralizedAggregator(
            num_agents=cfg.strategy.max_agents, 
            state_dim=self.state_dim
        ).to(self.device)

        # Optimizer handles the Aggregator AND all dynamically added Critics
        self.optimizer = optim.Adam(
            self.aggregator.parameters(), 
            lr=cfg.strategy.fmrl_lr
        )

        self._AsyncCriticClass = AsyncCritic

        # State Machine & Caching
        self.is_training_phase = True
        self.selected_clients_cache = []
        self.utilities_cache = {}
        self.stage1_data_cache = {} # Stores tensors for the backward pass
        self.saved_global_parameters = None
        self.warmup_rounds = getattr(cfg.strategy, "warmup_rounds", 5)

    def _get_critic(self, cid):
        """
        Retrieves or creates a Critic network for a specific client.
        Adds the new Critic parameters to the existing optimizer.
        """
        if cid not in self.critics:
            c = self._AsyncCriticClass(
                hidden_dim=self.cfg.strategy.critic_hidden_dim, 
                latent_dim=self.cfg.model.latent_dim
            ).to(self.device)
            
            # Critical: Add new parameters to the running optimizer
            self.optimizer.add_param_group({"params": c.parameters()})
            self.critics[cid] = c
            
        return self.critics[cid]

    # ----------------------------------------------------------------------
    # 1. CONFIGURE FIT (Standard FL Flow)
    # ----------------------------------------------------------------------
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # === PHASE A: TRAIN & AUDIT ===
        if self.is_training_phase:
            logger.info(f"{'=' * 60}")
            logger.info(f" ROUND {server_round} [PHASE A]: Training & Auditing")
            logger.info(f"{'=' * 60}")

            # Sample clients
            clients = client_manager.sample(
                num_clients=self.min_fit_clients, 
                min_num_clients=self.min_fit_clients
            )
            
            # Cache global parameters to return them unchanged after Phase A
            self.saved_global_parameters = parameters

            # Instruct clients to run 'audit' mode (short training + metric reporting)
            config = {"server_round": server_round, "phase": "train"}
            fit_ins = FitIns(parameters, config)
            
            return [(client, fit_ins) for client in clients]

        # === PHASE B: UPLOAD & AGGREGATE ===
        else:
            logger.info(f"{'=' * 60}")
            logger.info(f" ROUND {server_round} [PHASE B]: Uploading & Aggregation")
            logger.info(f"{'=' * 60}")

            # Only request weights from clients we decided to keep in Phase A
            selected_instructions = []
            config = {"server_round": server_round, "phase": "upload"}
            fit_ins = FitIns(parameters, config)

            for client in self.selected_clients_cache:
                selected_instructions.append((client, fit_ins))

            logger.info(f"   > Requesting heavy weights from {len(selected_instructions)} clients.")
            return selected_instructions

    # ----------------------------------------------------------------------
    # 2. AGGREGATE FIT (Logic Core)
    # ----------------------------------------------------------------------
    def aggregate_fit(self, server_round: int, results, failures):
        if not results and self.is_training_phase:
            return self.saved_global_parameters, {}

        # === PHASE A RESULT: CRITIC INFERENCE & SELECTION ===
        if self.is_training_phase:
            self._phase_a_logic(results)
            
            # Switch state to Phase B next
            self.is_training_phase = False
            
            # Return saved parameters (no update yet) so Phase B starts with same model
            return self.saved_global_parameters, {}

        # === PHASE B RESULT: WEIGHTED AGGREGATION & SERVER TRAINING ===
        else:
            new_params = self._phase_b_logic(results, server_round)
            
            # Switch state back to Phase A for next round
            self.is_training_phase = True
            return new_params, {}

    # ----------------------------------------------------------------------
    # LOGIC HELPERS
    # ----------------------------------------------------------------------

    def _phase_a_logic(self, results):
        """
        Process audit metrics, run Critics, and calculate weights via Sigmoid.
        """
        self.selected_clients_cache = []
        self.utilities_cache = {}
        self.stage1_data_cache = {} # Cache tensors for training step later
        selection_log = []

        raw_scores_list = []
        processed_clients = []

        for client, fit_res in results:
            try:
                metrics = fit_res.metrics
                
                # 1. Parse Metadata
                h_vec = json.loads(metrics["hidden_info"]) # Latent vector
                r_val = float(metrics["recent_reward"]) 
                r_hist = float(metrics["history_reward"]) / (max(1, self.cfg.server.num_rounds))
                recon_signal = float(metrics.get("utility_loss", 0.0)) # F1 Score
                td_error = float(metrics.get("td_error", 0.0))

                # 2. Create Tensors (Cache these for _train_server_models)
                data = {
                    "h": torch.tensor([h_vec], dtype=torch.float32, device=self.device),
                    "r": torch.tensor([[r_val]], dtype=torch.float32, device=self.device),
                    "rh": torch.tensor([[r_hist]], dtype=torch.float32, device=self.device),
                    "recon": torch.tensor([[recon_signal]], dtype=torch.float32, device=self.device),
                    "td": torch.tensor([[td_error]], dtype=torch.float32, device=self.device),
                }
                self.stage1_data_cache[client.cid] = data

                # 3. Critic Inference
                critic = self._get_critic(client.cid)
                critic.eval()
                with torch.no_grad():
                    # Get raw logit score
                    raw_score = critic(
                        data["h"], data["r"], data["rh"], data["td"], data["recon"]
                    ).item()
                
                raw_scores_list.append(raw_score)
                processed_clients.append((client, r_val, raw_score, recon_signal))

            except Exception as e:
                logger.warning(f"Client {client.cid} sent bad metadata: {e}")
                continue

        # 4. SIGMOID WEIGHTING (The Fix)
        # Replaced Softmax with Sigmoid + Normalize + Clamp
        # Softmax creates "Winner-Takes-All" competition. Sigmoid evaluates quality independently.
        if raw_scores_list:
            scores_tensor = torch.tensor(raw_scores_list)
            
            # Step A: Sigmoid (Is this client good?)
            # Maps logits to [0, 1] independently
            quality_scores = torch.sigmoid(scores_tensor) 

            # Step B: Normalize around 1.0
            # FedAvg expects weights to roughly average to 1.0 (magnitude preservation)
            avg_quality = quality_scores.mean()
            
            if avg_quality < 1e-6:
                # Avoid division by zero if all scores are effectively 0
                final_weights_tensor = torch.ones_like(quality_scores)
            else:
                final_weights_tensor = quality_scores / avg_quality

            # Step C: Safety Clamp
            # Prevent any single client from having > 3x influence or < 0.1x influence
            final_weights_tensor = torch.clamp(final_weights_tensor, 0.1, 3.0)
            
            # Convert to list
            final_weights_list = final_weights_tensor.tolist()

            # Map back to clients
            for idx, (client, r_val, raw_s, f1) in enumerate(processed_clients):
                final_w = final_weights_list[idx]
                
                self.selected_clients_cache.append(client)
                self.utilities_cache[client.cid] = final_w

                selection_log.append([
                    client.cid[:8], 
                    f"{r_val:.2f}", 
                    f"{raw_s:.3f}", 
                    f"{final_w:.3f}", 
                    f"{f1:.2f}", 
                    "YES"
                ])

        # Logging
        logger.info(f"{'-' * 80}")
        logger.info(f"{'Client':<8} | {'Reward':<8} | {'RawLogit':<8} | {'Final_W':<8} | {'F1':<8} | {'Sel'}")
        logger.info(f"{'-' * 80}")
        for row in selection_log:
            logger.info(f"{row[0]:<8} | {row[1]:<8} | {row[2]:<8} | {row[3]:<8} | {row[4]:<8} | {row[5]}")
        logger.info(f"{'-' * 80}\n")

    def _phase_b_logic(self, results, server_round):
        """
        Aggregate weights based on Utilities calculated in Phase A.
        Then, TRAIN the critics to align with the actual global reward.
        """
        weighted_weights = []
        total_utility = 0.0
        global_reward_accum = 0.0

        is_warmup = server_round <= self.warmup_rounds
        if is_warmup:
            logger.info(f"   > [Warmup Round {server_round}] Using Standard FedAvg (w=1.0)")

        # 1. Prepare Weighted Weights
        for client, fit_res in results:
            if is_warmup:
                w_i = 1.0
            else:
                w_i = self.utilities_cache.get(client.cid, 1.0)
                # Double safety clip
                w_i = max(0.01, min(w_i, 5.0))

            # Deserialize parameters
            weights = parameters_to_ndarrays(fit_res.parameters)
            weighted_weights.append((weights, w_i))
            
            total_utility += w_i
            global_reward_accum += fit_res.metrics.get("recent_reward", 0.0)

        # 2. Perform Aggregation
        if weighted_weights:
            # Normalize so weights sum to 1.0 (Standard FedAvg logic with custom weights)
            normalized_weights = [(w, util / total_utility) for w, util in weighted_weights]
            
            # Summation
            new_weights = [np.zeros_like(w) for w in weighted_weights[0][0]]
            for weights, norm_w in normalized_weights:
                for i, layer in enumerate(weights):
                    new_weights[i] += layer * norm_w

            # 3. Train Server Models (The RL Update)
            # Use the average local reward as a proxy for the 'Global Reward' 
            avg_reward = global_reward_accum / len(results) if results else 0.0
            
            if not is_warmup:
                self._train_server_models(results, avg_reward)

            return ndarrays_to_parameters(new_weights)

        return self.saved_global_parameters

    def _train_server_models(self, results, actual_reward):
        """
        Re-run the forward pass to build the gradient graph and update
        Critics + Aggregator based on how well they predicted the Global Reward.
        """
        try:
            self.optimizer.zero_grad()
            self.aggregator.train()
            
            # Switch active critics to train mode
            for cid in self.critics:
                self.critics[cid].train()

            weights_list = []
            latents_list = []

            # 1. Re-run Forward Pass for Critics
            for client, _ in results:
                data = self.stage1_data_cache.get(client.cid)
                if not data: 
                    continue 

                # Get the critic model
                critic = self._get_critic(client.cid)
                
                # Forward pass (Builds graph)
                w_i = critic(
                    data["h"], data["r"], data["rh"], data["td"], data["recon"]
                )
                
                weights_list.append(w_i)
                latents_list.append(data["h"])

            if not weights_list:
                return

            # 2. Concatenate for Aggregator
            w_cat = torch.cat(weights_list, dim=1)
            s_cat = torch.cat(latents_list, dim=1)

            # 3. Robust Padding
            curr_agents = w_cat.shape[1]
            max_agents = self.aggregator.num_agents
            
            curr_state_dim = s_cat.shape[1]
            target_state_dim = self.state_dim 

            if curr_state_dim < target_state_dim:
                s_cat = F.pad(s_cat, (0, target_state_dim - curr_state_dim), value=0.0)
            elif curr_state_dim > target_state_dim:
                s_cat = s_cat[:, :target_state_dim]

            if curr_agents < max_agents:
                w_cat = F.pad(w_cat, (0, max_agents - curr_agents), value=0.0)
            elif curr_agents > max_agents:
                w_cat = w_cat[:, :max_agents]

            # 4. Aggregator Forward Pass
            predicted_reward = self.aggregator(w_cat, s_cat)

            # 5. Loss & Backprop
            target = torch.tensor([[actual_reward]], dtype=torch.float32, device=self.device)
            loss = F.mse_loss(predicted_reward, target)
            
            loss.backward()

            # 6. Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.aggregator.parameters(), 1.0)
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, 1.0)

            self.optimizer.step()
            
            logger.info(f"   > Server RL Update | Loss: {loss.item():.6f} | Reward Target: {actual_reward:.4f}")

        except Exception as e:
            logger.error(f"Server model training failed: {e}")
            
            
def aggregate_fit_metrics(
    fit_metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]],
) -> Dict[str, float]:
    if not fit_metrics:
        return {}
    total_examples = sum(num for num, _ in fit_metrics)
    if total_examples == 0:
        return {}
    aggregated = {}
    for num, metrics in fit_metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + (float(v) * num)
    return {k: v / total_examples for k, v in aggregated.items()}


def aggregate_evaluate_metrics(
    eval_metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]],
) -> Dict[str, float]:
    if not eval_metrics:
        return {}
    total_examples = sum(num for num, _ in eval_metrics)
    if total_examples == 0:
        return {}
    aggregated = {}
    for num, metrics in eval_metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + (float(v) * num)
    return {k: v / total_examples for k, v in aggregated.items()}


def get_strategy(cfg: DictConfig) -> Strategy:
    """
    Factory to return the appropriate strategy.
    """
    strat_name = getattr(cfg.strategy, "name", "fedavg").lower()

    # Common arguments
    args = dict(
        fraction_fit=cfg.server.fraction_fit,
        fraction_evaluate=cfg.server.fraction_evaluate,
        min_fit_clients=cfg.server.min_fit_clients,
        min_evaluate_clients=cfg.server.min_evaluate_clients,
        min_available_clients=cfg.server.min_available_clients,
        on_fit_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    if strat_name == "fmrl_la":
        logger.info("--- Strategy: FMRL-LA (Two-Phase) ---")
        return FMRL_LA_Strategy(cfg=cfg, **args)

    elif strat_name == "fedprox":
        proximal_mu = float(cfg.server.get("proximal_mu", 0.1))
        logger.info("--- Strategy: FedProx (mu=%.2f) ---", proximal_mu)
        return FedProx(
            proximal_mu=proximal_mu,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            **args,
        )

    else:
        logger.info("--- Strategy: FedAvg (Simple) ---")
        return FedAvg(fit_metrics_aggregation_fn=aggregate_fit_metrics, **args)


def run_server(cfg: DictConfig, device: Optional[torch.device] = None) -> None:
    reset_reward_history()
    strategy = get_strategy(cfg)

    logger.info(
        "Starting server at %s for %s rounds.",
        cfg.server.address,
        cfg.server.num_rounds,
    )
    try:
        fl.server.start_server(
            server_address=cfg.server.address,
            config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
            strategy=strategy,
        )
    except RuntimeError as exc:
        if "Failed to bind to address" in str(exc):
            logger.error("Unable to bind to %s. Port in use.", cfg.server.address)
            raise SystemExit(1) from exc
        raise


def plot_reward_history(cfg: DictConfig) -> None:
    history = get_reward_history()
    if not history:
        logger.warning("No reward history collected.")
        return

    figure_dir = _resolve_path(cfg.paths.figures_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rounds, rewards = zip(*sorted(history, key=lambda item: item[0]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, rewards, marker="o", color="#2a9d8f", linewidth=2)
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Average Reward per Episode")
    ax.set_title("Client Reward Across Federated Rounds")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = figure_dir / "federated_rewards.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info("Saved reward history plot to %s", plot_path)
