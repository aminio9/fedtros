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

class FMRL_LA_Strategy(FedAvg):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.state_dim = int(cfg.model.latent_dim * cfg.strategy.max_agents)
        self.critics = {}

        self.aggregator = CentralizedAggregator(
            num_agents=cfg.strategy.max_agents, 
            state_dim=self.state_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.aggregator.parameters(), 
            lr=cfg.strategy.fmrl_lr
        )

        self._AsyncCriticClass = AsyncCritic
        self.is_training_phase = True
        self.selected_clients_cache = []
        self.utilities_cache = {}
        self.stage1_data_cache = {}
        self.saved_global_parameters = None
        self.warmup_rounds = getattr(cfg.strategy, "warmup_rounds", 5)

    def _get_critic(self, cid):
        if cid not in self.critics:
            c = self._AsyncCriticClass(
                hidden_dim=self.cfg.strategy.critic_hidden_dim, 
                latent_dim=self.cfg.model.latent_dim
            ).to(self.device)
            self.optimizer.add_param_group({"params": c.parameters()})
            self.critics[cid] = c
        return self.critics[cid]

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        if self.is_training_phase:
            logger.info(f"{'=' * 60}")
            logger.info(f" ROUND {server_round} [PHASE A]: Training & Auditing")
            logger.info(f"{'=' * 60}")
            clients = client_manager.sample(
                num_clients=self.min_fit_clients, 
                min_num_clients=self.min_fit_clients
            )
            self.saved_global_parameters = parameters
            config = {"server_round": server_round, "phase": "train"}
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]
        else:
            logger.info(f"{'=' * 60}")
            logger.info(f" ROUND {server_round} [PHASE B]: Uploading & Aggregation")
            logger.info(f"{'=' * 60}")
            selected_instructions = []
            config = {"server_round": server_round, "phase": "upload"}
            fit_ins = FitIns(parameters, config)
            for client in self.selected_clients_cache:
                selected_instructions.append((client, fit_ins))
            logger.info(f"   > Requesting heavy weights from {len(selected_instructions)} clients.")
            return selected_instructions

    def aggregate_fit(self, server_round: int, results, failures):
        if not results and self.is_training_phase:
            return self.saved_global_parameters, {}

        if self.is_training_phase:
            self._phase_a_logic(results)
            self.is_training_phase = False
            return self.saved_global_parameters, {}
        else:
            new_params = self._phase_b_logic(results, server_round)
            self.is_training_phase = True
            return new_params, {}

    def _phase_a_logic(self, results):
        """
        Calculates weights with EXPLICIT KL BOOSTING.
        """
        if results:
            results.sort(key=lambda x: x[0].cid)

        self.selected_clients_cache = []
        self.utilities_cache = {}
        self.stage1_data_cache = {} 
        selection_log = []

        raw_scores_list = []
        processed_clients = []

        for client, fit_res in results:
            try:
                metrics = fit_res.metrics
                h_vec = json.loads(metrics["hidden_info"]) 
                r_val = float(metrics["recent_reward"]) 
                r_hist = float(metrics["history_reward"]) / (max(1, self.cfg.server.num_rounds))
                recon_signal = float(metrics.get("utility_loss", 0.0)) 
                td_error = float(metrics.get("td_error", 0.0))
                kl_div = float(metrics.get("kl_div", 0.0)) 

                data = {
                    "h": torch.tensor([h_vec], dtype=torch.float32, device=self.device),
                    "r": torch.tensor([[r_val]], dtype=torch.float32, device=self.device),
                    "rh": torch.tensor([[r_hist]], dtype=torch.float32, device=self.device),
                    "recon": torch.tensor([[recon_signal]], dtype=torch.float32, device=self.device),
                    "td": torch.tensor([[td_error]], dtype=torch.float32, device=self.device),
                    "kl": torch.tensor([[kl_div]], dtype=torch.float32, device=self.device),
                }
                self.stage1_data_cache[client.cid] = data

                critic = self._get_critic(client.cid)
                critic.eval()
                with torch.no_grad():
                    # Score already includes (KL * 2.0) from the skip connection
                    raw_score = critic(
                        data["h"], data["r"], data["rh"], data["td"], data["recon"], data["kl"]
                    ).item()
                
                raw_scores_list.append(raw_score)
                # Store KL for the manual boost step next
                processed_clients.append((client, r_val, raw_score, recon_signal, kl_div))

            except Exception as e:
                logger.warning(f"Client {client.cid} sent bad metadata: {e}")
                continue

        # --- WEIGHT CALCULATION: SIGMOID + MANUAL KL BOOST ---
        if raw_scores_list:
            scores_tensor = torch.tensor(raw_scores_list)
            kl_tensor = torch.tensor([x[4] for x in processed_clients]) # Extract KLs
            
            # 1. Base Quality from Critic (0.0 to 1.0)
            base_quality = torch.sigmoid(scores_tensor) 

            # 2. MANUAL BOOST: Multiply by (1 + KL)
            # If KL is 5.0, weight becomes 6x larger immediately.
            # This ensures that even if Critic is conservative, high KL wins.
            boosted_scores = base_quality * (1.0 + kl_tensor)

            # 3. Normalize
            avg_quality = boosted_scores.mean()
            if avg_quality < 1e-6:
                final_weights_tensor = torch.ones_like(boosted_scores)
            else:
                final_weights_tensor = boosted_scores / avg_quality

            # 4. Clamp (Allow higher max influence for novel clients, up to 5x)
            final_weights_tensor = torch.clamp(final_weights_tensor, 0.1, 5.0)
            
            final_weights_list = final_weights_tensor.tolist()

            for idx, (client, r_val, raw_s, f1, kl) in enumerate(processed_clients):
                final_w = final_weights_list[idx]
                self.selected_clients_cache.append(client)
                self.utilities_cache[client.cid] = final_w

                selection_log.append([
                    client.cid[:8], 
                    f"{r_val:.2f}", 
                    f"{raw_s:.3f}", 
                    f"{final_w:.3f}", 
                    f"{f1:.2f}", 
                    f"{kl:.2f}",
                    "YES"
                ])

        logger.info(f"{'-' * 80}")
        logger.info(f"{'Client':<8} | {'Reward':<8} | {'RawLogit':<8} | {'Final_W':<8} | {'F1':<8} | {'KL':<8} | {'Sel'}")
        logger.info(f"{'-' * 80}")
        for row in selection_log:
            logger.info(f"{row[0]:<8} | {row[1]:<8} | {row[2]:<8} | {row[3]:<8} | {row[4]:<8} | {row[5]:<8} | {row[6]}")
        logger.info(f"{'-' * 80}\n")

    def _phase_b_logic(self, results, server_round):
        if results:
            results.sort(key=lambda x: x[0].cid)

        weighted_weights = []
        total_utility = 0.0
        global_reward_accum = 0.0

        is_warmup = server_round <= self.warmup_rounds
        if is_warmup:
            logger.info(f"   > [Warmup Round {server_round}] Using Standard FedAvg (w=1.0)")

        for client, fit_res in results:
            if is_warmup:
                w_i = 1.0
            else:
                w_i = self.utilities_cache.get(client.cid, 1.0)
                # Keep the same high clamp for aggregation
                w_i = max(0.01, min(w_i, 5.0))

            weights = parameters_to_ndarrays(fit_res.parameters)
            weighted_weights.append((weights, w_i))
            
            total_utility += w_i
            global_reward_accum += fit_res.metrics.get("recent_reward", 0.0)

        if weighted_weights:
            normalized_weights = [(w, util / total_utility) for w, util in weighted_weights]
            new_weights = [np.zeros_like(w) for w in weighted_weights[0][0]]
            for weights, norm_w in normalized_weights:
                for i, layer in enumerate(weights):
                    new_weights[i] += layer * norm_w

            avg_reward = global_reward_accum / len(results) if results else 0.0
            
            if not is_warmup:
                self._train_server_models(results, avg_reward)

            return ndarrays_to_parameters(new_weights)

        return self.saved_global_parameters

    def _train_server_models(self, results, actual_reward):
        try:
            self.optimizer.zero_grad()
            self.aggregator.train()
            for cid in self.critics:
                self.critics[cid].train()

            weights_list = []
            latents_list = []

            for client, _ in results:
                data = self.stage1_data_cache.get(client.cid)
                if not data: continue 

                critic = self._get_critic(client.cid)
                w_i = critic(
                    data["h"], data["r"], data["rh"], data["td"], data["recon"], data["kl"]
                )
                weights_list.append(w_i)
                latents_list.append(data["h"])

            if not weights_list: return

            w_cat = torch.cat(weights_list, dim=1)
            s_cat = torch.cat(latents_list, dim=1)

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

            predicted_reward = self.aggregator(w_cat, s_cat)
            target = torch.tensor([[actual_reward]], dtype=torch.float32, device=self.device)
            loss = F.mse_loss(predicted_reward, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.aggregator.parameters(), 1.0)
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, 1.0)

            self.optimizer.step()
            logger.info(f"   > Server RL Update | Loss: {loss.item():.6f}")

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
        logger.info("--- Strategy: FedAvg ---")
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