import logging
from typing import Tuple, List, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

try:
    from .models import OpenSetQChainModelFactory, ValueNetwork
    from .utils import (
        reparameterization_trick,
        calculate_kl_divergence,
        calculate_kl_divergence_raw,
        soft_update_target_network,
    )
except ImportError:
    from models import OpenSetQChainModelFactory, ValueNetwork
    from utils import (
        reparameterization_trick,
        calculate_kl_divergence,
        calculate_kl_divergence_raw,
        soft_update_target_network,
    )

logger = logging.getLogger(__name__)

def _ensure_action_index(a: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Ensure action tensor is a long int tensor of shape [B, 1]."""
    if a.dim() == 1:
        a = a.unsqueeze(1)
    a = a.long()
    return a.clamp_(0, num_actions - 1)


class Agent:
    """
    Implements the core training logic for the CVAE-DQN agent,
    including federated parameter handling.
    """
    def __init__(self, model_factory: OpenSetQChainModelFactory, train_cfg: DictConfig, device: torch.device):
        self.train_cfg = train_cfg
        self.device = device
        self.action_dim = model_factory.num_actions
        self.gamma = train_cfg.gamma
        self.use_double_dqn = train_cfg.use_double_dqn
        self.use_prior_kl_raw = bool(getattr(train_cfg, 'prior_kl_raw', False))
        self._prior_kl_fn = (
            calculate_kl_divergence_raw if self.use_prior_kl_raw else calculate_kl_divergence
        )
        self.prior_grad_clip_norm = getattr(train_cfg, 'prior_grad_clip_norm', None)

        # Nets (encoder + decoder stack)
        self.value_network: ValueNetwork = model_factory.create_value_network().to(device)
        self.encoder = self.value_network.encoder
        self.decoder = self.value_network.decoder
        
        # Global (Federated) Components
        self.prior_net = self.encoder.prior
        self.value_net_main = self.decoder.main_q
        self.generation_net = model_factory.create_generation_network().to(device)
        
        # Local (Personalized) Components
        self.recognition_net = self.encoder.recognition
        self.value_net_target = self.decoder.target_q
        
        # Sync target net initially
        self.value_net_target.load_state_dict(self.value_net_main.state_dict())
        self.value_net_target.eval()

        # Optimizers
        self.optimizer_prior = optim.Adam(self.prior_net.parameters(), lr=train_cfg.lr_prior)
        # RL optimizer trains *both* the local RecognitionNet and the global MainQNet
        rl_params = list(self.recognition_net.parameters()) + list(self.value_net_main.parameters())
        self.optimizer_q_rl = optim.Adam(rl_params, lr=train_cfg.lr_q_rl)

        self.td_loss_fn = nn.SmoothL1Loss()
        logger.debug("Agent initialized with Double-DQN: %s", self.use_double_dqn)

    @torch.no_grad()
    def _bootstrap_target(self, next_states_s: torch.Tensor) -> torch.Tensor:
        """
        Build y_t targets using the PRIOR for z_{t+1}.
        (Eq. 9 or 13, but using prior for z)
        """
        self.prior_net.eval()
        self.value_net_main.eval()
        self.value_net_target.eval()
        
        next_mu_p, next_log_var_p = self.prior_net(next_states_s)
        z_next = reparameterization_trick(next_mu_p, next_log_var_p)

        if self.use_double_dqn:
            # Double-DQN (Eq. 13)
            q_main_next = self.value_net_main(z_next, next_states_s)                  # [B, A]
            a_star = q_main_next.argmax(dim=1, keepdim=True)                          # [B, 1]
            q_target_next = self.value_net_target(z_next, next_states_s).gather(1, a_star)  # [B,1]
        else:
            # Standard DQN (Eq. 9)
            q_target_next = self.value_net_target(z_next, next_states_s).max(dim=1, keepdim=True)[0]  # [B,1]
        return q_target_next

    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float, float]:
        """
        Performs one full training step (Prior update + RL update).
        batch = (states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t)
        Returns: (td_loss_item, kl_loss_item, avg_q_item)
        """
        states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t = batch

        actions_a_t = _ensure_action_index(actions_a_t, self.action_dim)
        true_actions_a_t = _ensure_action_index(true_actions_a_t, self.action_dim)

        # ========== 1) PRIOR update: KL(q||p) using TRUE labels (Eq. 5) ==========
        self.prior_net.train()
        self.recognition_net.eval() # Recognition net is frozen for this part
        
        with torch.no_grad():
            mu_q_T, log_var_q_T = self.recognition_net(states_s, true_actions_a_t)  # q_phi(z|s, a_T)
        mu_p, log_var_p = self.prior_net(states_s)                                   # p_theta(z|s)

        kl_loss = self._prior_kl_fn(mu_q_T, log_var_q_T, mu_p, log_var_p)
        
        self.optimizer_prior.zero_grad()
        kl_loss.backward()
        if self.prior_grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.prior_net.parameters(), float(self.prior_grad_clip_norm))
        self.optimizer_prior.step()

        # ========== 2) RL update: recognition + main Q via TD (Eq. 16) ==========
        self.recognition_net.train()
        self.value_net_main.train()
        
        # Calculate target y_t (Eq. 13)
        with torch.no_grad():
            q_target_next = self._bootstrap_target(next_states_s)
            y_t = rewards_r + (1.0 - dones) * (self.gamma * q_target_next)
        
        # Calculate predicted Q(s_t, a_t)
        # Note: Uses z from recognition(s_t, a_t), *not* true_action_a_t
        mu_q_rl, log_var_q_rl = self.recognition_net(states_s, actions_a_t)
        z_now = reparameterization_trick(mu_q_rl, log_var_q_rl)
        q_values_all = self.value_net_main(z_now, states_s)
        q_pred = q_values_all.gather(1, actions_a_t)

        # Calculate TD Loss (Eq. 11)
        td_loss = self.td_loss_fn(q_pred, y_t)

        # Optimize both recognition and main_q nets
        self.optimizer_q_rl.zero_grad()
        td_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.recognition_net.parameters()) + list(self.value_net_main.parameters()),
            max_norm=1.0 # Common practice for DQN stability
        )
        self.optimizer_q_rl.step()

        avg_q = q_pred.mean().item()
        return td_loss.item(), kl_loss.item(), avg_q

    def update_target_network(self, tau: float):
        """Soft update the target network (Eq. 10)."""
        logger.debug("Performing soft target update with tau=%s", tau)
        soft_update_target_network(self.value_net_main, self.value_net_target, tau)
        
    def hard_update_target_network(self):
        """Hard update (copy) the target network."""
        logger.debug("Performing hard target update")
        self.value_net_target.load_state_dict(self.value_net_main.state_dict())

    # --- METHODS FOR FEDERATED LEARNING ---

    def get_federated_parameters(self) -> List[np.ndarray]:
        """Get the parameters of the components to be federated (Prior + MainQ)."""
        logger.debug("Getting federated parameters (PriorNet + MainQNet + GenerationNet)")
        
        prior_state_dict = self.prior_net.state_dict()
        main_q_state_dict = self.value_net_main.state_dict()
        generation_state_dict = self.generation_net.state_dict()
        
        # Combine parameters into a single list
        params = [val.cpu().numpy() for val in prior_state_dict.values()]
        params.extend([val.cpu().numpy() for val in main_q_state_dict.values()])
        params.extend([val.cpu().numpy() for val in generation_state_dict.values()])
            
        return params

    def set_federated_parameters(self, parameters: List[np.ndarray], hard_target_update: bool = True):
        """
        Set the received federated parameters (Prior + MainQ).
        This logic is the inverse of get_federated_parameters.
        """
        logger.debug("Setting federated parameters (PriorNet + MainQNet + GenerationNet)")

        prior_keys = list(self.prior_net.state_dict().keys())
        main_q_keys = list(self.value_net_main.state_dict().keys())
        generation_keys = list(self.generation_net.state_dict().keys())
        
        num_prior_params = len(prior_keys)
        num_main_q_params = len(main_q_keys)
        num_generation_params = len(generation_keys)
        expected_without_generator = num_prior_params + num_main_q_params
        expected_with_generator = expected_without_generator + num_generation_params

        if len(parameters) not in (expected_without_generator, expected_with_generator):
            logger.error(
                "Parameter mismatch: expected %s or %s tensors, but received %s",
                expected_without_generator,
                expected_with_generator,
                len(parameters),
            )
            raise ValueError("Mismatched number of parameters received from server")
        
        # Load PriorNet parameters
        prior_load_dict = OrderedDict(zip(
            prior_keys,
            [torch.tensor(p, device=self.device) for p in parameters[:num_prior_params]]
        ))
        
        # Load MainQNet parameters
        main_start = num_prior_params
        main_end = main_start + num_main_q_params
        main_q_load_dict = OrderedDict(zip(
            main_q_keys,
            [torch.tensor(p, device=self.device) for p in parameters[main_start:main_end]]
        ))
            
        self.prior_net.load_state_dict(prior_load_dict)
        self.value_net_main.load_state_dict(main_q_load_dict)

        if len(parameters) == expected_with_generator:
            generator_load_dict = OrderedDict(
                zip(
                    generation_keys,
                    [
                        torch.tensor(p, device=self.device)
                        for p in parameters[main_end:expected_with_generator]
                    ],
                )
            )
            self.generation_net.load_state_dict(generator_load_dict)
        else:
            logger.debug("No generation network weights provided; keeping local generator as-is.")
        
        if hard_target_update:
            logger.debug("Performing hard update of target network post-aggregation")
            self.hard_update_target_network()

    def train_generation_network(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        generator_cfg: DictConfig,
    ) -> Dict[str, float]:
        """
        Train the generation network on correctly classified samples using the
        procedure described for the Unknown Attack Recognition module.
        """
        features = features.detach().cpu().float()
        labels = labels.detach().cpu().long()
        if features.dim() != 2 or labels.dim() != 1:
            raise ValueError("Features must be [N, D] and labels must be [N].")

        gen_batch_size = int(getattr(generator_cfg, "batch_size", 128))
        gen_lr = float(getattr(generator_cfg, "lr", 5e-4))
        gen_rounds = max(1, int(getattr(generator_cfg, "rounds", 1)))
        gen_epochs_per_round = max(
            1,
            int(
                getattr(
                    generator_cfg,
                    "epochs_per_round",
                    getattr(generator_cfg, "epochs", 5),
                )
            ),
        )
        min_correct = int(getattr(generator_cfg, "min_correct_samples", 1))

        dataset = TensorDataset(features, labels)
        full_loader = DataLoader(dataset, batch_size=gen_batch_size, shuffle=False)

        correct_mask_parts: List[torch.Tensor] = []
        with torch.no_grad():
            self.prior_net.eval()
            self.value_net_main.eval()
            for states_s, true_actions in full_loader:
                states_s = states_s.to(self.device)
                true_actions = true_actions.to(self.device)
                mu_p, _ = self.prior_net(states_s)
                q_values = self.value_net_main(mu_p, states_s)
                preds = q_values.argmax(dim=1)
                correct_mask_parts.append((preds == true_actions).cpu())

        if not correct_mask_parts:
            logger.warning("Generator training skipped: no batches available.")
            return {}

        correct_mask = torch.cat(correct_mask_parts, dim=0)
        num_correct = int(correct_mask.sum().item())
        total_samples = int(features.shape[0])

        if num_correct < min_correct:
            logger.warning(
                "Generator training skipped: only %s/%s correctly classified samples (< %s).",
                num_correct,
                total_samples,
                min_correct,
            )
            return {
                "generator_samples": float(num_correct),
                "generator_correct_frac": num_correct / max(1, total_samples),
            }

        filtered_dataset = TensorDataset(features[correct_mask], labels[correct_mask])
        train_loader = DataLoader(filtered_dataset, batch_size=gen_batch_size, shuffle=True)

        optimizer = optim.Adam(self.generation_net.parameters(), lr=gen_lr)
        loss_fn = nn.MSELoss()

        self.generation_net.train()
        self.recognition_net.eval()

        last_epoch_loss = 0.0
        for round_idx in range(1, gen_rounds + 1):
            logger.info(
                "Generator round %02d/%02d | samples=%s/%s",
                round_idx,
                gen_rounds,
                num_correct,
                total_samples,
            )
            for epoch in range(1, gen_epochs_per_round + 1):
                total_loss = 0.0
                for states_s, true_actions in train_loader:
                    states_s = states_s.to(self.device)
                    true_actions = true_actions.to(self.device)
                    with torch.no_grad():
                        mu_q, log_var_q = self.recognition_net(states_s, true_actions)
                        latent_z = reparameterization_trick(mu_q, log_var_q)
                    recon = self.generation_net(latent_z, true_actions)
                    loss = loss_fn(recon, states_s)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                last_epoch_loss = total_loss / max(1, len(train_loader))
                logger.info(
                    "Round %02d/%02d | Epoch %02d/%02d | avg recon mse: %.6f",
                    round_idx,
                    gen_rounds,
                    epoch,
                    gen_epochs_per_round,
                    last_epoch_loss,
                )

        self.generation_net.eval()
        return {
            "generator_loss": float(last_epoch_loss),
            "generator_samples": float(num_correct),
            "generator_correct_frac": num_correct / max(1, total_samples),
        }
