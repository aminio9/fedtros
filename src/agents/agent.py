import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from src.models.models import OpenSetQChainModelFactory, ValueNetwork
from src.utils.utils import (
    calculate_kl_divergence,
    calculate_kl_divergence_raw,
    reparameterization_trick,
    soft_update_target_network,
)

logger = logging.getLogger("Agent")


def _ensure_action_index(a: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Ensure action tensor is a long int tensor of shape [B, 1]."""
    if a.dim() == 1:
        a = a.unsqueeze(1)
    a = a.long()
    return a.clamp_(0, num_actions - 1)


class DirectMLAdam(optim.Optimizer):
    """Adam variant that avoids unsupported ops on DirectML by using add/mul instead of lerp."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias_correction2**0.5) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Agent:
    """
    Implements the core training logic for the CVAE-DQN agent,
    including federated parameter handling.
    """

    def __init__(
        self,
        model_factory: OpenSetQChainModelFactory,
        train_cfg: DictConfig,
        device: torch.device,
    ):
        self.train_cfg = train_cfg
        self.device = device
        self.action_dim = model_factory.num_actions
        self.gamma = train_cfg.gamma
        self.use_double_dqn = train_cfg.use_double_dqn
        self.use_prior_kl_raw = bool(train_cfg.prior_kl_raw)
        self._prior_kl_fn = (
            calculate_kl_divergence_raw if self.use_prior_kl_raw else calculate_kl_divergence
        )
        self.prior_grad_clip_norm = train_cfg.prior_grad_clip_norm

        # Nets (encoder + decoder stack)
        self.value_network: ValueNetwork = model_factory.create_value_network().to(device)
        self.encoder = self.value_network.encoder
        self.decoder = self.value_network.decoder

        # Global (Federated) Components
        self.prior_net = self.encoder.prior
        self.recognition_net = self.encoder.recognition
        self.value_net_main = self.decoder.main_q
        self.generation_net = model_factory.create_generation_network().to(device)

        # Local (Personalized) Components
        self.value_net_target = self.decoder.target_q

        # Sync target net initially
        self.value_net_target.load_state_dict(self.value_net_main.state_dict())
        self.value_net_target.eval()

        use_directml_safe = device.type in {"dml", "directml", "privateuseone"}
        adam_cls = DirectMLAdam if use_directml_safe else optim.Adam

        # Optimizers (DirectML-safe variant avoids unsupported lerp scatter ops)
        if adam_cls is optim.Adam:
            self.optimizer_prior = adam_cls(
                self.prior_net.parameters(), lr=train_cfg.lr_prior, foreach=False
            )
        else:
            self.optimizer_prior = adam_cls(self.prior_net.parameters(), lr=train_cfg.lr_prior)
        # RL optimizer trains *both* the local RecognitionNet and the global MainQNet
        rl_params = list(self.recognition_net.parameters()) + list(self.value_net_main.parameters())
        if adam_cls is optim.Adam:
            self.optimizer_q_rl = adam_cls(rl_params, lr=train_cfg.lr_q_rl, foreach=False)
        else:
            self.optimizer_q_rl = adam_cls(rl_params, lr=train_cfg.lr_q_rl)

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
            q_main_next = self.value_net_main(z_next, next_states_s)  # [B, A]
            a_star = q_main_next.argmax(dim=1, keepdim=True)  # [B, 1]
            q_target_next = self.value_net_target(z_next, next_states_s).gather(1, a_star)  # [B,1]
        else:
            # Standard DQN (Eq. 9)
            q_target_next = self.value_net_target(z_next, next_states_s).max(dim=1, keepdim=True)[
                0
            ]  # [B,1]
        return q_target_next

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> tuple[float, float, float]:
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
        self.recognition_net.eval()  # Recognition net is frozen for this part

        with torch.no_grad():
            mu_q_T, log_var_q_T = self.recognition_net(
                states_s, true_actions_a_t
            )  # q_phi(z|s, a_T)
        mu_p, log_var_p = self.prior_net(states_s)  # p_theta(z|s)

        kl_loss = self._prior_kl_fn(mu_q_T, log_var_q_T, mu_p, log_var_p)

        self.optimizer_prior.zero_grad()
        kl_loss.backward()
        if self.prior_grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.prior_net.parameters(), float(self.prior_grad_clip_norm)
            )
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
            max_norm=1.0,  # Common practice for DQN stability
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

    def get_federated_parameters(self) -> list[np.ndarray]:
        """
        Get the parameters to federate (Prior + Recognition + MainQ + Generation).
        IMPORTANT: Includes Generator params if the network exists, even if training is disabled.
        This ensures the server can save the full model structure.
        """
        logger.debug("Getting federated parameters (Prior + Recognition + MainQ + Generation)")

        prior_state_dict = self.prior_net.state_dict()
        recog_state_dict = self.recognition_net.state_dict()
        main_q_state_dict = self.value_net_main.state_dict()

        params = [val.cpu().numpy() for val in prior_state_dict.values()]
        params.extend([val.cpu().numpy() for val in recog_state_dict.values()])
        params.extend([val.cpu().numpy() for val in main_q_state_dict.values()])

        # FIXED: Robust check if generator exists
        if self.generation_net is not None:
            generation_state_dict = self.generation_net.state_dict()
            params.extend([val.cpu().numpy() for val in generation_state_dict.values()])

        return params

    def set_federated_parameters(
        self, parameters: list[np.ndarray], hard_target_update: bool = True
    ):
        """
        Set the received federated parameters (Prior + Recognition + MainQ + Generation).
        Robustly handles cases where generator weights are missing (if server didn't send them).
        """
        # Logging check
        logger.debug("Setting federated parameters")

        prior_keys = list(self.prior_net.state_dict().keys())
        recog_keys = list(self.recognition_net.state_dict().keys())
        main_q_keys = list(self.value_net_main.state_dict().keys())

        num_prior = len(prior_keys)
        num_recog = len(recog_keys)
        num_main = len(main_q_keys)

        base_params_count = num_prior + num_recog + num_main

        # Determine if we have extra params for generator
        gen_keys = []
        num_gen = 0

        # FIXED: Robust check if generator exists
        if self.generation_net is not None:
            gen_keys = list(self.generation_net.state_dict().keys())
            num_gen = len(gen_keys)

        expected_with_gen = base_params_count + num_gen
        received_count = len(parameters)

        # Log what we are receiving for clarity
        if received_count == expected_with_gen:
            logger.info(
                f"   > Loading Global Update: Agent Core ({base_params_count} layers) + Generator ({num_gen} layers)"
            )
        elif received_count == base_params_count:
            logger.info(f"   > Loading Global Update: Agent Core ({base_params_count} layers) ONLY")
        else:
            logger.error(
                "Parameter mismatch: expected %s (base) or %s (w/ gen), but received %s",
                base_params_count,
                expected_with_gen,
                received_count,
            )
            raise ValueError("Mismatched number of parameters received from server")

        # --- Load Base Networks ---
        current_idx = 0

        # 1. Prior
        prior_params = parameters[current_idx : current_idx + num_prior]
        self.prior_net.load_state_dict(
            OrderedDict(
                zip(
                    prior_keys,
                    [torch.tensor(p, device=self.device) for p in prior_params],
                    strict=True,
                )
            )
        )
        current_idx += num_prior

        # 2. Recognition
        recog_params = parameters[current_idx : current_idx + num_recog]
        self.recognition_net.load_state_dict(
            OrderedDict(
                zip(
                    recog_keys,
                    [torch.tensor(p, device=self.device) for p in recog_params],
                    strict=True,
                )
            )
        )
        current_idx += num_recog

        # 3. Main Q
        main_params = parameters[current_idx : current_idx + num_main]
        self.value_net_main.load_state_dict(
            OrderedDict(
                zip(
                    main_q_keys,
                    [torch.tensor(p, device=self.device) for p in main_params],
                    strict=True,
                )
            )
        )
        current_idx += num_main

        # --- Load Generator (Conditional) ---
        if self.generation_net is not None:
            if received_count == expected_with_gen:
                gen_params = parameters[current_idx : current_idx + num_gen]
                self.generation_net.load_state_dict(
                    OrderedDict(
                        zip(
                            gen_keys,
                            [torch.tensor(p, device=self.device) for p in gen_params],
                            strict=True,
                        )
                    )
                )
            else:
                logger.debug(
                    "Local agent has generator, but server sent no weights. Keeping local generator as-is."
                )

        if hard_target_update:
            logger.debug("Performing hard update of target network post-aggregation")
            self.hard_update_target_network()

    def train_generation_network(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        generator_cfg: DictConfig,
    ) -> dict[str, float]:
        """
        Train the generation network on correctly classified samples.
        Now includes detailed logging every 5 epochs using logger.info.
        """
        # Safety check
        if self.generation_net is None:
            return {}

        features = features.detach().cpu().float()
        labels = labels.detach().cpu().long()

        # Configuration loading with fallbacks
        gen_batch_size = int(generator_cfg.batch_size)
        gen_lr = float(generator_cfg.lr)
        gen_rounds = max(1, int(generator_cfg.rounds))

        gen_epochs = max(1, int(generator_cfg.epochs_per_round))
        min_correct = int(generator_cfg.min_correct_samples)

        # Filter for correctly classified samples
        dataset = TensorDataset(features, labels)
        full_loader = DataLoader(dataset, batch_size=gen_batch_size, shuffle=False)

        correct_mask_parts: list[torch.Tensor] = []
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
            return {}

        correct_mask = torch.cat(correct_mask_parts, dim=0)
        num_correct = int(correct_mask.sum().item())
        total_samples = int(features.shape[0])

        # Check threshold
        if num_correct < min_correct:
            logger.warning(
                f"Generator training skipped: {num_correct}/{total_samples} correct (< {min_correct})"
            )
            return {
                "generator_samples": float(num_correct),
                "generator_correct_frac": num_correct / max(1, total_samples),
            }

        # Prepare training
        filtered_dataset = TensorDataset(features[correct_mask], labels[correct_mask])
        train_loader = DataLoader(filtered_dataset, batch_size=gen_batch_size, shuffle=True)

        use_directml_safe = self.device.type in {"dml", "directml", "privateuseone"}
        adam_cls = DirectMLAdam if use_directml_safe else optim.Adam

        # Safe optimizer initialization
        if adam_cls is optim.Adam:
            optimizer = adam_cls(self.generation_net.parameters(), lr=gen_lr, foreach=False)
        else:
            optimizer = adam_cls(self.generation_net.parameters(), lr=gen_lr)

        loss_fn = nn.MSELoss()

        self.generation_net.train()
        self.recognition_net.eval()

        logger.info(f"--- Generator Training Start (Samples: {num_correct}) ---")

        last_epoch_loss = 0.0

        for round_idx in range(1, gen_rounds + 1):
            for epoch in range(1, gen_epochs + 1):
                total_loss = 0.0
                batch_count = 0

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
                    batch_count += 1

                last_epoch_loss = total_loss / max(1, batch_count)

                # --- LOGGING EVERY 5 EPOCHS ---
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"   > [Round {round_idx}] Epoch {epoch:02d}/{gen_epochs} | "
                        f"Loss (MSE): {last_epoch_loss:.6f} | "
                        f"Batch Count: {batch_count}"
                    )

        self.generation_net.eval()
        return {
            "generator_loss": float(last_epoch_loss),
            "generator_samples": float(num_correct),
            "generator_correct_frac": num_correct / max(1, total_samples),
        }
