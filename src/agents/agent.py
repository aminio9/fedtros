import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
<<<<<<< HEAD
=======
import torch.nn.functional as F
>>>>>>> ea28efe (Initial commit with updated source code)
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

<<<<<<< HEAD
from src.models.models import OpenSetQChainModelFactory, ValueNetwork
from src.utils.utils import (
    calculate_kl_divergence,
    calculate_kl_divergence_raw,
=======
from src.models.cvae_dqn import OpenSetQChainModelFactory, ValueNetwork
from src.training.losses import (
    center_compactness_loss,
    diagonal_gaussian_kl,
    focal_cross_entropy_loss,
    kl_warmup_weight,
    smooth_reconstruction_loss,
    supervised_contrastive_loss,
)
from src.utils.imbalance import compute_class_weights
from src.utils.utils import (
>>>>>>> ea28efe (Initial commit with updated source code)
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


<<<<<<< HEAD
=======
def _cfg_float(cfg: object, path: str, default: float) -> float:
    value = _cfg_value(cfg, path, default)
    return float(default if value is None else value)


def _cfg_int(cfg: object, path: str, default: int) -> int:
    value = _cfg_value(cfg, path, default)
    return int(default if value is None else value)


def _cfg_bool(cfg: object, path: str, default: bool) -> bool:
    value = _cfg_value(cfg, path, default)
    return bool(default if value is None else value)


def _cfg_str(cfg: object, path: str, default: str) -> str:
    value = _cfg_value(cfg, path, default)
    return str(default if value is None else value)


def _cfg_value(cfg: object, path: str, default: object | None = None) -> object | None:
    value = cfg
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return default
    return value


>>>>>>> ea28efe (Initial commit with updated source code)
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
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger("Agent")
        self.train_cfg = train_cfg
        self.device = device
        self.action_dim = model_factory.num_actions
        self.gamma = train_cfg.gamma
        self.use_double_dqn = train_cfg.use_double_dqn
        self.use_prior_kl_raw = bool(train_cfg.prior_kl_raw)
<<<<<<< HEAD
        self._prior_kl_fn = (
            calculate_kl_divergence_raw if self.use_prior_kl_raw else calculate_kl_divergence
        )
        self.prior_grad_clip_norm = train_cfg.prior_grad_clip_norm
=======
        self.prior_grad_clip_norm = train_cfg.prior_grad_clip_norm
        self.q_grad_clip_norm = getattr(train_cfg, "q_grad_clip_norm", 1.0)
        self.train_step_index = 0
        self.local_class_counts: torch.Tensor | None = None
        self.local_class_mask: torch.Tensor | None = None
        self.missing_class_mask_enabled = _cfg_bool(
            train_cfg, "missing_class_gradient.enabled", False
        )
        self.missing_class_mask_value = _cfg_float(
            train_cfg, "missing_class_gradient.mask_value", -20.0
        )
        self.kl_free_nats = _cfg_float(train_cfg, "kl.free_nats", 0.25)
        self.kl_warmup_steps = _cfg_int(train_cfg, "kl.warmup_steps", 100)
        self.classification_loss_name = _cfg_str(
            train_cfg, "classification_loss.name", "focal"
        ).lower()
        self.focal_gamma = _cfg_float(train_cfg, "classification_loss.focal_gamma", 2.0)
        self.use_class_weights = _cfg_bool(
            train_cfg, "classification_loss.use_class_weights", True
        )
        self.optimizer_name = _cfg_str(train_cfg, "optimizer_name", "adamw").lower()
        self.optimizer_weight_decay = _cfg_float(train_cfg, "optimizer_weight_decay", 1e-4)
        self.optimizer_eps = _cfg_float(train_cfg, "optimizer_eps", 1e-8)
        self.optimizer_betas = self._optimizer_betas(train_cfg)
        self.loss_weights = {
            "prior_kl": _cfg_float(train_cfg, "loss_weights.prior_kl", 1.0),
            "q_td": _cfg_float(train_cfg, "loss_weights.q_td", 1.0),
            "bandit_q": _cfg_float(train_cfg, "loss_weights.bandit_q", 0.0),
            "classification": _cfg_float(train_cfg, "loss_weights.classification", 0.0),
            "generator_reconstruction": _cfg_float(
                train_cfg, "loss_weights.generator_reconstruction", 1.0
            ),
            "proximal": _cfg_float(train_cfg, "loss_weights.proximal", 1.0),
        }
>>>>>>> ea28efe (Initial commit with updated source code)

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
        self._proximal_reference: dict[str, list[torch.Tensor]] | None = None

        # Sync target net initially
        self.value_net_target.load_state_dict(self.value_net_main.state_dict())
        self.value_net_target.eval()

<<<<<<< HEAD
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
=======
        self._use_directml_safe_optimizer = device.type in {"dml", "directml", "privateuseone"}
        self.optimizer_prior = self._make_optimizer(
            self.prior_net.parameters(),
            lr=float(train_cfg.lr_prior),
        )
        # RL optimizer trains *both* the local RecognitionNet and the global MainQNet
        rl_params = list(self.recognition_net.parameters()) + list(self.value_net_main.parameters())
        self.optimizer_q_rl = self._make_optimizer(rl_params, lr=float(train_cfg.lr_q_rl))
>>>>>>> ea28efe (Initial commit with updated source code)

        self.td_loss_fn = nn.SmoothL1Loss()
        self._capture_proximal_reference()
        self.logger.debug("Agent initialized with Double-DQN: %s", self.use_double_dqn)

<<<<<<< HEAD
=======
    @staticmethod
    def _optimizer_betas(train_cfg: DictConfig) -> tuple[float, float]:
        raw_betas = _cfg_value(train_cfg, "optimizer_betas", None)
        if raw_betas is None:
            return (0.9, 0.95)
        betas = [float(value) for value in raw_betas]
        if len(betas) != 2:
            return (0.9, 0.95)
        return (betas[0], betas[1])

    def _make_optimizer(self, params, *, lr: float) -> optim.Optimizer:
        if self._use_directml_safe_optimizer:
            return DirectMLAdam(
                params,
                lr=lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
            )
        if self.optimizer_name == "adam":
            return optim.Adam(
                params,
                lr=lr,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
                weight_decay=self.optimizer_weight_decay,
                foreach=False,
            )
        return optim.AdamW(
            params,
            lr=lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            foreach=False,
        )

>>>>>>> ea28efe (Initial commit with updated source code)
    def to(self, device: torch.device | str) -> "Agent":
        """Move the agent and optimizer state to a new device."""
        target_device = torch.device(device)
        if target_device == self.device:
            return self

        self.value_network.to(target_device)
        if self.generation_net is not None:
            self.generation_net.to(target_device)
        self._move_optimizer_state(self.optimizer_prior, target_device)
        self._move_optimizer_state(self.optimizer_q_rl, target_device)
<<<<<<< HEAD
        self.device = target_device
        return self

=======
        if self.local_class_counts is not None:
            self.local_class_counts = self.local_class_counts.to(target_device)
        if self.local_class_mask is not None:
            self.local_class_mask = self.local_class_mask.to(target_device)
        self.device = target_device
        return self

    def set_local_class_counts(self, counts: list[int] | torch.Tensor) -> None:
        """Record local label support so local losses do not train absent classes."""
        counts_t = torch.as_tensor(counts, dtype=torch.float32, device=self.device)
        if counts_t.numel() < self.action_dim:
            counts_t = F.pad(counts_t, (0, self.action_dim - counts_t.numel()))
        counts_t = counts_t[: self.action_dim]
        self.local_class_counts = counts_t
        self.local_class_mask = counts_t > 0

    def _mask_absent_class_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask absent local classes for client-side supervised classification loss only."""
        if not self.missing_class_mask_enabled:
            return logits
        if self.local_class_mask is None:
            return logits
        mask = self.local_class_mask.to(device=logits.device)
        if bool(mask.all()):
            return logits
        return logits.masked_fill(~mask.view(1, -1), float(self.missing_class_mask_value))

>>>>>>> ea28efe (Initial commit with updated source code)
    @staticmethod
    def _move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)
                elif isinstance(value, list):
                    state[key] = [
                        item.to(device) if torch.is_tensor(item) else item for item in value
                    ]
                elif isinstance(value, tuple):
                    state[key] = tuple(
                        item.to(device) if torch.is_tensor(item) else item for item in value
                    )

    @torch.no_grad()
    def _bootstrap_target(self, next_states_s: torch.Tensor) -> torch.Tensor:
        """
        Build y_t targets using the PRIOR for z_{t+1}.
        (Eq. 9 or 13, but using prior for z)
        """
<<<<<<< HEAD
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
=======
        prior_was_training = self.prior_net.training
        main_was_training = self.value_net_main.training
        target_was_training = self.value_net_target.training

        try:
            self.prior_net.eval()
            self.value_net_main.eval()
            self.value_net_target.eval()

            next_mu_p, next_log_var_p = self.prior_net(next_states_s)
            z_next = reparameterization_trick(next_mu_p, next_log_var_p)

            if self.use_double_dqn:
                # Double-DQN (Eq. 13)
                q_main_next = self.value_net_main(z_next, next_states_s)  # [B, A]
                a_star = q_main_next.argmax(dim=1, keepdim=True)  # [B, 1]
                q_target_next = self.value_net_target(z_next, next_states_s).gather(
                    1, a_star
                )  # [B,1]
            else:
                # Standard DQN (Eq. 9)
                q_target_next = self.value_net_target(
                    z_next, next_states_s
                ).max(dim=1, keepdim=True)[0]  # [B,1]
            return q_target_next
        finally:
            self.prior_net.train(prior_was_training)
            self.value_net_main.train(main_was_training)
            self.value_net_target.train(target_was_training)
>>>>>>> ea28efe (Initial commit with updated source code)

    def train_step(
        self,
        batch: tuple[torch.Tensor, ...],
        *,
        proximal_mu: float = 0.0,
<<<<<<< HEAD
    ) -> tuple[float, float, float, float]:
        """
        Performs one full training step (Prior update + RL update).
        batch = (states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t)
        Returns: (td_loss_item, kl_loss_item, prox_loss_item, avg_q_item)
=======
    ) -> dict[str, float]:
        """
        Performs one full training step (Prior update + RL update).
        batch = (states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t)
        Returns a scalar dictionary with explicit loss, gradient, Q, KL, and LR metrics.
>>>>>>> ea28efe (Initial commit with updated source code)
        """
        states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t = batch

        actions_a_t = _ensure_action_index(actions_a_t, self.action_dim)
        true_actions_a_t = _ensure_action_index(true_actions_a_t, self.action_dim)
<<<<<<< HEAD
=======
        step_index = self.train_step_index + 1
>>>>>>> ea28efe (Initial commit with updated source code)
        proximal_mu = float(proximal_mu)
        prox_loss_total = torch.zeros((), device=self.device)

        # ========== 1) PRIOR update: KL(q||p) using TRUE labels (Eq. 5) ==========
        self.prior_net.train()
        self.recognition_net.eval()  # Recognition net is frozen for this part

        with torch.no_grad():
            mu_q_T, log_var_q_T = self.recognition_net(
                states_s, true_actions_a_t
            )  # q_phi(z|s, a_T)
        mu_p, log_var_p = self.prior_net(states_s)  # p_theta(z|s)

<<<<<<< HEAD
        kl_loss = self._prior_kl_fn(mu_q_T, log_var_q_T, mu_p, log_var_p)
        kl_objective = kl_loss
        if proximal_mu > 0.0:
            prior_prox = self._proximal_penalty("prior_net")
            prox_loss_total = prox_loss_total + 0.5 * proximal_mu * prior_prox
            kl_objective = kl_objective + 0.5 * proximal_mu * prior_prox

        self.optimizer_prior.zero_grad()
        kl_objective.backward()
        if self.prior_grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
=======
        kl_per_sample_raw = diagonal_gaussian_kl(
            mu_q_T,
            log_var_q_T,
            mu_p,
            log_var_p,
            free_nats=0.0,
            reduce="none",
            clamp_logvar=not self.use_prior_kl_raw,
        )
        kl_per_sample = diagonal_gaussian_kl(
            mu_q_T,
            log_var_q_T,
            mu_p,
            log_var_p,
            free_nats=self.kl_free_nats,
            reduce="none",
            clamp_logvar=not self.use_prior_kl_raw,
        )
        kl_loss = kl_per_sample.mean()
        kl_warmup = kl_warmup_weight(step_index, self.kl_warmup_steps)
        labels = true_actions_a_t.squeeze(1)
        supcon_lambda = _cfg_float(
            self.train_cfg, "auxiliary_losses.supervised_contrastive_lambda", 0.0
        )
        center_lambda = _cfg_float(self.train_cfg, "auxiliary_losses.center_loss_lambda", 0.0)
        supcon_temperature = _cfg_float(
            self.train_cfg, "auxiliary_losses.supervised_contrastive_temperature", 0.1
        )
        supcon_loss = torch.zeros((), device=self.device)
        center_loss = torch.zeros((), device=self.device)
        if supcon_lambda > 0.0:
            supcon_loss = supervised_contrastive_loss(
                mu_p,
                labels,
                temperature=supcon_temperature,
            )
        if center_lambda > 0.0:
            center_loss = center_compactness_loss(mu_p, labels)
        kl_objective = self.loss_weights["prior_kl"] * kl_warmup * kl_loss
        kl_objective = kl_objective + supcon_lambda * supcon_loss + center_lambda * center_loss
        if proximal_mu > 0.0:
            prior_prox = self._proximal_penalty("prior_net")
            prior_prox_weighted = self.loss_weights["proximal"] * 0.5 * proximal_mu * prior_prox
            prox_loss_total = prox_loss_total + prior_prox_weighted
            kl_objective = kl_objective + prior_prox_weighted

        self.optimizer_prior.zero_grad()
        kl_objective.backward()
        prior_grad_norm = torch.zeros((), device=self.device)
        if self.prior_grad_clip_norm is not None:
            prior_grad_norm = torch.nn.utils.clip_grad_norm_(
>>>>>>> ea28efe (Initial commit with updated source code)
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

<<<<<<< HEAD
        # Calculate TD Loss (Eq. 11)
        td_loss = self.td_loss_fn(q_pred, y_t)
        td_objective = td_loss
=======
        with torch.no_grad():
            mu_p_cls, _ = self.prior_net(states_s)
        cls_logits = self.value_net_main(mu_p_cls.detach(), states_s)
        cls_logits_for_loss = self._mask_absent_class_logits(cls_logits)
        class_weights = self._batch_class_weights(true_actions_a_t.squeeze(1))
        if self.classification_loss_name == "cross_entropy":
            cls_loss = F.cross_entropy(
                cls_logits_for_loss,
                true_actions_a_t.squeeze(1),
                weight=class_weights,
            )
        else:
            cls_loss = focal_cross_entropy_loss(
                cls_logits_for_loss,
                true_actions_a_t.squeeze(1),
                class_weights=class_weights,
                gamma=self.focal_gamma,
            )

        # Contextual-bandit full-action Q supervision. The traffic sample order is
        # not a causal MDP transition, so gamma=0 in GLOW, while the old TD path
        # remains available for backward-compatible configs.
        with torch.no_grad():
            mu_p_bandit, _ = self.prior_net(states_s)
        q_bandit_all = self.value_net_main(mu_p_bandit.detach(), states_s)
        if self.local_class_mask is not None and self.missing_class_mask_enabled:
            present_mask = self.local_class_mask.to(device=self.device, dtype=torch.bool)
        else:
            present_mask = torch.ones(self.action_dim, dtype=torch.bool, device=self.device)
        if not bool(present_mask.any()):
            present_mask = torch.ones(self.action_dim, dtype=torch.bool, device=self.device)

        reward_correct = _cfg_float(self.train_cfg, "reward.correct", 1.0)
        reward_incorrect = _cfg_float(self.train_cfg, "reward.incorrect", -1.0)
        bandit_targets = torch.zeros_like(q_bandit_all)
        bandit_targets[:, present_mask] = float(reward_incorrect)
        bandit_targets.scatter_(1, labels.view(-1, 1), float(reward_correct))
        bandit_loss_matrix = F.smooth_l1_loss(
            q_bandit_all,
            bandit_targets,
            reduction="none",
        )
        bandit_loss = bandit_loss_matrix[:, present_mask].mean()

        # Calculate TD Loss (Eq. 11)
        td_loss = self.td_loss_fn(q_pred, y_t)
        td_objective = (
            self.loss_weights["q_td"] * td_loss
            + self.loss_weights["bandit_q"] * bandit_loss
            + self.loss_weights["classification"] * cls_loss
        )
>>>>>>> ea28efe (Initial commit with updated source code)
        if proximal_mu > 0.0:
            rl_prox = self._proximal_penalty("recognition_net") + self._proximal_penalty(
                "value_net_main"
            )
<<<<<<< HEAD
            prox_loss_total = prox_loss_total + 0.5 * proximal_mu * rl_prox
            td_objective = td_objective + 0.5 * proximal_mu * rl_prox
=======
            rl_prox_weighted = self.loss_weights["proximal"] * 0.5 * proximal_mu * rl_prox
            prox_loss_total = prox_loss_total + rl_prox_weighted
            td_objective = td_objective + rl_prox_weighted
>>>>>>> ea28efe (Initial commit with updated source code)

        # Optimize both recognition and main_q nets
        self.optimizer_q_rl.zero_grad()
        td_objective.backward()
<<<<<<< HEAD
        torch.nn.utils.clip_grad_norm_(
            list(self.recognition_net.parameters()) + list(self.value_net_main.parameters()),
            max_norm=1.0,  # Common practice for DQN stability
        )
        self.optimizer_q_rl.step()

        avg_q = q_pred.mean().item()
        return td_loss.item(), kl_loss.item(), prox_loss_total.item(), avg_q
=======
        q_params = list(self.recognition_net.parameters()) + list(self.value_net_main.parameters())
        if self.q_grad_clip_norm is not None:
            q_grad_norm = torch.nn.utils.clip_grad_norm_(
                q_params,
                max_norm=float(self.q_grad_clip_norm),
            )
        else:
            q_grad_norm = self._grad_norm(q_params)
        self.optimizer_q_rl.step()

        q_pred_detached = q_pred.detach()
        q_values_detached = q_values_all.detach()
        self.train_step_index = step_index
        loss_total = kl_objective.detach() + td_objective.detach()
        return {
            "loss/total": float(loss_total.item()),
            "loss/prior_kl": float(kl_loss.item()),
            "loss/prior_kl_raw": float(kl_per_sample_raw.mean().item()),
            "loss/prior_kl_weighted": float(
                (self.loss_weights["prior_kl"] * kl_warmup * kl_loss).item()
            ),
            "loss/prior_kl_warmup": float(kl_warmup),
            "loss/supervised_contrastive": float(supcon_loss.item()),
            "loss/supervised_contrastive_weighted": float((supcon_lambda * supcon_loss).item()),
            "loss/center_compactness": float(center_loss.item()),
            "loss/center_compactness_weighted": float((center_lambda * center_loss).item()),
            "loss/q_td": float(td_loss.item()),
            "loss/q_td_weighted": float((self.loss_weights["q_td"] * td_loss).item()),
            "loss/bandit_q": float(bandit_loss.item()),
            "loss/bandit_q_weighted": float((self.loss_weights["bandit_q"] * bandit_loss).item()),
            "loss/classification": float(cls_loss.item()),
            "loss/classification_weighted": float(
                (self.loss_weights["classification"] * cls_loss).item()
            ),
            "loss/proximal": float(prox_loss_total.item()),
            "gradient/prior_norm": float(prior_grad_norm.item()),
            "gradient/q_norm": float(q_grad_norm.item()),
            "kl/mean": float(kl_per_sample.mean().item()),
            "kl/std": float(kl_per_sample.std(unbiased=False).item()),
            "q/pred_mean": float(q_pred_detached.mean().item()),
            "q/pred_std": float(q_pred_detached.std(unbiased=False).item()),
            "q/value_mean": float(q_values_detached.mean().item()),
            "q/value_std": float(q_values_detached.std(unbiased=False).item()),
            "q/value_min": float(q_values_detached.min().item()),
            "q/value_max": float(q_values_detached.max().item()),
            "q/target_mean": float(y_t.detach().mean().item()),
            "q/target_std": float(y_t.detach().std(unbiased=False).item()),
            "lr/prior": self._optimizer_lr(self.optimizer_prior),
            "lr/q_rl": self._optimizer_lr(self.optimizer_q_rl),
            "local_class_coverage_count": float(present_mask.sum().item()),
            "missing_class_mask_enabled": float(self.missing_class_mask_enabled),
        }
>>>>>>> ea28efe (Initial commit with updated source code)

    def update_target_network(self, tau: float):
        """Soft update the target network (Eq. 10)."""
        self.logger.debug("Performing soft target update with tau=%s", tau)
        soft_update_target_network(self.value_net_main, self.value_net_target, tau)

    def hard_update_target_network(self):
        """Hard update (copy) the target network."""
        self.logger.debug("Performing hard target update")
        self.value_net_target.load_state_dict(self.value_net_main.state_dict())

<<<<<<< HEAD
=======
    @staticmethod
    def _optimizer_lr(optimizer: optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0

    @staticmethod
    def _grad_norm(params: list[torch.nn.Parameter]) -> torch.Tensor:
        norms = [
            param.grad.detach().norm(2)
            for param in params
            if param.grad is not None
        ]
        if not norms:
            device = params[0].device if params else torch.device("cpu")
            return torch.zeros((), device=device)
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)

    def _batch_class_weights(self, labels: torch.Tensor) -> torch.Tensor | None:
        if not self.use_class_weights:
            return None
        imbalance_cfg = getattr(self.train_cfg, "imbalance", None)
        weights = compute_class_weights(
            labels.detach().cpu(),
            num_classes=self.action_dim,
            mode=_cfg_str(imbalance_cfg, "weight_mode", "inverse_frequency"),
            beta=_cfg_float(imbalance_cfg, "effective_number_beta", 0.999),
            min_weight=_cfg_float(imbalance_cfg, "min_weight", 0.2),
            max_weight=_cfg_float(imbalance_cfg, "max_weight", 5.0),
            normalize=_cfg_str(imbalance_cfg, "normalize", "mean"),
        )
        return weights.to(device=self.device)

>>>>>>> ea28efe (Initial commit with updated source code)
    def _capture_proximal_reference(self) -> None:
        """Store the current federated parameters as the FedProx anchor."""
        self._proximal_reference = {
            "prior_net": [param.detach().clone() for param in self.prior_net.parameters()],
            "recognition_net": [
                param.detach().clone() for param in self.recognition_net.parameters()
            ],
            "value_net_main": [param.detach().clone() for param in self.value_net_main.parameters()],
        }
        if self.generation_net is not None:
            self._proximal_reference["generation_net"] = [
                param.detach().clone() for param in self.generation_net.parameters()
            ]

    def _proximal_penalty(self, module_name: str) -> torch.Tensor:
        """Compute the squared distance to the saved global reference for one module."""
        if not self._proximal_reference or module_name not in self._proximal_reference:
            return torch.zeros((), device=self.device)

        module = getattr(self, module_name)
        reference_params = self._proximal_reference[module_name]
        penalty = torch.zeros((), device=self.device)
        for current_param, reference_param in zip(module.parameters(), reference_params, strict=True):
            penalty = penalty + (current_param - reference_param.to(current_param.device)).pow(2).sum()
        return penalty

    # --- METHODS FOR FEDERATED LEARNING ---

    def get_federated_parameters(self) -> list[np.ndarray]:
        """
        Get the parameters to federate (Prior + Recognition + MainQ + Generation).
        IMPORTANT: Includes Generator params if the network exists, even if training is disabled.
        This ensures the server can save the full model structure.
        """
        self.logger.debug("Getting federated parameters (Prior + Recognition + MainQ + Generation)")

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
        self.logger.debug("Setting federated parameters")

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
            self.logger.info(
                f"   > Loading Global Update: Agent Core ({base_params_count} layers) + Generator ({num_gen} layers)"
            )
        elif received_count == base_params_count:
            self.logger.info(f"   > Loading Global Update: Agent Core ({base_params_count} layers) ONLY")
        else:
            self.logger.error(
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
                self.logger.debug(
                    "Local agent has generator, but server sent no weights. Keeping local generator as-is."
                )

        if hard_target_update:
            self.logger.debug("Performing hard update of target network post-aggregation")
            self.hard_update_target_network()
        self._capture_proximal_reference()

    def train_generation_network(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        generator_cfg: DictConfig,
        proximal_mu: float = 0.0,
        logger: logging.Logger | None = None,
    ) -> dict[str, float]:
        """
        Train the generation network on correctly classified samples.
        Now includes detailed logging every 5 epochs using logger.info.
        """
        active_logger = logger or logging.getLogger("Agent")

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
        proximal_mu = float(proximal_mu)

        # Filter for correctly classified samples
        dataset = TensorDataset(features, labels)
        full_loader = DataLoader(dataset, batch_size=gen_batch_size, shuffle=False)

        correct_mask_parts: list[torch.Tensor] = []
<<<<<<< HEAD
=======
        prior_was_training = self.prior_net.training
        recognition_was_training = self.recognition_net.training
        q_was_training = self.value_net_main.training
        generator_was_training = self.generation_net.training
>>>>>>> ea28efe (Initial commit with updated source code)
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
<<<<<<< HEAD
=======
            self.prior_net.train(prior_was_training)
            self.recognition_net.train(recognition_was_training)
            self.value_net_main.train(q_was_training)
            self.generation_net.train(generator_was_training)
>>>>>>> ea28efe (Initial commit with updated source code)
            return {}

        correct_mask = torch.cat(correct_mask_parts, dim=0)
        num_correct = int(correct_mask.sum().item())
        total_samples = int(features.shape[0])

        # Check threshold
        if num_correct < min_correct:
            active_logger.warning(
                f"Generator training skipped: {num_correct}/{total_samples} correct (< {min_correct})"
            )
<<<<<<< HEAD
=======
            self.prior_net.train(prior_was_training)
            self.recognition_net.train(recognition_was_training)
            self.value_net_main.train(q_was_training)
            self.generation_net.train(generator_was_training)
>>>>>>> ea28efe (Initial commit with updated source code)
            return {
                "generator_samples": float(num_correct),
                "generator_correct_frac": num_correct / max(1, total_samples),
            }

        # Prepare training
        filtered_dataset = TensorDataset(features[correct_mask], labels[correct_mask])
        train_loader = DataLoader(filtered_dataset, batch_size=gen_batch_size, shuffle=True)

<<<<<<< HEAD
        use_directml_safe = self.device.type in {"dml", "directml", "privateuseone"}
        adam_cls = DirectMLAdam if use_directml_safe else optim.Adam

        # Safe optimizer initialization
        if adam_cls is optim.Adam:
            optimizer = adam_cls(self.generation_net.parameters(), lr=gen_lr, foreach=False)
        else:
            optimizer = adam_cls(self.generation_net.parameters(), lr=gen_lr)

        loss_fn = nn.MSELoss()
=======
        optimizer = self._make_optimizer(self.generation_net.parameters(), lr=gen_lr)
        recon_beta = _cfg_float(generator_cfg, "reconstruction_beta", 1.0)
        generator_grad_clip_norm = _cfg_value(generator_cfg, "grad_clip_norm", None)
>>>>>>> ea28efe (Initial commit with updated source code)

        self.generation_net.train()
        self.recognition_net.eval()

        active_logger.info(f"--- Generator Training Start (Samples: {num_correct}) ---")

        last_epoch_loss = 0.0
<<<<<<< HEAD
        last_epoch_prox_loss = 0.0
=======
        last_epoch_weighted_loss = 0.0
        last_epoch_prox_loss = 0.0
        last_epoch_grad_norm = 0.0
>>>>>>> ea28efe (Initial commit with updated source code)

        for round_idx in range(1, gen_rounds + 1):
            for epoch in range(1, gen_epochs + 1):
                total_loss = 0.0
<<<<<<< HEAD
                total_prox_loss = 0.0
=======
                total_weighted_loss = 0.0
                total_prox_loss = 0.0
                total_grad_norm = 0.0
>>>>>>> ea28efe (Initial commit with updated source code)
                batch_count = 0

                for states_s, true_actions in train_loader:
                    states_s = states_s.to(self.device)
                    true_actions = true_actions.to(self.device)

                    with torch.no_grad():
                        mu_q, log_var_q = self.recognition_net(states_s, true_actions)
                        latent_z = reparameterization_trick(mu_q, log_var_q)

                    recon = self.generation_net(latent_z, true_actions)
<<<<<<< HEAD
                    mse_loss = loss_fn(recon, states_s)
                    loss = mse_loss
                    prox_loss = torch.zeros((), device=self.device)
                    if proximal_mu > 0.0:
                        prox_loss = 0.5 * proximal_mu * self._proximal_penalty("generation_net")
=======
                    reconstruction_loss = smooth_reconstruction_loss(
                        recon,
                        states_s,
                        beta=recon_beta,
                    )
                    weighted_recon_loss = (
                        self.loss_weights["generator_reconstruction"] * reconstruction_loss
                    )
                    loss = weighted_recon_loss
                    prox_loss = torch.zeros((), device=self.device)
                    if proximal_mu > 0.0:
                        prox_loss = (
                            self.loss_weights["proximal"]
                            * 0.5
                            * proximal_mu
                            * self._proximal_penalty("generation_net")
                        )
>>>>>>> ea28efe (Initial commit with updated source code)
                        loss = loss + prox_loss

                    optimizer.zero_grad()
                    loss.backward()
<<<<<<< HEAD
                    optimizer.step()

                    total_loss += mse_loss.item()
                    total_prox_loss += prox_loss.item()
                    batch_count += 1

                last_epoch_loss = total_loss / max(1, batch_count)
                last_epoch_prox_loss = total_prox_loss / max(1, batch_count)
=======
                    generator_params = list(self.generation_net.parameters())
                    if generator_grad_clip_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            generator_params,
                            max_norm=float(generator_grad_clip_norm),
                        )
                    else:
                        grad_norm = self._grad_norm(generator_params)
                    optimizer.step()

                    total_loss += reconstruction_loss.item()
                    total_weighted_loss += loss.item()
                    total_prox_loss += prox_loss.item()
                    total_grad_norm += grad_norm.item()
                    batch_count += 1

                last_epoch_loss = total_loss / max(1, batch_count)
                last_epoch_weighted_loss = total_weighted_loss / max(1, batch_count)
                last_epoch_prox_loss = total_prox_loss / max(1, batch_count)
                last_epoch_grad_norm = total_grad_norm / max(1, batch_count)
>>>>>>> ea28efe (Initial commit with updated source code)

                # --- LOGGING EVERY 5 EPOCHS ---
                if epoch % 5 == 0 or epoch == 1:
                    active_logger.info(
                        f"   > [Round {round_idx}] Epoch {epoch:02d}/{gen_epochs} | "
<<<<<<< HEAD
                        f"Loss (MSE): {last_epoch_loss:.6f} | "
=======
                        f"Loss (SmoothL1): {last_epoch_loss:.6f} | "
>>>>>>> ea28efe (Initial commit with updated source code)
                        f"Prox: {last_epoch_prox_loss:.6f} | "
                        f"Batch Count: {batch_count}"
                    )

<<<<<<< HEAD
        self.generation_net.eval()
        return {
            "generator_loss": float(last_epoch_loss),
            "generator_prox_loss": float(last_epoch_prox_loss),
=======
        self.prior_net.train(prior_was_training)
        self.recognition_net.train(recognition_was_training)
        self.value_net_main.train(q_was_training)
        self.generation_net.train(generator_was_training)
        return {
            "generator_loss": float(last_epoch_loss),
            "generator_reconstruction_loss": float(last_epoch_loss),
            "generator_weighted_loss": float(last_epoch_weighted_loss),
            "generator_prox_loss": float(last_epoch_prox_loss),
            "generator_grad_norm": float(last_epoch_grad_norm),
            "generator_lr": gen_lr,
>>>>>>> ea28efe (Initial commit with updated source code)
            "generator_samples": float(num_correct),
            "generator_correct_frac": num_correct / max(1, total_samples),
        }
