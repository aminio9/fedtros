import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from src.models.models import OpenSetQChainModelFactory, ValueNetwork
from src.models.student import StudentIDSModel
from src.rl.class_balance import class_balanced_cross_entropy
from src.rl.distillation import (
    kd_temperature,
    mse_cosine_alignment,
    sentinel_bidirectional_kd_components,
)
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
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger("Agent")
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
        self._proximal_reference: dict[str, list[torch.Tensor]] | None = None

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

        # DKD-FedOS student: lightweight globally shared classifier.
        # It is always constructed so method switching through Hydra does not
        # require a different Agent class, but it is trained/federated only by
        # the dkd_fedos strategy.
        student_hidden = list(getattr(train_cfg, "dkd_student_hidden_dims", [64, 32, 16]))
        self.student_model = StudentIDSModel(
            input_dim=model_factory.state_dim,
            num_classes=model_factory.num_actions,
            hidden_dims=student_hidden,
        ).to(device)
        teacher_feature_dim = int(model_factory.latent_dim) + int(model_factory.num_actions)
        self.teacher_to_student_aligner = nn.Linear(
            teacher_feature_dim, int(self.student_model.feature_dim)
        ).to(device)
        dkd_params = list(self.student_model.parameters()) + list(self.teacher_to_student_aligner.parameters())
        dkd_lr = float(getattr(train_cfg, "dkd_student_lr", train_cfg.lr_q_rl))
        if adam_cls is optim.Adam:
            self.optimizer_dkd = adam_cls(dkd_params, lr=dkd_lr, foreach=False)
        else:
            self.optimizer_dkd = adam_cls(dkd_params, lr=dkd_lr)

        self.dkd_lambda_kd = float(getattr(train_cfg, "dkd_lambda_kd_init", 0.20))
        self.dkd_lambda_align = float(getattr(train_cfg, "dkd_lambda_align_init", 0.08))

        self.td_loss_fn = nn.SmoothL1Loss()
        self.last_prototype_loss = 0.0
        self.last_aux_ce_loss = 0.0
        self.last_dkd_task_loss = 0.0
        self.last_dkd_kd_loss = 0.0
        self.last_dkd_align_loss = 0.0
        self.last_dkd_temperature = 1.0
        self.last_dkd_agreement = 0.0
        self.last_dkd_confidence = 0.0
        self.last_dkd_align_score = 0.0
        self.last_dkd_teacher_task_loss = 0.0
        self.last_dkd_student_task_loss = 0.0
        self.last_dkd_t2s_loss = 0.0
        self.last_dkd_s2t_loss = 0.0
        self.last_dkd_teacher_batch_accuracy = 0.0
        self.last_dkd_student_batch_accuracy = 0.0
        self.last_dkd_correct_agreement = 0.0
        self.last_dkd_teacher_confidence = 0.0
        self.last_dkd_student_confidence = 0.0
        self.last_dkd_t2s_enabled = 0.0
        self.last_dkd_s2t_enabled = 0.0
        self._capture_proximal_reference()
        self.logger.debug("Agent initialized with Double-DQN: %s", self.use_double_dqn)

    def to(self, device: torch.device | str) -> "Agent":
        """Move the agent and optimizer state to a new device."""
        target_device = torch.device(device)
        if target_device == self.device:
            return self

        self.value_network.to(target_device)
        if self.generation_net is not None:
            self.generation_net.to(target_device)
        if hasattr(self, "student_model"):
            self.student_model.to(target_device)
        if hasattr(self, "teacher_to_student_aligner"):
            self.teacher_to_student_aligner.to(target_device)
        self._move_optimizer_state(self.optimizer_prior, target_device)
        self._move_optimizer_state(self.optimizer_q_rl, target_device)
        if hasattr(self, "optimizer_dkd"):
            self._move_optimizer_state(self.optimizer_dkd, target_device)
        self.device = target_device
        return self

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

    def train_step(
        self,
        batch: tuple[torch.Tensor, ...],
        *,
        proximal_mu: float = 0.0,
        global_prototypes: torch.Tensor | None = None,
        global_prototype_mask: torch.Tensor | None = None,
        prototype_lambda: float = 0.0,
        prototype_feature: str = "latent_q",
        aux_ce_weight: float = 0.0,
        aux_ce_label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        dkd_enabled: bool = False,
        dkd_round: int = 0,
        dkd_class_weights: torch.Tensor | None = None,
        dkd_present_classes: torch.Tensor | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Performs one full training step (Prior update + RL update).
        batch = (states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t)
        Returns: (td_loss_item, kl_loss_item, prox_loss_item, avg_q_item)
        """
        states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t = batch

        actions_a_t = _ensure_action_index(actions_a_t, self.action_dim)
        true_actions_a_t = _ensure_action_index(true_actions_a_t, self.action_dim)
        proximal_mu = float(proximal_mu)
        prototype_lambda = float(prototype_lambda)
        aux_ce_weight = float(aux_ce_weight)
        aux_ce_label_smoothing = float(aux_ce_label_smoothing)
        prox_loss_total = torch.zeros((), device=self.device)
        proto_loss_total = torch.zeros((), device=self.device)
        self.last_prototype_loss = 0.0
        self.last_aux_ce_loss = 0.0
        self.last_dkd_task_loss = 0.0
        self.last_dkd_kd_loss = 0.0
        self.last_dkd_align_loss = 0.0
        self.last_dkd_temperature = 1.0
        self.last_dkd_agreement = 0.0
        self.last_dkd_confidence = 0.0
        self.last_dkd_align_score = 0.0
        self.last_dkd_teacher_task_loss = 0.0
        self.last_dkd_student_task_loss = 0.0
        self.last_dkd_t2s_loss = 0.0
        self.last_dkd_s2t_loss = 0.0
        self.last_dkd_teacher_batch_accuracy = 0.0
        self.last_dkd_student_batch_accuracy = 0.0
        self.last_dkd_correct_agreement = 0.0
        self.last_dkd_teacher_confidence = 0.0
        self.last_dkd_student_confidence = 0.0
        self.last_dkd_t2s_enabled = 0.0
        self.last_dkd_s2t_enabled = 0.0

        # ========== 1) PRIOR update: KL(q||p) using TRUE labels (Eq. 5) ==========
        self.prior_net.train()
        self.recognition_net.eval()  # Recognition net is frozen for this part

        with torch.no_grad():
            mu_q_T, log_var_q_T = self.recognition_net(
                states_s, true_actions_a_t
            )  # q_phi(z|s, a_T)
        mu_p, log_var_p = self.prior_net(states_s)  # p_theta(z|s)

        kl_loss = self._prior_kl_fn(mu_q_T, log_var_q_T, mu_p, log_var_p)
        kl_objective = kl_loss
        if prototype_lambda > 0.0 and global_prototypes is not None:
            feature_name = str(prototype_feature).lower()
            if feature_name in {"latent_q", "mu_q", "latent+q", "prior_q"}:
                prior_features = F.normalize(mu_p, dim=1)
                prior_targets = global_prototypes[:, : mu_p.shape[1]].to(self.device)
            elif feature_name in {"latent", "prior", "prior_mu", "mu"}:
                prior_features = mu_p
                prior_targets = global_prototypes[:, : mu_p.shape[1]].to(self.device)
            else:
                prior_features = None
                prior_targets = None
            if prior_features is not None and prior_targets is not None:
                prior_proto_loss = self._prototype_alignment_loss(
                    prior_features, true_actions_a_t, prior_targets, global_prototype_mask
                )
                if prior_proto_loss is not None:
                    kl_objective = kl_objective + prototype_lambda * prior_proto_loss
                    proto_loss_total = proto_loss_total + prototype_lambda * prior_proto_loss.detach()
        if proximal_mu > 0.0:
            prior_prox = self._proximal_penalty("prior_net")
            prox_loss_total = prox_loss_total + 0.5 * proximal_mu * prior_prox
            kl_objective = kl_objective + 0.5 * proximal_mu * prior_prox

        self.optimizer_prior.zero_grad()
        kl_objective.backward()
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

        # Calculate TD Loss (Eq. 11).  This remains the RL objective.
        td_loss = self.td_loss_fn(q_pred, y_t)
        td_objective = td_loss

        # Auxiliary supervised classification term.  The environment is a
        # sampled data-pool bandit, so adding a small CE loss on Q-values makes
        # learning less sensitive to short-horizon TD noise and class imbalance
        # while preserving the DQN update as the main objective.
        aux_ce_loss = torch.zeros((), device=self.device)
        if aux_ce_weight > 0.0:
            ce_targets = true_actions_a_t.view(-1).long().clamp(0, self.action_dim - 1)
            ce_weights = None
            if class_weights is not None and class_weights.numel() >= self.action_dim:
                ce_weights = class_weights.to(self.device, dtype=q_values_all.dtype)[: self.action_dim]
                ce_weights = ce_weights / ce_weights.mean().clamp_min(1e-8)
            aux_ce_loss = F.cross_entropy(
                q_values_all,
                ce_targets,
                weight=ce_weights,
                label_smoothing=max(aux_ce_label_smoothing, 0.0),
            )
            td_objective = td_objective + aux_ce_weight * aux_ce_loss

        # DKD-FedOS v2: keep Sentinel's local/global roles separated.
        #
        # Stage A happens inside the teacher RL objective below: the CVAE-DQN
        # teacher learns from local labels/rewards.  Student training and
        # student->teacher global transfer happen after the teacher local step,
        # so missing-class protection can block harmful local gradients without
        # blocking useful KD from the global student.
        dkd_teacher_task_loss = torch.zeros((), device=self.device)
        dkd_student_task_loss = torch.zeros((), device=self.device)
        dkd_t2s = torch.zeros((), device=self.device)
        dkd_s2t = torch.zeros((), device=self.device)
        dkd_align = torch.zeros((), device=self.device)
        dkd_task_loss = torch.zeros((), device=self.device)
        dkd_kd = torch.zeros((), device=self.device)
        dkd_temperature_value = 1.0
        dkd_stats: dict[str, float] = {
            "agreement": 0.0,
            "joint_confidence": 0.0,
            "teacher_batch_accuracy": 0.0,
            "student_batch_accuracy": 0.0,
            "correct_agreement": 0.0,
            "teacher_confidence": 0.0,
            "student_confidence": 0.0,
        }
        dkd_align_score = 0.0
        dkd_t2s_enabled = False
        dkd_s2t_enabled = False
        dkd_cb_weights = None
        dkd_targets = true_actions_a_t.view(-1).long().clamp(0, self.action_dim - 1)
        if dkd_enabled:
            dkd_cb_weights = dkd_class_weights if dkd_class_weights is not None else class_weights
            dkd_teacher_task_loss = self._masked_class_balanced_ce(
                q_values_all,
                dkd_targets,
                dkd_cb_weights,
                dkd_present_classes,
                label_smoothing=max(aux_ce_label_smoothing, 0.0),
            )
            teacher_task_weight = float(getattr(self.train_cfg, "dkd_teacher_task_weight", 0.5))
            td_objective = td_objective + teacher_task_weight * dkd_teacher_task_loss

        if prototype_lambda > 0.0 and global_prototypes is not None:
            proto_features = self._build_prototype_features(
                z_now, q_values_all, prototype_feature=prototype_feature
            )
            proto_targets = global_prototypes[:, : proto_features.shape[1]].to(self.device)
            rl_proto_loss = self._prototype_alignment_loss(
                proto_features, true_actions_a_t, proto_targets, global_prototype_mask
            )
            if rl_proto_loss is not None:
                td_objective = td_objective + prototype_lambda * rl_proto_loss
                proto_loss_total = proto_loss_total + prototype_lambda * rl_proto_loss.detach()
        if proximal_mu > 0.0:
            rl_prox = self._proximal_penalty("recognition_net") + self._proximal_penalty(
                "value_net_main"
            )
            prox_loss_total = prox_loss_total + 0.5 * proximal_mu * rl_prox
            td_objective = td_objective + 0.5 * proximal_mu * rl_prox

        # Stage A optimization: local CVAE-DQN teacher update.
        # Local gradients are allowed to update only locally observed output rows;
        # missing-class rows are protected from CE/TD overwrite here.
        self.optimizer_q_rl.zero_grad()
        td_objective.backward()
        if dkd_enabled and dkd_present_classes is not None:
            self._protect_absent_class_rows(dkd_present_classes, target="teacher")
        torch.nn.utils.clip_grad_norm_(
            list(self.recognition_net.parameters()) + list(self.value_net_main.parameters()),
            max_norm=1.0,
        )
        self.optimizer_q_rl.step()

        # Stage B/C optimization: Sentinel-style student update and gated global
        # student -> teacher transfer.  This is intentionally separate from the
        # local teacher backward pass so absent-class KD can remain active.
        if dkd_enabled:
            (
                dkd_student_task_loss,
                dkd_t2s,
                dkd_s2t,
                dkd_align,
                dkd_stats,
                dkd_align_score,
                dkd_temperature_value,
                dkd_t2s_enabled,
                dkd_s2t_enabled,
            ) = self._run_dkd_teacher_student_updates(
                states_s=states_s,
                labels=dkd_targets,
                class_weights=dkd_cb_weights,
                present_classes=dkd_present_classes,
                round_num=int(dkd_round),
                label_smoothing=max(aux_ce_label_smoothing, 0.0),
            )
            dkd_task_loss = 0.5 * (dkd_teacher_task_loss.detach() + dkd_student_task_loss.detach())
            dkd_kd = dkd_t2s.detach() + dkd_s2t.detach()
            self._update_dkd_lambdas(
                agreement=dkd_stats.get("agreement", 0.0),
                align_score=dkd_align_score,
                round_num=int(dkd_round),
            )
            self.last_dkd_temperature = float(dkd_temperature_value)
            self.last_dkd_agreement = float(dkd_stats.get("agreement", 0.0))
            self.last_dkd_confidence = float(dkd_stats.get("joint_confidence", 0.0))
            self.last_dkd_align_score = float(dkd_align_score)
            self.last_dkd_teacher_batch_accuracy = float(dkd_stats.get("teacher_batch_accuracy", 0.0))
            self.last_dkd_student_batch_accuracy = float(dkd_stats.get("student_batch_accuracy", 0.0))
            self.last_dkd_correct_agreement = float(dkd_stats.get("correct_agreement", 0.0))
            self.last_dkd_teacher_confidence = float(dkd_stats.get("teacher_confidence", 0.0))
            self.last_dkd_student_confidence = float(dkd_stats.get("student_confidence", 0.0))
            self.last_dkd_t2s_enabled = float(bool(dkd_t2s_enabled))
            self.last_dkd_s2t_enabled = float(bool(dkd_s2t_enabled))

        avg_q = q_pred.mean().item()
        self.last_prototype_loss = float(proto_loss_total.item())
        self.last_aux_ce_loss = float(aux_ce_loss.detach().item())
        self.last_dkd_teacher_task_loss = float(dkd_teacher_task_loss.detach().item())
        self.last_dkd_student_task_loss = float(dkd_student_task_loss.detach().item())
        self.last_dkd_task_loss = float(dkd_task_loss.detach().item())
        self.last_dkd_t2s_loss = float(dkd_t2s.detach().item())
        self.last_dkd_s2t_loss = float(dkd_s2t.detach().item())
        self.last_dkd_kd_loss = float(dkd_kd.detach().item())
        self.last_dkd_align_loss = float(dkd_align.detach().item())
        return td_loss.item(), kl_loss.item(), prox_loss_total.item(), avg_q

    def _masked_class_balanced_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None,
        present_classes: torch.Tensor | None,
        *,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Class-balanced CE that does not touch absent output rows.

        Cross entropy over all logits pushes absent classes down even when a
        client never observes them.  For non-IID clients, local label losses
        should only update locally present class rows.  Global KD is handled in
        a separate pass and can still update absent rows.
        """
        labels = labels.view(-1).long().to(logits.device).clamp(0, self.action_dim - 1)
        if (
            present_classes is None
            or not bool(getattr(self.train_cfg, "dkd_protect_absent_classes", True))
        ):
            return class_balanced_cross_entropy(
                logits,
                labels,
                weights,
                label_smoothing=label_smoothing,
            )

        present_mask = present_classes.to(logits.device).bool().view(-1)
        if present_mask.numel() < self.action_dim:
            present_mask = F.pad(present_mask, (0, self.action_dim - present_mask.numel()), value=False)
        present_mask = present_mask[: self.action_dim]
        present_idx = torch.nonzero(present_mask, as_tuple=False).view(-1)
        if present_idx.numel() <= 1:
            # With one local class, local CE has no useful class-discrimination
            # signal.  KD from the global student remains the missing-class
            # knowledge path.
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        remap = torch.full((self.action_dim,), -1, device=logits.device, dtype=torch.long)
        remap[present_idx] = torch.arange(present_idx.numel(), device=logits.device)
        valid = remap[labels] >= 0
        if not bool(valid.any().item()):
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        selected_logits = logits[valid][:, present_idx]
        selected_labels = remap[labels[valid]]
        selected_weights = None
        if weights is not None and weights.numel() >= self.action_dim:
            selected_weights = weights.to(logits.device, dtype=logits.dtype)[: self.action_dim][present_idx]
            selected_weights = selected_weights / selected_weights.mean().clamp_min(1e-8)
        return F.cross_entropy(
            selected_logits,
            selected_labels,
            weight=selected_weights,
            label_smoothing=max(float(label_smoothing), 0.0),
        )

    def _run_dkd_teacher_student_updates(
        self,
        *,
        states_s: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None,
        present_classes: torch.Tensor | None,
        round_num: int,
        label_smoothing: float,
        allow_student_to_teacher: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], float, float, bool, bool]:
        """Run Sentinel-style student update and gated global-to-teacher KD.

        Stage B updates only the student/aligner with local CBCE, teacher->student
        KD, and feature alignment.  Stage C optionally updates the teacher Q-head
        from the student without absent-class row protection, allowing missing
        class knowledge to transfer from the global student.
        """
        self.student_model.train()
        self.teacher_to_student_aligner.train()
        labels = labels.view(-1).long().to(self.device).clamp(0, self.action_dim - 1)
        temperature = kd_temperature(
            int(round_num),
            base=float(getattr(self.train_cfg, "dkd_kd_base_temperature", 3.0)),
            minimum=float(getattr(self.train_cfg, "dkd_kd_min_temperature", 1.0)),
            decay=float(getattr(self.train_cfg, "dkd_kd_decay", 0.95)),
        )
        if class_weights is not None:
            mean_weight = class_weights.to(self.device, dtype=states_s.dtype)[labels].mean().detach()
        else:
            mean_weight = torch.ones((), device=self.device, dtype=states_s.dtype)

        # Teacher snapshot after local update.  Use inference path (prior -> Q)
        # because this is what closed-set/open-set evaluation uses.
        with torch.no_grad():
            mu_teacher, _ = self.prior_net(states_s)
            teacher_logits_snapshot = self.value_net_main(mu_teacher, states_s)
            teacher_features = self._teacher_distillation_features(mu_teacher, teacher_logits_snapshot)

        student_features, student_logits = self.student_model(states_s)
        student_task_loss = self._masked_class_balanced_ce(
            student_logits,
            labels,
            class_weights,
            present_classes,
            label_smoothing=label_smoothing,
        )
        t2s_component, _s2t_component_for_stats, stats = sentinel_bidirectional_kd_components(
            teacher_logits_snapshot,
            student_logits,
            temperature=temperature,
            mean_class_weight=mean_weight,
            labels=labels,
        )
        projected_teacher = self.teacher_to_student_aligner(teacher_features.detach())
        align_loss, align_score = mse_cosine_alignment(
            projected_teacher,
            student_features,
            cosine_weight=float(getattr(self.train_cfg, "dkd_align_cos_weight", 0.5)),
            mse_weight=float(getattr(self.train_cfg, "dkd_align_mse_weight", 1.0)),
        )

        t2s_start = int(getattr(self.train_cfg, "dkd_teacher_to_student_start_round", 4))
        align_start = int(getattr(self.train_cfg, "dkd_alignment_start_round", 4))
        t2s_enabled = int(round_num) >= t2s_start
        align_enabled = int(round_num) >= align_start

        self.optimizer_dkd.zero_grad()
        student_objective = (
            float(getattr(self.train_cfg, "dkd_student_task_weight", 1.0)) * student_task_loss
        )
        if t2s_enabled:
            student_objective = student_objective + self.dkd_lambda_kd * t2s_component
        if align_enabled:
            student_objective = student_objective + self.dkd_lambda_align * align_loss
        if student_objective.requires_grad:
            student_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.student_model.parameters()) + list(self.teacher_to_student_aligner.parameters()),
                max_norm=float(getattr(self.train_cfg, "dkd_grad_clip_norm", 1.0)),
            )
            self.optimizer_dkd.step()

        # Stage C: global student -> teacher.  Crucially, this pass is not
        # protected against absent classes; it is the only path that can teach
        # missing output rows on clients with label holes.
        if allow_student_to_teacher is None:
            allow_student_to_teacher = bool(
                getattr(self.train_cfg, "dkd_update_teacher_from_student", False)
            )
        s2t_start = int(getattr(self.train_cfg, "dkd_student_to_teacher_start_round", 999))
        min_student_conf = float(getattr(self.train_cfg, "dkd_min_student_confidence", 0.0))
        s2t_enabled = (
            bool(allow_student_to_teacher)
            and int(round_num) >= s2t_start
            and stats.get("student_confidence", 0.0) >= min_student_conf
        )
        s2t_loss = torch.zeros((), device=self.device)
        if s2t_enabled:
            with torch.no_grad():
                _student_features_after, student_logits_after = self.student_model(states_s)
                mu_teacher_s2t, _ = self.prior_net(states_s)
            teacher_logits_for_kd = self.value_net_main(mu_teacher_s2t.detach(), states_s)
            _unused_t2s, s2t_component, s2t_stats = sentinel_bidirectional_kd_components(
                teacher_logits_for_kd,
                student_logits_after.detach(),
                temperature=temperature,
                mean_class_weight=mean_weight,
                labels=labels,
            )
            # In sentinel_bidirectional_kd_components arguments are
            # (teacher_logits, student_logits); the returned s2t component updates
            # the first argument toward the second argument, which is what we need.
            s2t_loss = s2t_component
            stats.update({f"s2t_{k}": v for k, v in s2t_stats.items()})
            self.optimizer_q_rl.zero_grad()
            (self.dkd_lambda_kd * s2t_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_net_main.parameters(),
                max_norm=float(getattr(self.train_cfg, "dkd_grad_clip_norm", 1.0)),
            )
            self.optimizer_q_rl.step()

        return (
            student_task_loss.detach(),
            t2s_component.detach() if t2s_enabled else torch.zeros((), device=self.device),
            s2t_loss.detach(),
            align_loss.detach() if align_enabled else torch.zeros((), device=self.device),
            stats,
            align_score,
            float(temperature),
            bool(t2s_enabled),
            bool(s2t_enabled),
        )

    def train_dkd_fedos_dataset(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        cfg_training: DictConfig,
        *,
        round_num: int,
        class_weights: torch.Tensor | None,
        present_classes: torch.Tensor | None,
        device: torch.device,
        logger: logging.Logger | None = None,
    ) -> dict[str, float]:
        """Run Sentinel Algorithm-1-style DKD training on local dataset batches.

        This is intentionally separate from replay-buffer RL.  The RL loop keeps
        the CVAE-DQN teacher stable, while this supervised loop implements the
        paper's mini-batch teacher/student distillation objective on D_i.
        """
        active_logger = logger or self.logger
        local_epochs = int(getattr(cfg_training, "dkd_local_epochs", 1))
        if local_epochs <= 0:
            return {}
        batch_size = int(getattr(cfg_training, "dkd_batch_size", getattr(cfg_training, "batch_size", 64)))
        label_smoothing = float(getattr(cfg_training, "dkd_label_smoothing", getattr(cfg_training, "aux_ce_label_smoothing", 0.0)))
        features = features.detach().float().to(device)
        labels = labels.detach().long().view(-1).to(device).clamp(0, self.action_dim - 1)
        dataset = TensorDataset(features, labels)
        if len(dataset) == 0:
            return {"dkd_dataset_train_steps": 0.0}
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        if present_classes is not None:
            present_classes = present_classes.to(device)

        totals = {
            "teacher_task": 0.0,
            "student_task": 0.0,
            "t2s": 0.0,
            "s2t": 0.0,
            "align": 0.0,
            "agreement": 0.0,
            "correct_agreement": 0.0,
            "teacher_acc": 0.0,
            "student_acc": 0.0,
            "teacher_conf": 0.0,
            "student_conf": 0.0,
            "t2s_enabled": 0.0,
            "s2t_enabled": 0.0,
        }
        train_steps = 0
        for _epoch in range(local_epochs):
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).long().view(-1).clamp(0, self.action_dim - 1)

                # Stage A (safe default): measure teacher CE on the inference path,
                # but DO NOT update the CVAE-DQN teacher from dataset mini-batches.
                # RL/replay remains the owner of prior_net, recognition_net, and Q-head
                # updates.  This prevents Sentinel-style supervised DKD from breaking
                # the RL dynamics.  Set dkd_dataset_update_teacher=true only for an
                # ablation after the global student is known to be stable.
                update_teacher_from_dataset = bool(
                    getattr(cfg_training, "dkd_dataset_update_teacher", False)
                )
                self.prior_net.train(update_teacher_from_dataset)
                self.value_net_main.train(update_teacher_from_dataset)
                if update_teacher_from_dataset:
                    self.optimizer_prior.zero_grad()
                    self.optimizer_q_rl.zero_grad()
                    mu_p, _ = self.prior_net(batch_features)
                    teacher_logits = self.value_net_main(mu_p, batch_features)
                    teacher_task_loss = self._masked_class_balanced_ce(
                        teacher_logits,
                        batch_labels,
                        class_weights,
                        present_classes,
                        label_smoothing=label_smoothing,
                    )
                    teacher_objective = (
                        float(getattr(cfg_training, "dkd_teacher_task_weight", 0.0))
                        * teacher_task_loss
                    )
                    if teacher_objective.requires_grad:
                        teacher_objective.backward()
                        if present_classes is not None:
                            self._protect_absent_class_rows(present_classes, target="teacher")
                        torch.nn.utils.clip_grad_norm_(
                            list(self.prior_net.parameters()) + list(self.value_net_main.parameters()),
                            max_norm=float(getattr(cfg_training, "dkd_grad_clip_norm", 1.0)),
                        )
                        self.optimizer_prior.step()
                        self.optimizer_q_rl.step()
                else:
                    self.prior_net.eval()
                    self.value_net_main.eval()
                    with torch.no_grad():
                        mu_p, _ = self.prior_net(batch_features)
                        teacher_logits = self.value_net_main(mu_p, batch_features)
                        teacher_task_loss = self._masked_class_balanced_ce(
                            teacher_logits,
                            batch_labels,
                            class_weights,
                            present_classes,
                            label_smoothing=label_smoothing,
                        )

                # Stage B/C: student update and optional gated global-student -> teacher transfer.
                (
                    student_task_loss,
                    t2s_loss,
                    s2t_loss,
                    align_loss,
                    stats,
                    align_score,
                    temperature,
                    t2s_enabled,
                    s2t_enabled,
                ) = self._run_dkd_teacher_student_updates(
                    states_s=batch_features,
                    labels=batch_labels,
                    class_weights=class_weights,
                    present_classes=present_classes,
                    round_num=int(round_num),
                    label_smoothing=label_smoothing,
                    allow_student_to_teacher=bool(
                        getattr(cfg_training, "dkd_update_teacher_from_student", False)
                    ),
                )
                self._update_dkd_lambdas(
                    agreement=stats.get("agreement", 0.0),
                    align_score=align_score,
                    round_num=int(round_num),
                )

                totals["teacher_task"] += float(teacher_task_loss.detach().item())
                totals["student_task"] += float(student_task_loss.detach().item())
                totals["t2s"] += float(t2s_loss.detach().item())
                totals["s2t"] += float(s2t_loss.detach().item())
                totals["align"] += float(align_loss.detach().item())
                totals["agreement"] += float(stats.get("agreement", 0.0))
                totals["correct_agreement"] += float(stats.get("correct_agreement", 0.0))
                totals["teacher_acc"] += float(stats.get("teacher_batch_accuracy", 0.0))
                totals["student_acc"] += float(stats.get("student_batch_accuracy", 0.0))
                totals["teacher_conf"] += float(stats.get("teacher_confidence", 0.0))
                totals["student_conf"] += float(stats.get("student_confidence", 0.0))
                totals["t2s_enabled"] += float(bool(t2s_enabled))
                totals["s2t_enabled"] += float(bool(s2t_enabled))
                train_steps += 1

        denom = max(train_steps, 1)
        metrics = {
            "dkd_dataset_train_steps": float(train_steps),
            "avg_dkd_teacher_task_loss": totals["teacher_task"] / denom,
            "avg_dkd_student_task_loss": totals["student_task"] / denom,
            "avg_dkd_t2s_loss": totals["t2s"] / denom,
            "avg_dkd_s2t_loss": totals["s2t"] / denom,
            "avg_dkd_align_loss": totals["align"] / denom,
            "dkd_agreement": totals["agreement"] / denom,
            "dkd_correct_agreement": totals["correct_agreement"] / denom,
            "dkd_teacher_batch_accuracy": totals["teacher_acc"] / denom,
            "dkd_student_batch_accuracy": totals["student_acc"] / denom,
            "dkd_teacher_confidence": totals["teacher_conf"] / denom,
            "dkd_student_confidence": totals["student_conf"] / denom,
            "dkd_t2s_enabled_rate": totals["t2s_enabled"] / denom,
            "dkd_s2t_enabled_rate": totals["s2t_enabled"] / denom,
            "dkd_lambda_kd": float(self.dkd_lambda_kd),
            "dkd_lambda_align": float(self.dkd_lambda_align),
            "dkd_temperature": float(temperature if train_steps else 1.0),
            "dkd_dataset_updates_teacher": float(
                bool(getattr(cfg_training, "dkd_dataset_update_teacher", False))
            ),
            "dkd_update_teacher_from_student": float(
                bool(getattr(cfg_training, "dkd_update_teacher_from_student", False))
            ),
        }
        self.last_dkd_teacher_task_loss = metrics["avg_dkd_teacher_task_loss"]
        self.last_dkd_student_task_loss = metrics["avg_dkd_student_task_loss"]
        self.last_dkd_t2s_loss = metrics["avg_dkd_t2s_loss"]
        self.last_dkd_s2t_loss = metrics["avg_dkd_s2t_loss"]
        self.last_dkd_kd_loss = metrics["avg_dkd_t2s_loss"] + metrics["avg_dkd_s2t_loss"]
        self.last_dkd_align_loss = metrics["avg_dkd_align_loss"]
        self.last_dkd_task_loss = 0.5 * (
            metrics["avg_dkd_teacher_task_loss"] + metrics["avg_dkd_student_task_loss"]
        )
        active_logger.info(
            "DKD-FedOS dataset training | steps=%d teacher_ce=%.4f student_ce=%.4f t2s=%.4f s2t=%.4f align=%.4f",
            train_steps,
            metrics["avg_dkd_teacher_task_loss"],
            metrics["avg_dkd_student_task_loss"],
            metrics["avg_dkd_t2s_loss"],
            metrics["avg_dkd_s2t_loss"],
            metrics["avg_dkd_align_loss"],
        )
        return metrics

    def _teacher_distillation_features(self, latent: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.normalize(latent, dim=1), F.normalize(q_values, dim=1)], dim=1)

    def _update_dkd_lambdas(self, *, agreement: float, align_score: float, round_num: int) -> None:
        alpha = max(0.7, 0.9 - 0.03 * float(max(round_num, 0)))
        self.dkd_lambda_kd = (alpha * self.dkd_lambda_kd) + ((1.0 - alpha) * (1.0 - float(agreement)))
        self.dkd_lambda_align = (alpha * self.dkd_lambda_align) + (
            (1.0 - alpha) * (1.0 - float(align_score))
        )
        kd_min = float(getattr(self.train_cfg, "dkd_lambda_kd_min", 0.03))
        kd_max = min(
            float(getattr(self.train_cfg, "dkd_lambda_kd_max", 0.35)),
            float(getattr(self.train_cfg, "dkd_lambda_kd_max_base", 0.18))
            + float(getattr(self.train_cfg, "dkd_lambda_kd_max_growth", 0.02)) * float(max(round_num, 0)),
        )
        align_min = float(getattr(self.train_cfg, "dkd_lambda_align_min", 0.01))
        align_max = min(
            float(getattr(self.train_cfg, "dkd_lambda_align_max", 0.12)),
            float(getattr(self.train_cfg, "dkd_lambda_align_max_base", 0.06))
            + float(getattr(self.train_cfg, "dkd_lambda_align_max_growth", 0.01)) * float(max(round_num, 0)),
        )
        self.dkd_lambda_kd = float(np.clip(self.dkd_lambda_kd, kd_min, max(kd_min, kd_max)))
        self.dkd_lambda_align = float(
            np.clip(self.dkd_lambda_align, align_min, max(align_min, align_max))
        )

    def _protect_absent_class_rows(self, present_classes: torch.Tensor, *, target: str = "teacher") -> None:
        """Zero local-label gradients for absent output rows.

        This is used only for local evidence losses.  Global student->teacher KD
        deliberately does not call this function, so absent classes can be
        recovered from the global student.
        """
        if not bool(getattr(self.train_cfg, "dkd_protect_absent_classes", True)):
            return
        mask = present_classes.to(self.device).bool().view(-1)
        if mask.numel() < self.action_dim:
            mask = F.pad(mask, (0, self.action_dim - mask.numel()), value=False)
        absent = ~mask[: self.action_dim]
        if not bool(absent.any().item()):
            return
        if target in {"teacher", "both"}:
            # Dueling Q-network stores class-specific rows in advantage_fc2.
            layer = getattr(self.value_net_main, "advantage_fc2", None)
            if layer is not None:
                if layer.weight.grad is not None:
                    layer.weight.grad[absent] = 0.0
                if layer.bias is not None and layer.bias.grad is not None:
                    layer.bias.grad[absent] = 0.0
        if target in {"student", "both"}:
            student_head = getattr(self.student_model, "head", None)
            if student_head is not None:
                if student_head.weight.grad is not None:
                    student_head.weight.grad[absent] = 0.0
                if student_head.bias is not None and student_head.bias.grad is not None:
                    student_head.bias.grad[absent] = 0.0

    def _build_prototype_features(
        self,
        latent: torch.Tensor,
        q_values: torch.Tensor,
        *,
        prototype_feature: str,
    ) -> torch.Tensor:
        feature_name = str(prototype_feature).lower()
        if feature_name in {"latent", "prior", "prior_mu", "mu"}:
            return latent
        if feature_name in {"q", "q_values", "logits"}:
            return q_values
        if feature_name in {"latent_q", "mu_q", "latent+q", "prior_q"}:
            return torch.cat(
                [
                    F.normalize(latent, dim=1),
                    F.normalize(q_values, dim=1),
                ],
                dim=1,
            )
        self.logger.warning("Unknown prototype_feature=%s; falling back to latent_q", prototype_feature)
        return torch.cat([F.normalize(latent, dim=1), F.normalize(q_values, dim=1)], dim=1)

    def _prototype_alignment_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        global_prototypes: torch.Tensor,
        global_prototype_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if features.numel() == 0 or global_prototypes.numel() == 0:
            return None

        labels = labels.view(-1).long().to(features.device)
        targets = global_prototypes.to(features.device, dtype=features.dtype)
        if targets.shape[1] != features.shape[1]:
            target_dim = min(targets.shape[1], features.shape[1])
            targets = targets[:, :target_dim]
            features = features[:, :target_dim]

        if global_prototype_mask is not None:
            mask = global_prototype_mask.to(features.device).bool().view(-1)
        else:
            mask = torch.ones(targets.shape[0], device=features.device, dtype=torch.bool)

        losses: list[torch.Tensor] = []
        for class_idx in labels.unique(sorted=True):
            class_int = int(class_idx.item())
            if class_int < 0 or class_int >= targets.shape[0] or not bool(mask[class_int].item()):
                continue
            class_mask = labels == class_int
            if not bool(class_mask.any().item()):
                continue
            local_proto = features[class_mask].mean(dim=0)
            losses.append(F.mse_loss(local_proto, targets[class_int], reduction="mean"))

        if not losses:
            return None
        return torch.stack(losses).mean()

    def update_target_network(self, tau: float):
        """Soft update the target network (Eq. 10)."""
        self.logger.debug("Performing soft target update with tau=%s", tau)
        soft_update_target_network(self.value_net_main, self.value_net_target, tau)

    def hard_update_target_network(self):
        """Hard update (copy) the target network."""
        self.logger.debug("Performing hard target update")
        self.value_net_target.load_state_dict(self.value_net_main.state_dict())

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

    def get_student_parameters(self) -> list[np.ndarray]:
        """Return compact DKD-FedOS student parameters only."""
        return [val.detach().cpu().numpy() for val in self.student_model.state_dict().values()]

    def set_student_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load compact DKD-FedOS student parameters only."""
        keys = list(self.student_model.state_dict().keys())
        if len(parameters) != len(keys):
            raise ValueError(
                f"Student parameter mismatch: expected {len(keys)}, received {len(parameters)}"
            )
        self.student_model.load_state_dict(
            OrderedDict(
                zip(keys, [torch.tensor(p, device=self.device) for p in parameters], strict=True)
            )
        )

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
            active_logger.warning(
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

        active_logger.info(f"--- Generator Training Start (Samples: {num_correct}) ---")

        last_epoch_loss = 0.0
        last_epoch_prox_loss = 0.0

        for round_idx in range(1, gen_rounds + 1):
            for epoch in range(1, gen_epochs + 1):
                total_loss = 0.0
                total_prox_loss = 0.0
                batch_count = 0

                for states_s, true_actions in train_loader:
                    states_s = states_s.to(self.device)
                    true_actions = true_actions.to(self.device)

                    with torch.no_grad():
                        mu_q, log_var_q = self.recognition_net(states_s, true_actions)
                        latent_z = reparameterization_trick(mu_q, log_var_q)

                    recon = self.generation_net(latent_z, true_actions)
                    mse_loss = loss_fn(recon, states_s)
                    loss = mse_loss
                    prox_loss = torch.zeros((), device=self.device)
                    if proximal_mu > 0.0:
                        prox_loss = 0.5 * proximal_mu * self._proximal_penalty("generation_net")
                        loss = loss + prox_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += mse_loss.item()
                    total_prox_loss += prox_loss.item()
                    batch_count += 1

                last_epoch_loss = total_loss / max(1, batch_count)
                last_epoch_prox_loss = total_prox_loss / max(1, batch_count)

                # --- LOGGING EVERY 5 EPOCHS ---
                if epoch % 5 == 0 or epoch == 1:
                    active_logger.info(
                        f"   > [Round {round_idx}] Epoch {epoch:02d}/{gen_epochs} | "
                        f"Loss (MSE): {last_epoch_loss:.6f} | "
                        f"Prox: {last_epoch_prox_loss:.6f} | "
                        f"Batch Count: {batch_count}"
                    )

        self.generation_net.eval()
        return {
            "generator_loss": float(last_epoch_loss),
            "generator_prox_loss": float(last_epoch_prox_loss),
            "generator_samples": float(num_correct),
            "generator_correct_frac": num_correct / max(1, total_samples),
        }
