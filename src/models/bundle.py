"""FedTROS-PR Client Model Bundle managing Private Variational Classifier Teacher (VCT) and Federated Student."""

import logging
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from src.models.models import ModelFactory
from src.models.student import StudentIDSModel
from src.models.variational_teacher import (
    VariationalClassifierTeacher,
    kl_standard_normal,
)
from src.training.class_balance import class_balanced_cross_entropy
from src.training.distillation import (
    directional_kd_loss,
    kd_temperature,
    mse_cosine_alignment,
    disagreement_gated_teacher_to_student_kd,
    prediction_stats,
)

logger = logging.getLogger("FedTROSModelBundle")


class FedTROSModelBundle:
    """Client model bundle managing the client-private Variational Classifier Teacher (VCT)

    and the compact globally-shared Student IDS Model.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        train_cfg: DictConfig,
        device: torch.device,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger("FedTROSModelBundle")
        self.train_cfg = train_cfg
        self.device = device
        self.num_classes = int(model_factory.num_classes)
        self.feature_dim = int(model_factory.feature_dim)
        self.teacher_latent_dim = int(getattr(train_cfg, "teacher_latent_dim", getattr(model_factory, "latent_dim", 64)))
        self.latent_dim = self.teacher_latent_dim

        # 1. Optional private Variational Classifier Teacher (VCT).
        # Architecture-matched FedAvg/FedProx baselines and A1 no-teacher do not
        # instantiate private-teacher/aligner state, so compute/memory accounting is fair.
        self.teacher_enabled = bool(getattr(train_cfg, "teacher_enabled", True))
        self.teacher_stochastic_training = bool(getattr(train_cfg, "teacher_stochastic_training", True))
        self.kd_enabled = bool(getattr(train_cfg, "kd_enabled", True))
        self.kd_gating_enabled = bool(getattr(train_cfg, "kd_gating_enabled", True))
        self.alignment_enabled = bool(getattr(train_cfg, "alignment_enabled", True))
        teacher_hidden = list(getattr(train_cfg, "teacher_hidden_dims", [512, 256]))
        teacher_dropout = float(getattr(train_cfg, "teacher_dropout", 0.0))
        self.teacher: VariationalClassifierTeacher | None = None
        if self.teacher_enabled:
            self.teacher = VariationalClassifierTeacher(
                input_dim=self.feature_dim,
                num_classes=self.num_classes,
                latent_dim=self.teacher_latent_dim,
                hidden_dims=teacher_hidden,
                transformer_cfg=getattr(model_factory, "transformer_cfg", None),
                dropout=teacher_dropout,
            ).to(device)

        # 2. Federated Shared Student (the only communicated model)
        student_hidden = list(getattr(train_cfg, "fedtros_student_hidden_dims", [512, 256, 128]))
        student_osr_enabled = bool(getattr(train_cfg, "student_osr_enabled", True))
        student_open_set_enabled = bool(getattr(train_cfg, "student_open_set_enabled", False))
        student_osr_hidden = list(getattr(train_cfg, "student_osr_hidden_dims", [128, 64]))
        student_osr_decoder_hidden = list(getattr(train_cfg, "student_osr_decoder_hidden_dims", [64, 128]))

        self.student_model = StudentIDSModel(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dims=student_hidden,
            activation=str(getattr(train_cfg, "fedtros_student_activation", "gelu")),
            dropout=float(getattr(train_cfg, "fedtros_student_dropout", 0.05)),
            norm=str(getattr(train_cfg, "fedtros_student_norm", "layernorm")),
            osr_enabled=student_osr_enabled,
            osr_latent_dim=int(getattr(train_cfg, "student_osr_latent_dim", 8)),
            osr_hidden_dims=student_osr_hidden,
            osr_decoder_hidden_dims=student_osr_decoder_hidden,
            osr_dropout=float(getattr(train_cfg, "student_osr_dropout", 0.05)),
            osr_norm=str(getattr(train_cfg, "student_osr_norm", "layernorm")),
            osr_activation=str(getattr(train_cfg, "student_osr_activation", "gelu")),
            osr_detach_features=bool(getattr(train_cfg, "student_osr_detach_features", True)),
            open_set_enabled=student_open_set_enabled,
        ).to(device)

        # 3. Optional frozen global-student anchor. It is omitted for architecture-matched
        # baselines and the A2 no-anchor variant.
        self.anchor_enabled = float(getattr(train_cfg, "fedtros_global_anchor_weight", 0.0)) > 0.0
        self.student_anchor_model: StudentIDSModel | None = None
        if self.anchor_enabled:
            self.student_anchor_model = StudentIDSModel(
                input_dim=self.feature_dim,
                num_classes=self.num_classes,
                hidden_dims=student_hidden,
                activation=str(getattr(train_cfg, "fedtros_student_activation", "gelu")),
                dropout=float(getattr(train_cfg, "fedtros_student_dropout", 0.05)),
                norm=str(getattr(train_cfg, "fedtros_student_norm", "layernorm")),
                osr_enabled=student_osr_enabled,
                osr_latent_dim=int(getattr(train_cfg, "student_osr_latent_dim", 8)),
                osr_hidden_dims=student_osr_hidden,
                osr_decoder_hidden_dims=student_osr_decoder_hidden,
                osr_dropout=float(getattr(train_cfg, "student_osr_dropout", 0.05)),
                osr_norm=str(getattr(train_cfg, "student_osr_norm", "layernorm")),
                osr_activation=str(getattr(train_cfg, "student_osr_activation", "gelu")),
                osr_detach_features=bool(getattr(train_cfg, "student_osr_detach_features", True)),
                open_set_enabled=student_open_set_enabled,
            ).to(device)
            self.student_anchor_model.load_state_dict(self.student_model.state_dict())
            self.student_anchor_model.eval()
            for param in self.student_anchor_model.parameters():
                param.requires_grad_(False)

        # 4. Teacher-to-student feature aligner is private and exists only when a teacher
        # is instantiated.
        self.teacher_to_student_aligner: nn.Linear | None = None
        if self.teacher is not None:
            self.teacher_to_student_aligner = nn.Linear(
                self.teacher.latent_dim, int(self.student_model.feature_dim)
            ).to(device)

        # 5. Hyperparameters
        self.beta_kl = float(getattr(train_cfg, "teacher_beta_kl", getattr(train_cfg, "beta_kl", 0.01)))
        self.lambda_kd = float(getattr(train_cfg, "lambda_kd", getattr(train_cfg, "lambda_kd_init", 0.20)))
        self.lambda_align = float(getattr(train_cfg, "lambda_align", getattr(train_cfg, "lambda_align_init", 0.08)))

        # 6. Optimizers. Baselines carry no unused VCT/aligner optimizer state.
        teacher_lr = float(getattr(train_cfg, "teacher_lr", getattr(train_cfg, "lr_prior", 1e-3)))
        teacher_wd = float(getattr(train_cfg, "teacher_weight_decay", 1e-4))
        self.optimizer_teacher: optim.Optimizer | None = None
        if self.teacher is not None:
            self.optimizer_teacher = optim.AdamW(self.teacher.parameters(), lr=teacher_lr, weight_decay=teacher_wd)

        student_lr = float(getattr(train_cfg, "student_lr", 1e-3))
        student_wd = float(getattr(train_cfg, "student_weight_decay", 1e-4))
        student_params = list(self.student_model.classifier_parameters())
        if self.teacher_to_student_aligner is not None:
            student_params += list(self.teacher_to_student_aligner.parameters())
        self.optimizer_student = optim.AdamW(student_params, lr=student_lr, weight_decay=student_wd)

        osr_params = list(self.student_model.osr_parameters())
        if osr_params:
            osr_lr = float(getattr(train_cfg, "student_osr_lr", student_lr))
            osr_wd = float(getattr(train_cfg, "student_osr_weight_decay", 1e-4))
            self.optimizer_student_osr = optim.AdamW(osr_params, lr=osr_lr, weight_decay=osr_wd)
        else:
            self.optimizer_student_osr = None

        open_set_params = list(self.student_model.open_set_parameters())
        if open_set_params:
            open_lr = float(getattr(train_cfg, "student_open_set_lr", student_lr))
            open_wd = float(getattr(train_cfg, "student_open_set_weight_decay", 1e-4))
            self.optimizer_student_open_set = optim.AdamW(open_set_params, lr=open_lr, weight_decay=open_wd)
        else:
            self.optimizer_student_open_set = None

        # Energy is an evaluation baseline by default. Retain this optimizer only when an
        # explicit train-time energy auxiliary is enabled in a non-canonical ablation.
        self.optimizer_student_energy: optim.Optimizer | None = None
        if bool(getattr(train_cfg, "energy_train_margin_enabled", False)):
            energy_head_params = list(self.student_model.head.parameters())
            energy_lr = float(getattr(train_cfg, "student_energy_lr", max(student_lr * 0.2, 1e-5)))
            energy_wd = float(getattr(train_cfg, "student_energy_weight_decay", 1e-4))
            self.optimizer_student_energy = optim.AdamW(energy_head_params, lr=energy_lr, weight_decay=energy_wd)

        # Proximal reference for FedProx-Student baseline
        self._proximal_reference: list[torch.Tensor] | None = None
        self._capture_proximal_reference()

        # Tracking state
        self.last_teacher_task_loss = 0.0
        self.last_teacher_kl_loss = 0.0
        self.last_student_task_loss = 0.0
        self.last_kd_loss = 0.0
        self.last_align_loss = 0.0
        self.last_temperature = 1.0
        self.last_agreement = 0.0
        self.last_confidence = 0.0
        self.last_align_score = 0.0

        self.logger.info(
            "Client model bundle initialized | teacher_enabled=%s | teacher_hidden=%s | student_hidden=%s | latent_dim=%d | "
            "student_osr_enabled=%s | anchor_enabled=%s | beta_kl=%.4f | lambda_kd=%.4f | lambda_align=%.4f",
            self.teacher_enabled,
            teacher_hidden,
            student_hidden,
            self.teacher_latent_dim,
            student_osr_enabled,
            self.anchor_enabled,
            self.beta_kl,
            self.lambda_kd,
            self.lambda_align,
        )

    def to(self, device: torch.device | str) -> "FedTROSModelBundle":
        """Move bundle models and optimizer states to target device."""
        target_device = torch.device(device)
        if target_device == self.device:
            return self

        if self.teacher is not None:
            self.teacher.to(target_device)
        self.student_model.to(target_device)
        if self.student_anchor_model is not None:
            self.student_anchor_model.to(target_device)
        if self.teacher_to_student_aligner is not None:
            self.teacher_to_student_aligner.to(target_device)

        if self.optimizer_teacher is not None:
            self._move_optimizer_state(self.optimizer_teacher, target_device)
        self._move_optimizer_state(self.optimizer_student, target_device)
        if self.optimizer_student_osr is not None:
            self._move_optimizer_state(self.optimizer_student_osr, target_device)
        if self.optimizer_student_open_set is not None:
            self._move_optimizer_state(self.optimizer_student_open_set, target_device)
        if self.optimizer_student_energy is not None:
            self._move_optimizer_state(self.optimizer_student_energy, target_device)

        self.device = target_device
        return self

    @staticmethod
    def _move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)
                elif isinstance(value, list):
                    state[key] = [item.to(device) if torch.is_tensor(item) else item for item in value]
                elif isinstance(value, tuple):
                    state[key] = tuple(item.to(device) if torch.is_tensor(item) else item for item in value)

    def _capture_proximal_reference(self) -> None:
        """Store current student parameters for FedProx proximal regularization."""
        self._proximal_reference = [
            param.detach().clone() for param in self.student_model.parameters()
        ]

    def _proximal_penalty(self) -> torch.Tensor:
        """Compute proximal penalty ||w - w_global||^2 for FedProx-Student."""
        if not self._proximal_reference:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        for current_param, ref_param in zip(self.student_model.parameters(), self._proximal_reference, strict=True):
            penalty = penalty + (current_param - ref_param.to(current_param.device)).pow(2).sum()
        return penalty

    def reset_federated_optimizers(self) -> None:
        """Reset local optimizers between communication rounds for clean baselines."""
        student_lr = float(getattr(self.train_cfg, "student_lr", 1e-3))
        student_wd = float(getattr(self.train_cfg, "student_weight_decay", 1e-4))
        student_params = list(self.student_model.classifier_parameters())
        if self.teacher_to_student_aligner is not None:
            student_params += list(self.teacher_to_student_aligner.parameters())
        self.optimizer_student = optim.AdamW(student_params, lr=student_lr, weight_decay=student_wd)

    def train_teacher_step(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        class_weights: torch.Tensor | None = None,
        beta_kl: float | None = None,
        label_smoothing: float = 0.02,
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        """Execute one supervised VIB training step for the private teacher.

        Objective: L_T = L_CBCE^T + beta_T * D_KL[q_phi(z|x) || N(0, I)]
        """
        if self.teacher is None or self.optimizer_teacher is None:
            raise RuntimeError("VCT teacher step requested while teacher_enabled=false")
        self.teacher.train()
        self.optimizer_teacher.zero_grad()

        logits, mu, logvar, _ = self.teacher(features, sample=self.teacher_stochastic_training)
        loss_cls = class_balanced_cross_entropy(
            logits, labels, class_weights, label_smoothing=label_smoothing
        )
        loss_kl = kl_standard_normal(mu, logvar)
        b_kl = self.beta_kl if beta_kl is None else float(beta_kl)
        total_loss = loss_cls + (b_kl * loss_kl)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm=max_grad_norm)
        self.optimizer_teacher.step()

        self.last_teacher_task_loss = float(loss_cls.detach().item())
        self.last_teacher_kl_loss = float(loss_kl.detach().item())
        return {
            "teacher_loss": float(total_loss.detach().item()),
            "teacher_cls_loss": self.last_teacher_task_loss,
            "teacher_kl_loss": self.last_teacher_kl_loss,
        }

    def train_student_step(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        class_weights: torch.Tensor | None = None,
        present_classes: torch.Tensor | None = None,
        kappa_i: float | None = None,
        round_num: int = 0,
        label_smoothing: float = 0.02,
        proximal_mu: float = 0.0,
        t2s_start_round: int = 1,
        align_start_round: int = 1,
    ) -> dict[str, float]:
        """Execute one training step for the federated student with VCT guidance.

        Objective: L_S = L_CBCE + lambda_anchor * L_anchor + lambda_KD * L_T->S + lambda_align * L_align
        """
        self.student_model.train()
        if self.teacher_to_student_aligner is not None:
            self.teacher_to_student_aligner.train()
        if self.teacher is not None:
            self.teacher.eval()
        self.optimizer_student.zero_grad()

        # 1. Student Forward Pass
        student_features, student_logits = self.student_model(features)

        # 2. Deterministic VCT snapshot.  It is skipped entirely for the no-teacher ablation.
        teacher_logits: torch.Tensor | None = None
        teacher_mu: torch.Tensor | None = None
        if self.teacher is not None and self.teacher_enabled and (self.kd_enabled or self.alignment_enabled):
            with torch.no_grad():
                teacher_logits, teacher_mu, _ = self.teacher.distill_forward(features)
                teacher_logits = teacher_logits.detach()
                teacher_mu = teacher_mu.detach()

        # 3. Student Task Loss (Class-Balanced CE)
        student_task_loss = class_balanced_cross_entropy(
            student_logits, labels, class_weights, label_smoothing=label_smoothing
        )

        # 4. Coverage-Adaptive Global Anchor Loss
        anchor_loss = torch.zeros((), device=self.device)
        anchor_weight = 0.0
        class_coverage = 1.0
        
        canonical = bool(getattr(self.train_cfg, "canonical", False))
        if canonical and kappa_i is not None:
            class_coverage = float(kappa_i)
        elif present_classes is not None:
            present_mask = present_classes.to(self.device).bool().view(-1)[: self.num_classes]
            class_coverage = float(present_mask.float().mean().detach().item()) if present_mask.numel() else 1.0

        base_anchor_weight = float(
            getattr(self.train_cfg, "fedtros_global_anchor_weight", 2.0)
        )
        if base_anchor_weight > 0.0 and self.student_anchor_model is not None:
            anchor_power = float(
                getattr(self.train_cfg, "fedtros_global_anchor_coverage_power", 1.0)
            )
            anchor_min = float(
                getattr(self.train_cfg, "fedtros_global_anchor_min_weight", 0.0)
            )
            coverage_gap = max(0.0, min(1.0, 1.0 - float(class_coverage)))
            coverage_factor = max(anchor_min, coverage_gap ** max(anchor_power, 0.0))
            anchor_weight = base_anchor_weight * coverage_factor
            self.student_anchor_model.eval()
            with torch.no_grad():
                _anchor_features, anchor_logits = self.student_anchor_model(features)
            anchor_loss = directional_kd_loss(
                anchor_logits,
                student_logits,
                temperature=float(getattr(self.train_cfg, "kd_base_temperature", 2.0)),
                mean_class_weight=1.0,
            )

        # 5. Disagreement-Gated T -> S KD
        temperature = kd_temperature(
            int(round_num),
            base=float(getattr(self.train_cfg, "kd_base_temperature", 3.0)),
            minimum=float(getattr(self.train_cfg, "kd_min_temperature", 1.0)),
            decay=float(getattr(self.train_cfg, "kd_decay", 0.95)),
        )
        zero = torch.zeros((), device=self.device)
        stats = {
            "agreement": 0.0, "joint_confidence": 0.0, "teacher_batch_accuracy": 0.0,
            "student_batch_accuracy": float((student_logits.argmax(dim=1) == labels).float().mean().detach().item()),
            "correct_agreement": 0.0,
        }
        t2s_component = zero
        t2s_enabled = bool(self.teacher_enabled and self.kd_enabled and teacher_logits is not None and int(round_num) >= int(t2s_start_round))
        if t2s_enabled:
            if self.kd_gating_enabled:
                t2s_component, stats = disagreement_gated_teacher_to_student_kd(
                    teacher_logits, student_logits, temperature=temperature, mean_class_weight=1.0, labels=labels
                )
            else:
                stats = prediction_stats(teacher_logits, student_logits, labels=labels)
                t2s_component = directional_kd_loss(
                    teacher_logits, student_logits, temperature=temperature, mean_class_weight=1.0
                )

        # 6. Teacher-Student Feature Alignment (mu_T -> A_phi -> h_S)
        align_loss = zero
        align_score = 0.0
        align_enabled = bool(self.teacher_enabled and self.alignment_enabled and teacher_mu is not None and int(round_num) >= int(align_start_round))
        if align_enabled:
            if self.teacher_to_student_aligner is None:
                raise RuntimeError("Feature alignment enabled without a VCT aligner")
            projected_teacher = self.teacher_to_student_aligner(teacher_mu)
            align_loss, align_score = mse_cosine_alignment(
                projected_teacher, student_features,
                cosine_weight=float(getattr(self.train_cfg, "align_cos_weight", 0.5)),
                mse_weight=float(getattr(self.train_cfg, "align_mse_weight", 1.0)),
            )

        # 7. Total Student Objective
        student_objective = (
            float(getattr(self.train_cfg, "student_task_weight", 1.0)) * student_task_loss
        )
        if anchor_weight > 0.0:
            student_objective = student_objective + (anchor_weight * anchor_loss)
        if t2s_enabled:
            student_objective = student_objective + (self.lambda_kd * t2s_component)
        if align_enabled:
            student_objective = student_objective + (self.lambda_align * align_loss)
        if proximal_mu > 0.0:
            student_objective = student_objective + (0.5 * proximal_mu * self._proximal_penalty())

        student_objective.backward()
        clip_params = list(self.student_model.classifier_parameters())
        if self.teacher_to_student_aligner is not None:
            clip_params += list(self.teacher_to_student_aligner.parameters())
        torch.nn.utils.clip_grad_norm_(
            clip_params, max_norm=float(getattr(self.train_cfg, "grad_clip_norm", 1.0))
        )
        self.optimizer_student.step()

        self._update_lambdas(
            agreement=stats.get("agreement", 0.0),
            align_score=align_score,
            round_num=round_num,
        )

        self.last_student_task_loss = float(student_task_loss.detach().item())
        self.last_kd_loss = float(t2s_component.detach().item()) if t2s_enabled else 0.0
        self.last_align_loss = float(align_loss.detach().item()) if align_enabled else 0.0
        self.last_temperature = float(temperature)
        self.last_agreement = float(stats.get("agreement", 0.0))
        self.last_confidence = float(stats.get("joint_confidence", 0.0))
        self.last_align_score = float(align_score)

        return {
            "student_task_loss": self.last_student_task_loss,
            "student_anchor_loss": float(anchor_loss.detach().item()),
            "student_anchor_weight": float(anchor_weight),
            "student_kd_loss": self.last_kd_loss,
            "student_align_loss": self.last_align_loss,
            "student_total_loss": float(student_objective.detach().item()),
            "agreement": self.last_agreement,
            "confidence": self.last_confidence,
            "align_score": self.last_align_score,
            "teacher_acc": float(stats.get("teacher_batch_accuracy", 0.0)),
            "student_acc": float(stats.get("student_batch_accuracy", 0.0)),
            "correct_agreement": float(stats.get("correct_agreement", 0.0)),
            "lambda_kd": float(self.lambda_kd),
            "lambda_align": float(self.lambda_align),
            "temperature": float(temperature),
        }

    def _update_lambdas(self, *, agreement: float, align_score: float, round_num: int) -> None:
        """Optionally adapt transfer weights in a non-canonical ablation.

        The canonical FedTROS-PR target uses fixed, configuration-frozen ``lambda_kd``
        and ``lambda_align`` values.  The old implicit agreement-driven weight update is
        disabled unless ``training.adaptive_transfer_weights=true`` is explicitly set.
        """
        if not bool(getattr(self.train_cfg, "adaptive_transfer_weights", False)):
            return
        alpha = max(0.7, 0.9 - 0.03 * float(max(round_num, 0)))
        self.lambda_kd = (alpha * self.lambda_kd) + ((1.0 - alpha) * (1.0 - float(agreement)))
        self.lambda_align = (alpha * self.lambda_align) + ((1.0 - alpha) * (1.0 - float(align_score)))
        kd_min = float(getattr(self.train_cfg, "lambda_kd_min", 0.03))
        kd_max = float(getattr(self.train_cfg, "lambda_kd_max", 0.35))
        align_min = float(getattr(self.train_cfg, "lambda_align_min", 0.01))
        align_max = float(getattr(self.train_cfg, "lambda_align_max", 0.12))
        self.lambda_kd = float(np.clip(self.lambda_kd, kd_min, max(kd_min, kd_max)))
        self.lambda_align = float(np.clip(self.lambda_align, align_min, max(align_min, align_max)))

    def train_fedtros_dataset(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        cfg_training: DictConfig,
        *,
        round_num: int,
        class_weights: torch.Tensor | None = None,
        present_classes: torch.Tensor | None = None,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
    ) -> dict[str, float]:
        """Execute full local dataset training round for FedTROS-PR:

        1. Train private VCT teacher over local epochs
        2. Train federated student guided by VCT and anchor
        """
        active_logger = logger or self.logger
        dev = device or self.device
        batch_size = int(getattr(cfg_training, "batch_size", 64))
        teacher_epochs = int(getattr(cfg_training, "teacher_epochs", getattr(cfg_training, "local_epochs", 2)))
        student_epochs = int(getattr(cfg_training, "student_epochs", 2))
        label_smoothing = float(getattr(cfg_training, "label_smoothing", 0.02))
        t2s_start = int(getattr(cfg_training, "fedtros_teacher_to_student_start_round", 1))
        align_start = int(getattr(cfg_training, "fedtros_alignment_start_round", 1))

        features = features.detach().float().to(dev)
        labels = labels.detach().long().view(-1).to(dev).clamp(0, self.num_classes - 1)
        
        counts = torch.bincount(labels, minlength=self.num_classes)[:self.num_classes]
        total_samples = int(counts.sum().item())
        if total_samples > 0:
            probs = counts[counts > 0].float() / total_samples
            unnorm_entropy = -(probs * torch.log(probs)).sum().item()
            kappa_i = float(np.exp(unnorm_entropy) / max(self.num_classes, 1))
        else:
            kappa_i = 1.0 / max(self.num_classes, 1)

        dataset = TensorDataset(features, labels)
        if len(dataset) == 0:
            return {"dataset_train_steps": 0.0}

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        if class_weights is not None:
            class_weights = class_weights.to(dev)
        if present_classes is not None:
            present_classes = present_classes.to(dev)

        # Stage 1: Private Teacher Training (VIB objective)
        teacher_start = time.perf_counter()
        teacher_loss_sum = 0.0
        teacher_cls_sum = 0.0
        teacher_kl_sum = 0.0
        teacher_steps = 0

        if self.teacher_enabled:
            for _ in range(max(1, teacher_epochs)):
                for bx, by in loader:
                    t_metrics = self.train_teacher_step(
                        bx, by, class_weights=class_weights, label_smoothing=label_smoothing
                    )
                    teacher_loss_sum += t_metrics["teacher_loss"]
                    teacher_cls_sum += t_metrics["teacher_cls_loss"]
                    teacher_kl_sum += t_metrics["teacher_kl_loss"]
                    teacher_steps += 1

        t_denom = max(1, teacher_steps)
        avg_teacher_loss = teacher_loss_sum / t_denom
        avg_teacher_cls = teacher_cls_sum / t_denom
        avg_teacher_kl = teacher_kl_sum / t_denom
        teacher_seconds = float(time.perf_counter() - teacher_start) if self.teacher_enabled else 0.0

        # Stage 2: Student Training with VCT guidance and global anchor
        student_start = time.perf_counter()
        student_loss_sum = 0.0
        student_task_sum = 0.0
        kd_sum = 0.0
        align_sum = 0.0
        anchor_sum = 0.0
        student_steps = 0
        last_s_metrics: dict[str, float] = {}

        for _ in range(max(1, student_epochs)):
            for bx, by in loader:
                s_metrics = self.train_student_step(
                    bx,
                    by,
                    class_weights=class_weights,
                    present_classes=present_classes,
                    kappa_i=kappa_i,
                    round_num=round_num,
                    label_smoothing=label_smoothing,
                    t2s_start_round=t2s_start,
                    align_start_round=align_start,
                )
                student_loss_sum += s_metrics["student_total_loss"]
                student_task_sum += s_metrics["student_task_loss"]
                kd_sum += s_metrics["student_kd_loss"]
                align_sum += s_metrics["student_align_loss"]
                anchor_sum += s_metrics["student_anchor_loss"]
                student_steps += 1
                last_s_metrics = s_metrics

        student_seconds = float(time.perf_counter() - student_start)
        s_denom = max(1, student_steps)
        metrics = {
            "teacher_enabled": float(self.teacher_enabled),
            "teacher_stochastic_training": float(self.teacher_stochastic_training),
            "teacher_epochs": float(teacher_epochs if self.teacher_enabled else 0),
            "student_epochs": float(student_epochs),
            "teacher_train_steps": float(teacher_steps),
            "student_train_steps": float(student_steps),
            "runtime/teacher_seconds": teacher_seconds,
            "runtime/student_seconds": student_seconds,
            "avg_teacher_loss": avg_teacher_loss,
            "avg_teacher_cls_loss": avg_teacher_cls,
            "avg_teacher_kl_loss": avg_teacher_kl,
            "avg_student_total_loss": student_loss_sum / s_denom,
            "avg_student_task_loss": student_task_sum / s_denom,
            "avg_student_kd_loss": kd_sum / s_denom,
            "avg_student_align_loss": align_sum / s_denom,
            "avg_student_anchor_loss": anchor_sum / s_denom,
            "student_anchor_weight": float(last_s_metrics.get("student_anchor_weight", 0.0)),
            "agreement": float(last_s_metrics.get("agreement", 0.0)),
            "confidence": float(last_s_metrics.get("confidence", 0.0)),
            "align_score": float(last_s_metrics.get("align_score", 0.0)),
            "teacher_acc": float(last_s_metrics.get("teacher_acc", 0.0)),
            "student_acc": float(last_s_metrics.get("student_acc", 0.0)),
            "correct_agreement": float(last_s_metrics.get("correct_agreement", 0.0)),
            "lambda_kd": float(self.lambda_kd),
            "lambda_align": float(self.lambda_align),
            "temperature": float(last_s_metrics.get("temperature", 1.0)),
        }

        active_logger.info(
            "FedTROS local round | teacher_loss=%.4f (cls=%.4f kl=%.4f) | "
            "student_loss=%.4f (task=%.4f kd=%.4f align=%.4f anchor=%.4f)",
            avg_teacher_loss,
            avg_teacher_cls,
            avg_teacher_kl,
            metrics["avg_student_total_loss"],
            metrics["avg_student_task_loss"],
            metrics["avg_student_kd_loss"],
            metrics["avg_student_align_loss"],
            metrics["avg_student_anchor_loss"],
        )
        return metrics


    def _make_known_boundary_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        ratio: float,
        mixup_alpha: float,
        mask_probability: float,
        noise_std: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create known-derived boundary samples from known local data."""
        if x.shape[0] < 2 or float(ratio) <= 0.0:
            return x.new_zeros((0, x.shape[1])), y.new_zeros((0,), dtype=torch.long)
        n_pseudo = max(1, int(round(float(ratio) * x.shape[0])))
        idx1 = torch.randint(0, x.shape[0], (n_pseudo,), device=x.device)
        idx2 = torch.randint(0, x.shape[0], (n_pseudo,), device=x.device)
        for _ in range(3):
            same = y[idx1] == y[idx2]
            if not bool(same.any().item()):
                break
            idx2[same] = torch.randint(0, x.shape[0], (int(same.sum().item()),), device=x.device)
        alpha = max(float(mixup_alpha), 1.0e-3)
        beta = torch.distributions.Beta(alpha, alpha)
        lam = beta.sample((n_pseudo,)).to(x.device).view(-1, 1)
        pseudo = lam * x[idx1] + (1.0 - lam) * x[idx2]
        if float(mask_probability) > 0.0:
            mask = torch.rand_like(pseudo) < float(mask_probability)
            pseudo = pseudo.masked_fill(mask, 0.0)
        if float(noise_std) > 0.0:
            pseudo = pseudo + (torch.randn_like(pseudo) * float(noise_std))
        boundary_labels = y[idx1].long().clamp(0, self.num_classes - 1)
        return pseudo, boundary_labels

    def train_student_osr_on_dataset(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        osr_cfg: DictConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> dict[str, float]:
        """Train the disentangled student OSR generator branch locally."""
        active_logger = logger or self.logger
        if not bool(getattr(self.student_model, "osr_enabled", False)):
            active_logger.info("FedTROS OSR local training skipped: student OSR branch disabled.")
            return {"osr_enabled": 0.0}
        if self.optimizer_student_osr is None:
            active_logger.warning("FedTROS OSR local training skipped: optimizer missing.")
            return {"osr_enabled": 0.0, "osr_optimizer_missing": 1.0}

        features = features.detach().float().to(self.device)
        labels = labels.detach().long().view(-1).to(self.device)
        known_mask = (labels >= 0) & (labels < self.num_classes)
        features = features[known_mask]
        labels = labels[known_mask].clamp(0, self.num_classes - 1)
        known_samples = int(labels.numel())
        if known_samples == 0:
            return {"osr_enabled": 1.0, "known_samples": 0.0, "osr_steps": 0.0}

        batch_size = int(getattr(osr_cfg, "batch_size", 256))
        local_epochs = int(getattr(osr_cfg, "local_epochs", 1))
        beta_kl = float(getattr(osr_cfg, "beta_kl", 0.02))
        nll_weight = float(getattr(osr_cfg, "latent_nll_weight", 0.10))
        recon_weight = float(getattr(osr_cfg, "recon_weight", 1.0))
        grad_clip = float(getattr(osr_cfg, "grad_clip_norm", 5.0))

        boundary_cfg = getattr(osr_cfg, "boundary_samples", None)
        boundary_enabled = bool(getattr(boundary_cfg, "enabled", True)) if boundary_cfg is not None else True
        boundary_ratio = float(getattr(boundary_cfg, "ratio", 0.5)) if boundary_cfg is not None else 0.5
        boundary_margin = float(getattr(boundary_cfg, "margin", 1.5)) if boundary_cfg is not None else 1.5
        boundary_weight = float(getattr(boundary_cfg, "margin_weight", 0.25)) if boundary_cfg is not None else 0.25
        mixup_alpha = float(getattr(boundary_cfg, "mixup_alpha", 0.4)) if boundary_cfg is not None else 0.4
        mask_probability = float(getattr(boundary_cfg, "mask_probability", 0.15)) if boundary_cfg is not None else 0.15
        noise_std = float(getattr(boundary_cfg, "noise_std", 0.05)) if boundary_cfg is not None else 0.05

        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        self.student_model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_boundary = 0.0
        steps = 0

        for _ in range(max(1, local_epochs)):
            for xb, yb in loader:
                out = self.student_model.osr_score(
                    xb, yb, nll_weight=nll_weight, detach_features=True
                )
                recon = out["recon_error"].mean()
                kl = out["latent_nll"].mean()
                loss = (recon_weight * recon) + (beta_kl * kl)

                if boundary_enabled and xb.shape[0] >= 2:
                    xp, yp = self._make_known_boundary_batch(
                        xb, yb, ratio=boundary_ratio, mixup_alpha=mixup_alpha,
                        mask_probability=mask_probability, noise_std=noise_std
                    )
                    if yp.numel() > 0:
                        p_out = self.student_model.osr_score(
                            xp, yp, nll_weight=nll_weight, detach_features=True
                        )
                        boundary_score = p_out["recon_error"]
                        known_ref = out["recon_error"].detach()
                        pair_n = min(int(boundary_score.shape[0]), int(known_ref.shape[0]))
                        if pair_n > 0:
                            gap = boundary_score[:pair_n] - known_ref[:pair_n]
                            boundary_loss = F.relu(float(boundary_margin) - gap).mean()
                        else:
                            boundary_loss = F.relu(float(boundary_margin) - (boundary_score.mean() - known_ref.mean()))
                        loss = loss + (boundary_weight * boundary_loss)
                        total_boundary += float(boundary_loss.detach().item())

                self.optimizer_student_osr.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.osr_parameters(), max_norm=grad_clip)
                self.optimizer_student_osr.step()

                total_loss += float(loss.detach().item())
                total_recon += float(recon.detach().item())
                total_kl += float(kl.detach().item())
                steps += 1

        denom = max(1, steps)
        metrics = {
            "osr_enabled": 1.0,
            "osr_steps": float(steps),
            "avg_osr_loss": total_loss / denom,
            "avg_osr_recon_loss": total_recon / denom,
            "avg_osr_kl_loss": total_kl / denom,
            "avg_osr_boundary_loss": total_boundary / denom,
        }
        active_logger.info(
            "Student OSR local training | steps=%d recon=%.6f kl=%.6f pseudo=%.6f",
            steps,
            metrics["avg_osr_recon_loss"],
            metrics["avg_osr_kl_loss"],
            metrics["avg_osr_boundary_loss"],
        )
        return metrics

    # --- PRIVATE CLIENT CHECKPOINT STATE ---

    def private_state_dict(self) -> dict[str, Any]:
        """Return client-local state needed for exact FedTROS simulation resume."""
        payload: dict[str, Any] = {"optimizer_student": self.optimizer_student.state_dict()}
        if self.teacher is not None:
            payload["teacher"] = self.teacher.state_dict()
        if self.teacher_to_student_aligner is not None:
            payload["teacher_to_student_aligner"] = self.teacher_to_student_aligner.state_dict()
        if self.optimizer_teacher is not None:
            payload["optimizer_teacher"] = self.optimizer_teacher.state_dict()
        for name in ("optimizer_student_osr", "optimizer_student_open_set", "optimizer_student_energy"):
            optimizer = getattr(self, name, None)
            if optimizer is not None:
                payload[name] = optimizer.state_dict()
        return payload

    def load_private_state_dict(self, payload: dict[str, Any], *, strict: bool = True) -> None:
        """Restore private teacher/aligner/optimizer state for the configured variant."""
        if self.teacher is not None:
            if "teacher" not in payload:
                raise ValueError("Private checkpoint is missing VCT state for a teacher-enabled run.")
            self.teacher.load_state_dict(payload["teacher"], strict=strict)
        elif "teacher" in payload:
            raise ValueError("Private checkpoint contains a VCT but current configuration disables the teacher.")
        if self.teacher_to_student_aligner is not None:
            if "teacher_to_student_aligner" not in payload:
                raise ValueError("Private checkpoint is missing teacher-to-student aligner state.")
            self.teacher_to_student_aligner.load_state_dict(payload["teacher_to_student_aligner"], strict=strict)
        if self.optimizer_teacher is not None and "optimizer_teacher" in payload:
            self.optimizer_teacher.load_state_dict(payload["optimizer_teacher"])
        if "optimizer_student" in payload:
            self.optimizer_student.load_state_dict(payload["optimizer_student"])
        for name in ("optimizer_student_osr", "optimizer_student_open_set", "optimizer_student_energy"):
            optimizer = getattr(self, name, None)
            if optimizer is not None and name in payload:
                optimizer.load_state_dict(payload[name])
        if self.optimizer_teacher is not None:
            self._move_optimizer_state(self.optimizer_teacher, self.device)
        self._move_optimizer_state(self.optimizer_student, self.device)
        for name in ("optimizer_student_osr", "optimizer_student_open_set", "optimizer_student_energy"):
            optimizer = getattr(self, name, None)
            if optimizer is not None:
                self._move_optimizer_state(optimizer, self.device)

    # --- FEDERATED PARAMETERS ---

    def get_student_parameters(self) -> list[np.ndarray]:
        """Return compact federated student model parameters only."""
        return [val.detach().cpu().numpy() for val in self.student_model.state_dict().values()]

    def set_student_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load compact federated student model parameters and update anchor snapshot."""
        keys = list(self.student_model.state_dict().keys())
        if len(parameters) != len(keys):
            raise ValueError(
                f"Student parameter mismatch: expected {len(keys)}, received {len(parameters)}"
            )
        state = OrderedDict(
            zip(keys, [torch.tensor(p, device=self.device) for p in parameters], strict=True)
        )
        self.student_model.load_state_dict(state)
        if self.student_anchor_model is not None:
            self.student_anchor_model.load_state_dict(self.student_model.state_dict())
            self.student_anchor_model.eval()
        self._capture_proximal_reference()

    def get_federated_parameters(self) -> list[np.ndarray]:
        """Alias to get_student_parameters() for standard FL compatibility."""
        return self.get_student_parameters()

    def set_federated_parameters(self, parameters: list[np.ndarray], hard_target_update: bool = True) -> None:
        """Alias to set_student_parameters() for standard FL compatibility."""
        _ = hard_target_update
        self.set_student_parameters(parameters)


# Canonical Aliases
ClientModelBundle = FedTROSModelBundle
Agent = FedTROSModelBundle
