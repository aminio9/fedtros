"""Dynamic knowledge distillation and feature alignment utilities for FedTROS-PR.

These helpers implement disagreement-gated teacher-to-student distillation,
temperature scheduling, and feature alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_temperature(
    round_num: int,
    *,
    base: float = 3.0,
    minimum: float = 1.0,
    decay: float = 0.95,
) -> float:
    """Sentinel temperature schedule.

    Tr = max(Tmin, Tbase * Tdecay ** min(r/10, 5))
    """
    exponent = min(float(max(int(round_num), 0)) / 10.0, 5.0)
    return float(max(float(minimum), float(base) * (float(decay) ** exponent)))


def compute_prediction_agreement(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> float:
    """Return fraction of samples where teacher and student agree on top-1 class."""
    with torch.no_grad():
        t_pred = teacher_logits.argmax(dim=1)
        s_pred = student_logits.argmax(dim=1)
        return float((t_pred == s_pred).float().mean().item())


def prediction_stats(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return agreement/confidence diagnostics for KD logging and gating."""
    teacher_probs = F.softmax(teacher_logits, dim=1)
    student_probs = F.softmax(student_logits, dim=1)
    teacher_conf, teacher_pred = teacher_probs.max(dim=1)
    student_conf, student_pred = student_probs.max(dim=1)

    agreement_mask = teacher_pred == student_pred
    stats = {
        "agreement": float(agreement_mask.float().mean().detach().item()),
        "joint_confidence": float(
            torch.sqrt((teacher_conf * student_conf).clamp_min(1e-12)).mean().detach().item()
        ),
        "teacher_confidence": float(teacher_conf.mean().detach().item()),
        "student_confidence": float(student_conf.mean().detach().item()),
    }
    if labels is not None:
        flat_labels = labels.view(-1).long().to(teacher_logits.device)
        teacher_correct = teacher_pred == flat_labels
        student_correct = student_pred == flat_labels
        stats["teacher_batch_accuracy"] = float(teacher_correct.float().mean().detach().item())
        stats["student_batch_accuracy"] = float(student_correct.float().mean().detach().item())
        stats["correct_agreement"] = float(
            (agreement_mask & teacher_correct & student_correct).float().mean().detach().item()
        )
    else:
        stats["teacher_batch_accuracy"] = 0.0
        stats["student_batch_accuracy"] = 0.0
        stats["correct_agreement"] = 0.0
    return stats


def directional_kd_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    temperature: float,
    mean_class_weight: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """One-way KD loss that updates target_logits toward source_logits.

    This is KL(p_source || p_target) implemented as kl_div(log p_target, p_source).
    The source distribution is detached; gradients flow only through target_logits.
    """
    t = max(float(temperature), 1e-6)
    source_prob = F.softmax(source_logits.detach() / t, dim=1)
    target_log_prob = F.log_softmax(target_logits / t, dim=1)
    weight = torch.as_tensor(mean_class_weight, device=target_logits.device, dtype=target_logits.dtype)
    return weight * F.kl_div(target_log_prob, source_prob, reduction="batchmean") * (t**2)


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation module for teacher-to-student transfer."""

    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mean_class_weight: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        return directional_kd_loss(
            teacher_logits,
            student_logits,
            temperature=self.temperature,
            mean_class_weight=mean_class_weight,
        )


class FeatureAligner(nn.Module):
    """Feature aligner projecting teacher latent dimension to student feature dimension."""

    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim)

    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        return self.proj(teacher_features)


def disagreement_gated_teacher_to_student_kd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float,
    mean_class_weight: torch.Tensor | float = 1.0,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One-way VCT→student KD weighted by prediction disagreement.

    Gradients never flow into the teacher.  The gate is ``1 - agreement``; the
    reverse student→teacher path from the legacy implementation does not exist.
    """
    stats = prediction_stats(teacher_logits, student_logits, labels=labels)
    agreement = torch.as_tensor(
        stats["agreement"], device=teacher_logits.device, dtype=teacher_logits.dtype
    )
    t2s = (1.0 - agreement) * directional_kd_loss(
        teacher_logits,
        student_logits,
        temperature=temperature,
        mean_class_weight=mean_class_weight,
    )
    return t2s, stats

def mse_cosine_alignment(
    projected_teacher_features: torch.Tensor,
    student_features: torch.Tensor,
    *,
    cosine_weight: float = 0.5,
    mse_weight: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Lightweight Sentinel feature alignment: geometric + directional terms."""
    mse = F.mse_loss(projected_teacher_features, student_features, reduction="mean")
    cosine_score = F.cosine_similarity(projected_teacher_features, student_features, dim=1).mean()
    cosine_loss = 1.0 - cosine_score
    loss = (float(mse_weight) * mse) + (float(cosine_weight) * cosine_loss)
    return loss, float(cosine_score.detach().clamp(-1.0, 1.0).item())
