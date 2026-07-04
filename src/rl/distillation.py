"""Dynamic knowledge distillation utilities for DKD-FedOS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_temperature(
    round_num: int,
    *,
    base: float = 3.0,
    minimum: float = 1.0,
    decay: float = 0.95,
) -> float:
    exponent = min(float(max(int(round_num), 0)) / 10.0, 5.0)
    return float(max(float(minimum), float(base) * (float(decay) ** exponent)))


def prediction_agreement_and_confidence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_probs = F.softmax(teacher_logits, dim=1)
    student_probs = F.softmax(student_logits, dim=1)
    teacher_conf, teacher_pred = teacher_probs.max(dim=1)
    student_conf, student_pred = student_probs.max(dim=1)
    agreement = (teacher_pred == student_pred).float().mean()
    joint_confidence = torch.sqrt((teacher_conf * student_conf).clamp_min(1e-12)).mean()
    return agreement, joint_confidence


def bidirectional_kd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float,
    mean_class_weight: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, float, float]:
    """Sentinel-style confidence-weighted bidirectional KL loss."""
    t = max(float(temperature), 1e-6)
    agreement, joint_conf = prediction_agreement_and_confidence(teacher_logits, student_logits)
    p_teacher = F.softmax(teacher_logits / t, dim=1)
    p_student = F.softmax(student_logits / t, dim=1)
    log_teacher = F.log_softmax(teacher_logits / t, dim=1)
    log_student = F.log_softmax(student_logits / t, dim=1)

    # KL(pS || pT): use teacher target distribution, student log distribution
    kl_student_to_teacher = F.kl_div(log_student, p_teacher.detach(), reduction="batchmean")
    # KL(pT || pS): global/student teaches back only under confident agreement.
    kl_teacher_to_student = F.kl_div(log_teacher, p_student.detach(), reduction="batchmean")
    weight = torch.as_tensor(mean_class_weight, device=teacher_logits.device, dtype=teacher_logits.dtype)
    loss = weight * (((1.0 - agreement) * kl_student_to_teacher) + (joint_conf * kl_teacher_to_student)) * (t**2)
    return loss, float(agreement.detach().item()), float(joint_conf.detach().item())


def mse_cosine_alignment(
    projected_teacher_features: torch.Tensor,
    student_features: torch.Tensor,
    *,
    cosine_weight: float = 0.5,
    mse_weight: float = 1.0,
) -> tuple[torch.Tensor, float]:
    mse = F.mse_loss(projected_teacher_features, student_features, reduction="mean")
    cosine_score = F.cosine_similarity(projected_teacher_features, student_features, dim=1).mean()
    cosine_loss = 1.0 - cosine_score
    loss = (float(mse_weight) * mse) + (float(cosine_weight) * cosine_loss)
    return loss, float(cosine_score.detach().clamp(-1.0, 1.0).item())
