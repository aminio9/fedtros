"""Integration tests for Variational Classifier Teacher + Student IDS Model in FedTROS-PR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.student import StudentIDSModel
from src.models.variational_teacher import VariationalClassifierTeacher
from src.training.class_balance import class_balanced_cross_entropy
from src.training.distillation import directional_kd_loss, mse_cosine_alignment


def test_kd_output_matching():
    input_dim = 40
    num_classes = 8
    latent_dim = 64
    batch_size = 16

    teacher = VariationalClassifierTeacher(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hidden_dims=(128, 64),
    )
    student = StudentIDSModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[64, 32, 16],
    )

    x = torch.randn(batch_size, input_dim)
    teacher.eval()
    student.train()

    with torch.no_grad():
        teacher_logits, teacher_mu, _ = teacher.distill_forward(x)
        teacher_logits = teacher_logits.detach()

    student_features, student_logits = student(x)

    assert teacher_logits.shape == student_logits.shape == (batch_size, num_classes)
    kd_loss = directional_kd_loss(teacher_logits, student_logits, temperature=2.0)
    assert torch.isfinite(kd_loss)
    assert kd_loss.item() >= 0.0


def test_feature_alignment():
    input_dim = 40
    num_classes = 8
    latent_dim = 64
    batch_size = 16

    teacher = VariationalClassifierTeacher(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hidden_dims=(128, 64),
    )
    student = StudentIDSModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[64, 32, 16],
    )
    # Aligner projects teacher mu (latent_dim) -> student feature dim
    aligner = nn.Linear(latent_dim, student.feature_dim)

    x = torch.randn(batch_size, input_dim)
    with torch.no_grad():
        _, teacher_mu, _ = teacher.distill_forward(x)
        teacher_mu = teacher_mu.detach()

    student_features, _ = student(x)
    projected_teacher = aligner(teacher_mu)

    assert projected_teacher.shape == student_features.shape == (batch_size, student.feature_dim)
    align_loss, cos_score = mse_cosine_alignment(projected_teacher, student_features)

    assert torch.isfinite(align_loss)
    assert -1.0 <= cos_score <= 1.0


def test_strict_gradient_isolation():
    input_dim = 40
    num_classes = 8
    latent_dim = 64
    batch_size = 16

    teacher = VariationalClassifierTeacher(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hidden_dims=(128, 64),
    )
    student = StudentIDSModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[64, 32, 16],
    )
    aligner = nn.Linear(latent_dim, student.feature_dim)

    teacher.eval()
    student.train()
    aligner.train()

    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))

    # 1. Deterministic teacher evaluation without gradients
    with torch.no_grad():
        teacher_logits, teacher_mu, _ = teacher.distill_forward(x)
        teacher_logits = teacher_logits.detach()
        teacher_mu = teacher_mu.detach()

    # 2. Student forward pass
    student_features, student_logits = student(x)

    # 3. Student losses
    loss_task = class_balanced_cross_entropy(student_logits, y)
    loss_kd = directional_kd_loss(teacher_logits, student_logits, temperature=2.0)
    loss_align, _ = mse_cosine_alignment(aligner(teacher_mu), student_features)

    student_total_loss = loss_task + 0.2 * loss_kd + 0.1 * loss_align

    # 4. Backward pass
    student_total_loss.backward()

    # Student and aligner must have gradients
    assert any(p.grad is not None and p.grad.norm() > 0 for p in student.parameters())
    assert aligner.weight.grad is not None and aligner.weight.grad.norm() > 0

    # Teacher parameters MUST NEVER receive gradients from student updates
    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher parameter '{name}' unexpectedly received gradient!"
