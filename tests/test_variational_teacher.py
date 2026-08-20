"""Unit tests for the Private Variational Classifier Teacher (VCT)."""

import pytest
import torch

from src.models.variational_teacher import (
    VariationalClassifierTeacher,
    kl_standard_normal,
)


@pytest.fixture
def teacher():
    return VariationalClassifierTeacher(
        input_dim=40,
        num_classes=10,
        latent_dim=64,
        hidden_dims=(128, 64),
    )


def test_teacher_output_shapes(teacher):
    batch_size = 16
    x = torch.randn(batch_size, 40)
    logits, mu, logvar, h = teacher(x, sample=True)

    assert logits.shape == (batch_size, 10)
    assert mu.shape == (batch_size, 64)
    assert logvar.shape == (batch_size, 64)
    assert h.shape == (batch_size, 64)


def test_teacher_kl_finite(teacher):
    batch_size = 8
    x = torch.randn(batch_size, 40)
    _logits, mu, logvar, _h = teacher(x, sample=True)
    kl = kl_standard_normal(mu, logvar)

    assert torch.isfinite(kl)
    assert kl.dim() == 0  # scalar mean


def test_teacher_kl_nonnegative():
    # Exactly standard normal -> KL should be 0.0
    mu_zero = torch.zeros(4, 16)
    logvar_zero = torch.zeros(4, 16)
    kl_zero = kl_standard_normal(mu_zero, logvar_zero)
    assert abs(kl_zero.item()) < 1e-6

    # Arbitrary random inputs -> KL must be strictly >= 0
    for _ in range(10):
        mu = torch.randn(8, 32) * 2.0
        logvar = torch.randn(8, 32)
        kl = kl_standard_normal(mu, logvar)
        assert kl.item() >= -1e-6


def test_training_is_stochastic(teacher):
    teacher.train()
    x = torch.randn(8, 40)
    torch.manual_seed(42)
    logits1, _, _, _ = teacher(x, sample=True)
    logits2, _, _, _ = teacher(x, sample=True)

    # Stochastic reparameterization should produce different outputs across calls
    assert not torch.allclose(logits1, logits2, atol=1e-5)


def test_distillation_is_deterministic(teacher):
    teacher.eval()
    x = torch.randn(8, 40)
    logits1, mu1, h1 = teacher.distill_forward(x)
    logits2, mu2, h2 = teacher.distill_forward(x)

    # Distillation uses mu directly with no random sampling
    assert torch.allclose(logits1, logits2, atol=1e-7)
    assert torch.allclose(mu1, mu2, atol=1e-7)
    assert torch.allclose(h1, h2, atol=1e-7)


def test_mu_receives_gradient(teacher):
    x = torch.randn(8, 40)
    _logits, mu, _logvar, _h = teacher(x, sample=True)
    loss = mu.sum()
    loss.backward()

    assert teacher.mu_head.weight.grad is not None
    assert teacher.mu_head.weight.grad.norm().item() > 0.0


def test_logvar_receives_gradient(teacher):
    x = torch.randn(8, 40)
    _logits, _mu, logvar, _h = teacher(x, sample=True)
    loss = logvar.sum()
    loss.backward()

    assert teacher.logvar_head.weight.grad is not None
    assert teacher.logvar_head.weight.grad.norm().item() > 0.0


def test_classifier_receives_gradient(teacher):
    x = torch.randn(8, 40)
    logits, _mu, _logvar, _h = teacher(x, sample=True)
    loss = logits.sum()
    loss.backward()

    assert teacher.classifier.weight.grad is not None
    assert teacher.classifier.weight.grad.norm().item() > 0.0


def test_no_q_parameters(teacher):
    param_names = [name for name, _ in teacher.named_parameters()]
    forbidden_substrings = ["value_net", "q_net", "target_q", "q_network", "dueling", "advantage"]
    for name in param_names:
        for forbidden in forbidden_substrings:
            assert forbidden not in name.lower(), f"Forbidden substring '{forbidden}' in parameter name '{name}'"


def test_no_rl_state(teacher):
    assert not hasattr(teacher, "replay_buffer")
    assert not hasattr(teacher, "env")
    assert not hasattr(teacher, "epsilon")
    assert not hasattr(teacher, "gamma")
