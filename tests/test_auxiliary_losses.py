import torch
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.models.cvae_dqn import OpenSetQChainModelFactory
from src.training.losses import (
    center_compactness_loss,
    diagonal_gaussian_kl,
    focal_cross_entropy_loss,
    kl_warmup_weight,
    smooth_reconstruction_loss,
    supervised_contrastive_loss,
)


def _model_cfg():
    return OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 3})


def _training_cfg(*, supcon: float = 0.0, center: float = 0.0):
    return OmegaConf.create(
        {
            "gamma": 0.7,
            "use_double_dqn": True,
            "lr_prior": 1e-3,
            "lr_q_rl": 1e-3,
            "prior_grad_clip_norm": 1.0,
            "q_grad_clip_norm": 1.0,
            "prior_kl_raw": False,
            "loss_weights": {
                "prior_kl": 1.0,
                "q_td": 1.0,
                "classification": 1.0,
                "generator_reconstruction": 1.0,
                "proximal": 1.0,
            },
            "auxiliary_losses": {
                "supervised_contrastive_lambda": supcon,
                "supervised_contrastive_temperature": 0.1,
                "center_loss_lambda": center,
            },
        }
    )


def _batch(batch_size: int = 6):
    labels = torch.tensor([[0], [0], [1], [1], [2], [2]], dtype=torch.long)[:batch_size]
    return (
        torch.randn(batch_size, 5),
        labels.clone(),
        torch.ones(batch_size, 1),
        torch.randn(batch_size, 5),
        torch.zeros(batch_size, 1),
        labels,
    )


def test_supervised_contrastive_loss_returns_finite_scalar_and_backpropagates():
    embeddings = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = supervised_contrastive_loss(embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()


def test_center_compactness_loss_returns_finite_scalar_and_backpropagates():
    embeddings = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = center_compactness_loss(embeddings, labels)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()


def test_focal_cross_entropy_returns_finite_scalar_and_backpropagates():
    logits = torch.randn(8, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    class_weights = torch.tensor([0.5, 1.5, 1.0])

    loss = focal_cross_entropy_loss(
        logits,
        labels,
        class_weights=class_weights,
        gamma=2.0,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_diagonal_gaussian_kl_free_bits_and_warmup_scaling():
    mu = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)

    raw_kl = diagonal_gaussian_kl(mu, logvar, mu, logvar, free_nats=0.0)
    free_kl = diagonal_gaussian_kl(mu, logvar, mu, logvar, free_nats=0.25)

    assert raw_kl.item() == 0.0
    assert torch.isclose(free_kl, torch.tensor(0.75))
    assert kl_warmup_weight(0, 10) == 0.0
    assert kl_warmup_weight(5, 10) == 0.5
    assert kl_warmup_weight(10, 10) == 1.0


def test_smooth_reconstruction_loss_returns_finite_scalar_and_backpropagates():
    recon = torch.randn(5, 4, requires_grad=True)
    target = torch.randn(5, 4)

    loss = smooth_reconstruction_loss(recon, target, beta=1.0)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(recon.grad).all()


def test_agent_auxiliary_losses_are_logged_and_backpropagate():
    torch.manual_seed(7)
    agent = Agent(
        OpenSetQChainModelFactory(_model_cfg()),
        _training_cfg(supcon=0.01, center=0.01),
        torch.device("cpu"),
    )

    metrics = agent.train_step(_batch())

    assert torch.isfinite(torch.tensor(metrics["loss/total"]))
    assert metrics["loss/supervised_contrastive"] >= 0.0
    assert metrics["loss/center_compactness"] >= 0.0
    assert metrics["gradient/prior_norm"] > 0.0


def test_disabling_auxiliary_losses_keeps_weighted_terms_zero():
    agent = Agent(OpenSetQChainModelFactory(_model_cfg()), _training_cfg(), torch.device("cpu"))

    metrics = agent.train_step(_batch())

    assert metrics["loss/supervised_contrastive_weighted"] == 0.0
    assert metrics["loss/center_compactness_weighted"] == 0.0
