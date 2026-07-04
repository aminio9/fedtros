import numpy as np
import torch
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.models.models import OpenSetQChainModelFactory
from src.rl.class_balance import effective_number_class_weights


def _model_cfg():
    return OmegaConf.create({"state_dim": 5, "latent_dim": 3, "num_actions": 4})


def _training_cfg():
    return OmegaConf.create(
        {
            "gamma": 0.7,
            "use_double_dqn": True,
            "lr_prior": 1e-3,
            "lr_q_rl": 1e-3,
            "prior_grad_clip_norm": 1.0,
            "prior_kl_raw": False,
            "dkd_student_hidden_dims": [8, 4],
            "dkd_student_lr": 1e-3,
            "dkd_lambda_kd_init": 0.20,
            "dkd_lambda_align_init": 0.08,
        }
    )


def test_effective_number_weights_upweight_minority_class():
    labels = torch.tensor([0] * 20 + [1] * 2 + [2] * 1)
    weights = effective_number_class_weights(labels, 4, beta=0.999, device="cpu")
    assert weights[2] > weights[1] > weights[0]
    assert torch.isfinite(weights).all()


def test_dkd_train_step_updates_student_and_records_losses():
    factory = OpenSetQChainModelFactory(_model_cfg())
    agent = Agent(factory, _training_cfg(), torch.device("cpu"))
    before = [p.copy() for p in agent.get_student_parameters()]

    batch_size = 4
    batch = (
        torch.randn(batch_size, 5),
        torch.zeros(batch_size, 1, dtype=torch.long),
        torch.ones(batch_size, 1),
        torch.randn(batch_size, 5),
        torch.zeros(batch_size, 1),
        torch.tensor([[0], [1], [1], [2]], dtype=torch.long),
    )

    td_loss, kl_loss, prox_loss, avg_q = agent.train_step(
        batch,
        aux_ce_weight=0.1,
        aux_ce_label_smoothing=0.01,
        dkd_enabled=True,
        dkd_round=2,
        dkd_class_weights=torch.ones(4),
        dkd_present_classes=torch.tensor([1, 1, 1, 0], dtype=torch.bool),
    )

    after = agent.get_student_parameters()
    assert any(not np.allclose(a, b) for a, b in zip(before, after, strict=True))
    assert td_loss >= 0.0
    assert kl_loss >= 0.0
    assert prox_loss >= 0.0
    assert agent.last_dkd_task_loss > 0.0
    assert agent.last_dkd_kd_loss >= 0.0
    assert agent.last_dkd_align_loss >= 0.0
    assert np.isfinite(avg_q)


def test_dataset_dkd_freezes_teacher_by_default_but_updates_student():
    cfg = _training_cfg()
    cfg.dkd_dataset_training = True
    cfg.dkd_dataset_update_teacher = False
    cfg.dkd_update_teacher_from_student = False
    cfg.dkd_local_epochs = 1
    cfg.dkd_batch_size = 4
    cfg.dkd_teacher_to_student_start_round = 1
    cfg.dkd_alignment_start_round = 1
    cfg.dkd_student_to_teacher_start_round = 999
    factory = OpenSetQChainModelFactory(_model_cfg())
    agent = Agent(factory, cfg, torch.device("cpu"))

    teacher_before = [p.detach().clone() for p in agent.prior_net.parameters()] + [
        p.detach().clone() for p in agent.value_net_main.parameters()
    ]
    student_before = [p.copy() for p in agent.get_student_parameters()]

    features = torch.randn(16, 5)
    labels = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
    metrics = agent.train_dkd_fedos_dataset(
        features=features,
        labels=labels,
        cfg_training=cfg,
        round_num=2,
        class_weights=torch.ones(4),
        present_classes=torch.ones(4, dtype=torch.bool),
        device=torch.device("cpu"),
    )

    teacher_after = [p.detach().clone() for p in agent.prior_net.parameters()] + [
        p.detach().clone() for p in agent.value_net_main.parameters()
    ]
    student_after = agent.get_student_parameters()

    assert metrics["dkd_dataset_updates_teacher"] == 0.0
    assert metrics["dkd_update_teacher_from_student"] == 0.0
    assert metrics["dkd_dataset_train_steps"] > 0.0
    assert all(torch.allclose(a, b) for a, b in zip(teacher_before, teacher_after, strict=True))
    assert any(not np.allclose(a, b) for a, b in zip(student_before, student_after, strict=True))
