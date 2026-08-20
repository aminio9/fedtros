"""Workstream A Scientific Core Verification Tests for FedTROS-PR.

Validates VCT formulation, student gradient isolation, prototype-rank rejection,
70/30 disjoint calibration, preprocessing isolation, and payload boundaries.
"""

import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from src.data.preprocessing import run_preprocessing
from src.models.bundle import FedTROSModelBundle
from src.models.models import FedTROSModelFactory
from src.models.variational_teacher import VariationalClassifierTeacher, kl_standard_normal
from src.openset.prototype_bank import PrototypeBank, fit_prototype_bank
from src.openset.prototype_rank import (
    compute_prototype_rank_scores,
    evaluate_prototype_rank_rejection,
)
from src.openset.rank_calibration import (
    empirical_cdf_rank,
    fit_empirical_rank_threshold,
    make_stratified_70_30_split,
)


@pytest.fixture
def model_cfg():
    return OmegaConf.create({
        "feature_dim": 20,
        "latent_dim": 32,
        "num_classes": 4,
    })


@pytest.fixture
def train_cfg():
    return OmegaConf.create({
        "teacher_lr": 1e-3,
        "student_lr": 1e-3,
        "teacher_beta_kl": 0.01,
        "lambda_kd_init": 0.20,
        "lambda_align_init": 0.08,
        "fedtros_global_anchor_weight": 2.0,
        "fedtros_student_hidden_dims": [64, 32, 16],
    })


def test_teacher_output_shapes():
    teacher = VariationalClassifierTeacher(input_dim=20, num_classes=4, latent_dim=32, hidden_dims=(64, 32))
    x = torch.randn(8, 20)
    logits, mu, logvar, h = teacher(x, sample=True)
    assert logits.shape == (8, 4)
    assert mu.shape == (8, 32)
    assert logvar.shape == (8, 32)
    assert h.shape == (8, 32)


def test_teacher_kl_nonnegative():
    mu_zero = torch.zeros(4, 16)
    logvar_zero = torch.zeros(4, 16)
    kl_zero = kl_standard_normal(mu_zero, logvar_zero)
    assert abs(kl_zero.item()) < 1e-6

    for _ in range(5):
        mu = torch.randn(8, 32) * 2.0
        logvar = torch.randn(8, 32)
        kl = kl_standard_normal(mu, logvar)
        assert kl.item() >= -1e-6


def test_teacher_stochastic_training():
    teacher = VariationalClassifierTeacher(input_dim=20, num_classes=4, latent_dim=32)
    teacher.train()
    x = torch.randn(8, 20)
    logits1, _, _, _ = teacher(x, sample=True)
    logits2, _, _, _ = teacher(x, sample=True)
    assert not torch.allclose(logits1, logits2, atol=1e-5)


def test_teacher_deterministic_distillation():
    teacher = VariationalClassifierTeacher(input_dim=20, num_classes=4, latent_dim=32)
    teacher.eval()
    x = torch.randn(8, 20)
    logits1, mu1, h1 = teacher.distill_forward(x)
    logits2, mu2, h2 = teacher.distill_forward(x)
    assert torch.allclose(logits1, logits2, atol=1e-7)
    assert torch.allclose(mu1, mu2, atol=1e-7)
    assert torch.allclose(h1, h2, atol=1e-7)


def test_teacher_gradients():
    teacher = VariationalClassifierTeacher(input_dim=20, num_classes=4, latent_dim=32)
    x = torch.randn(8, 20)
    logits, mu, logvar, _ = teacher(x, sample=True)
    loss = logits.sum() + kl_standard_normal(mu, logvar)
    loss.backward()

    assert teacher.mu_head.weight.grad is not None
    assert teacher.logvar_head.weight.grad is not None
    assert teacher.classifier.weight.grad is not None


def test_no_rl_parameters(model_cfg, train_cfg):
    factory = FedTROSModelFactory(model_cfg)
    bundle = FedTROSModelBundle(factory, train_cfg, torch.device("cpu"))

    forbidden = ["q_net", "value_net", "target_q", "replay_buffer", "epsilon", "gamma"]
    for term in forbidden:
        assert not hasattr(bundle, term)
        assert not hasattr(bundle.teacher, term)
        assert not hasattr(bundle.student_model, term)


def test_student_does_not_backprop_teacher(model_cfg, train_cfg):
    factory = FedTROSModelFactory(model_cfg)
    bundle = FedTROSModelBundle(factory, train_cfg, torch.device("cpu"))

    features = torch.randn(16, 20)
    labels = torch.randint(0, 4, (16,))

    bundle.train_student_step(
        features,
        labels,
        round_num=1,
        t2s_start_round=1,
        align_start_round=1,
    )

    for name, p in bundle.teacher.named_parameters():
        assert p.grad is None, f"Teacher parameter '{name}' received gradient during student step!"


def test_alignment_dimensions(model_cfg, train_cfg):
    factory = FedTROSModelFactory(model_cfg)
    bundle = FedTROSModelBundle(factory, train_cfg, torch.device("cpu"))

    mu_t = torch.randn(8, bundle.teacher.latent_dim)
    projected = bundle.teacher_to_student_aligner(mu_t)
    assert projected.shape == (8, bundle.student_model.feature_dim)


def test_federated_payload_student_only(model_cfg, train_cfg):
    factory = FedTROSModelFactory(model_cfg)
    bundle = FedTROSModelBundle(factory, train_cfg, torch.device("cpu"))

    params = bundle.get_federated_parameters()
    student_state_keys = list(bundle.student_model.state_dict().keys())
    assert len(params) == len(student_state_keys)


def test_known_only_preprocessing(tmp_path):
    # Construct raw DataFrame with known and unknown classes
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "cat1": ["A", "B", "A", "B", "C", "A", "B", "D"],
        "label": ["Normal", "DoS", "Normal", "DoS", "FoT", "Normal", "DoS", "FoT"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    cfg = OmegaConf.create({
        "seed": 42,
        "dataset": {
            "name": "test_dataset",
            "preprocessing": {
                "raw_file": str(csv_path),
                "label_column": "label",
                "output_dir": str(tmp_path / "proc"),
                "protocol": "open_set",
                "known_labels": ["Normal", "DoS"],
                "unknown_labels": ["FoT"],
                "unknown_label_id": -1,
                "source_labels": ["Normal", "DoS", "FoT"],
                "closed_set_test_size": 0.25,
                "validation_split": 0.25,
                "numerical_cols": None,
                "categorical_cols": None,
                "numeric_threshold": 0.9,
                "num_clients": 2,
                "alpha": 0.5,
                "iid": True,
                "categorical_schema_scope": "known_train",
            },
        }
    })

    summary = run_preprocessing(cfg, project_root=tmp_path)
    assert summary["feature_dim"] > 0
    num_classes = summary.get("num_classes")
    assert num_classes == 2


def test_pr_disjoint_split():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    split = make_stratified_70_30_split(labels, seed=42, fit_fraction=0.70)
    split.assert_disjoint()
    assert len(split.fit_indices) + len(split.cal_indices) == len(labels)
    assert len(split.fit_sha256) == 64
    assert len(split.cal_sha256) == 64


def test_pr_rank_formula():
    ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    rank_low = empirical_cdf_rank(0.05, ref)
    rank_mid = empirical_cdf_rank(0.55, ref)
    rank_high = empirical_cdf_rank(1.5, ref)

    assert rank_low == 0.0
    assert abs(rank_mid - 0.5) < 0.1
    assert rank_high == 1.0


def test_e2_e4_prototype_rank():
    # Synthetic 2-class knowns with 1 unknown class
    rng = np.random.default_rng(42)
    c0 = rng.normal(loc=[1.0, 0.0], scale=0.1, size=(20, 2))
    c1 = rng.normal(loc=[-1.0, 0.0], scale=0.1, size=(20, 2))
    c_unknown = rng.normal(loc=[0.0, 2.0], scale=0.1, size=(10, 2))

    feats_by_class = {0: c0, 1: c1}
    bank = fit_prototype_bank(feats_by_class, num_prototypes_per_class=2, num_negative_prototypes=4)

    test_feats = np.vstack([c0[:5], c1[:5], c_unknown])
    test_preds = np.array([0] * 5 + [1] * 5 + [0] * 10)
    test_labels = np.array([0] * 5 + [1] * 5 + [-1] * 10)

    # Reference distance distribution on known calibration set
    cal_feats = np.vstack([c0[5:], c1[5:]])
    cal_preds = np.array([0] * 15 + [1] * 15)
    sorted_ref = {0: np.sort(bank.score(c0[5:], 0)), 1: np.sort(bank.score(c1[5:], 1))}

    cal_ranks = compute_prototype_rank_scores(cal_feats, cal_preds, bank, sorted_ref)
    threshold = fit_empirical_rank_threshold(cal_ranks, target_fpr=0.05)

    test_ranks = compute_prototype_rank_scores(test_feats, test_preds, bank, sorted_ref)
    metrics = evaluate_prototype_rank_rejection(test_labels, test_preds, test_ranks, threshold)
    assert metrics["auroc"] > 0.85
    assert metrics["unknown_recall"] > 0.80
