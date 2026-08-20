"""Test disjoint prototype-fit vs threshold-calibration split in FedTROS-PR."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.models.student import StudentIDSModel
from src.evaluation.run import (
    _compose_prototype_rank_runtime_config,
    _resolve_prototype_rank_checkpoint,
)
from src.openset.prototype_rank_pipeline import calibrate_prototype_rank


def test_runtime_config_includes_protocol_level_calibration():
    cfg = OmegaConf.create({
        "unknown_label_id": -7,
        "open_set_label_id": 77,
        "calibration": {"min_samples_per_class": 2},
        "prototype_rank": {"enabled": True, "prototype": {"enabled": True}},
    })
    runtime = _compose_prototype_rank_runtime_config(cfg)
    assert runtime.unknown_label_id == -7
    assert runtime.open_set_label_id == 77
    assert runtime.calibration.min_samples_per_class == 2
    assert runtime.prototype.enabled is True


def test_prototype_rank_checkpoint_falls_back_to_canonical_path(tmp_path):
    canonical = tmp_path / "checkpoints" / "latest.pt"
    canonical.parent.mkdir()
    canonical.touch()
    cfg = OmegaConf.create({
        "checkpointing": {"dir": str(canonical.parent)},
        "evaluation": {"checkpoint_path": str(canonical)},
    })

    assert _resolve_prototype_rank_checkpoint(cfg, project_root=tmp_path) == canonical


def test_disjoint_calibration_split():
    torch.manual_seed(42)
    np.random.seed(42)

    num_samples = 100
    feature_dim = 16
    num_classes = 4

    features = torch.randn(num_samples, feature_dim)
    labels = torch.randint(0, num_classes, (num_samples,))

    student = StudentIDSModel(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dims=[64, 32],
        osr_enabled=True,
        osr_latent_dim=8,
        osr_hidden_dims=[32, 16],
        osr_decoder_hidden_dims=[16, 32],
    )
    student.eval()

    cfg = OmegaConf.create(
        {
            "unknown_label_id": -1,
            "open_set_label_id": 99,
            "calibration": {
                "prototype_fit_fraction": 0.70,
                "threshold_calibration_fraction": 0.30,
                "target_known_fpr": 0.05,
                "min_samples_per_class": 2,
                "fit_correct_only": False,
                "strict_disjoint": True,
            },
            "prototype": {
                "feature_source": "osr_mu",
                "normalize": True,
                "radius_quantile": 0.95,
                "num_prototypes_per_class": 4,
                "min_samples_per_prototype": 1,
                "seed": 42,
                "negative": {
                    "enabled": True,
                    "num_prototypes": 8,
                    "max_samples": 100,
                    "mixup_alpha": 1.0,
                    "noise_std": 0.005,
                    "radius_quantile": 0.75,
                    "weight": 0.35,
                    "random_seed": 43,
                },
            },
            "score_fusion": {
                "method": "prototype_rank",
            },
        }
    )

    prototype_bank, df, meta = calibrate_prototype_rank(
        features,
        labels,
        student_model=student,
        batch_size=32,
        device=torch.device("cpu"),
        cfg=cfg,
    )

    assert meta["backend"] == "prototype_rank"
    assert "split_provenance" in meta
    prov = meta["split_provenance"]
    assert prov["disjoint_split"] is True
    assert prov["prototype_fit_samples"] > 0
    assert prov["threshold_calibration_samples"] > 0
    assert prov["prototype_fit_samples"] + prov["threshold_calibration_samples"] == num_samples
    assert "proto_indices_hash" in prov
    assert "calib_indices_hash" in prov
    assert len(prototype_bank.prototypes) == num_classes
