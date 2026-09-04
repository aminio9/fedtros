"""Training routines and optimization utilities for FedTROS-PR."""

from src.training.centralized import run_training
from src.training.distillation import (
    FeatureAligner,
    KnowledgeDistillationLoss,
    compute_prediction_agreement,
    directional_kd_loss,
    kd_temperature,
    mse_cosine_alignment,
    prediction_stats,
    disagreement_gated_teacher_to_student_kd,
)
from src.training.local_training import run_local_training_round

__all__ = [
    "run_training",
    "run_local_training_round",
    "kd_temperature",
    "prediction_stats",
    "directional_kd_loss",
    "disagreement_gated_teacher_to_student_kd",
    "mse_cosine_alignment",
    "compute_prediction_agreement",
    "KnowledgeDistillationLoss",
    "FeatureAligner",
]
