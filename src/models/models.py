"""PyTorch model definitions for FedTROS-PR."""

import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.common import ResidualBlock, TabularTransformerEncoder
from src.models.student import StudentIDSModel
from src.models.variational_teacher import (
    VariationalClassifierTeacher,
    kl_standard_normal,
)

logger = logging.getLogger("Models")


class ModelFactory:
    """Factory to construct Teacher and Student models from model configuration."""

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.feature_dim = int(getattr(model_cfg, "feature_dim", getattr(model_cfg, "state_dim", 40)))
        self.latent_dim = int(getattr(model_cfg, "latent_dim", 64))
        self.num_classes = int(getattr(model_cfg, "num_classes", getattr(model_cfg, "num_actions", 4)))
        self.transformer_cfg = getattr(model_cfg, "transformer", None)

    def create_teacher(self, hidden_dims: tuple[int, ...] = (512, 256)) -> VariationalClassifierTeacher:
        return VariationalClassifierTeacher(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            latent_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            transformer_cfg=self.transformer_cfg,
        )

    def create_student(
        self,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        *,
        osr_enabled: bool = True,
        osr_latent_dim: int = 8,
        open_set_enabled: bool = False,
    ) -> StudentIDSModel:
        return StudentIDSModel(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dims=hidden_dims,
            osr_enabled=osr_enabled,
            osr_latent_dim=osr_latent_dim,
            open_set_enabled=open_set_enabled,
        )


# Aliases for FedTROS-PR model factory
FedTROSModelFactory = ModelFactory

__all__ = [
    "ResidualBlock",
    "TabularTransformerEncoder",
    "VariationalClassifierTeacher",
    "StudentIDSModel",
    "ModelFactory",
    "FedTROSModelFactory",
    "kl_standard_normal",
]
