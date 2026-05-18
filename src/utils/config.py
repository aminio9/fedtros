from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

REQUIRED_CONFIG_KEYS = (
    "seed",
    "device.prefer",
    "dataset.name",
    "dataset.preprocessing.raw_file",
    "dataset.preprocessing.output_dir",
    "dataset.preprocessing.label_column",
    "dataset.preprocessing.known_labels",
    "model.name",
    "model.state_dim",
    "model.latent_dim",
    "model.num_actions",
    "training.epochs",
    "training.batch_size",
    "training.local_episodes_per_round",
    "training.steps_per_episode",
    "training.replay_buffer_size",
    "training.min_buffer_size",
    "training.gamma",
    "training.lr_prior",
    "training.lr_q_rl",
    "training.tau",
    "training.target_update_freq",
    "training.epsilon_start",
    "training.epsilon_end",
    "training.epsilon_decay_rate",
    "federated.num_clients",
    "federated.num_rounds",
    "federated.server.address",
    "federated.strategy.name",
    "open_set.evt.tail_size_percent",
    "open_set.evt.target_known_fpr",
    "plotting.required_plots",
    "tracking.run_dir",
    "checkpointing.latest_checkpoint_path",
    "logging.level",
)


def validate_config(cfg: DictConfig, extra_required: Iterable[str] = ()) -> None:
    """Fail early when a required Hydra key is missing or unresolved."""
    missing: list[str] = []
    for key in tuple(REQUIRED_CONFIG_KEYS) + tuple(extra_required):
        value = OmegaConf.select(cfg, key, default=None)
        if value is None or value == "???":
            missing.append(key)
    if missing:
        formatted = "\n  - ".join(missing)
        raise ValueError(f"Missing required Hydra config values:\n  - {formatted}")

    known_labels = OmegaConf.select(cfg, "dataset.preprocessing.known_labels")
    if not known_labels:
        raise ValueError("dataset.preprocessing.known_labels must contain at least one label.")

    num_clients = int(OmegaConf.select(cfg, "federated.num_clients"))
    if num_clients <= 0:
        raise ValueError("federated.num_clients must be positive.")
    preprocessing_num_clients = int(OmegaConf.select(cfg, "dataset.preprocessing.num_clients"))
    if preprocessing_num_clients != num_clients:
        raise ValueError(
            "dataset.preprocessing.num_clients must match federated.num_clients. "
            "Override federated.num_clients to change the client count for preprocessing and FL."
        )

    batch_size = int(OmegaConf.select(cfg, "training.batch_size"))
    min_buffer_size = int(OmegaConf.select(cfg, "training.min_buffer_size"))
    if batch_size <= 0 or min_buffer_size <= 0:
        raise ValueError("training.batch_size and training.min_buffer_size must be positive.")


def resolve_path(project_root: Path, path_like: str | Path) -> Path:
    """Resolve a project-relative or absolute path."""
    path = Path(path_like)
    return path if path.is_absolute() else (project_root / path)


def to_plain_container(cfg: DictConfig, *, resolve: bool = True) -> dict[str, Any]:
    """Convert a Hydra config to a plain Python dictionary."""
    return OmegaConf.to_container(cfg, resolve=resolve, throw_on_missing=True)  # type: ignore[return-value]
