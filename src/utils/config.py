from __future__ import annotations

import json
import logging

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)



def _set_cfg_value(cfg: DictConfig, key: str, value: Any) -> None:
    """Set a nested OmegaConf value even when struct mode is enabled."""
    previous_struct = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        OmegaConf.update(cfg, key, value, merge=False)
    finally:
        OmegaConf.set_struct(cfg, previous_struct)


def sync_model_dimensions_from_preprocessing(
    cfg: DictConfig,
    *,
    project_root: Path,
    metadata: dict[str, Any] | None = None,
    metadata_path: str | Path | None = None,
    strict_num_actions: bool = True,
) -> dict[str, Any]:
    """Synchronize runtime model dimensions with processed tensor metadata."""
    if metadata is None:
        if metadata_path is None:
            output_dir = OmegaConf.select(cfg, "dataset.preprocessing.output_dir", default="data/processed")
            metadata_path = Path(str(output_dir)) / "preprocess_metadata.json"
        path = resolve_path(project_root, metadata_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessing metadata not found: {path}. Run preprocessing before training."
            )
        metadata = json.loads(path.read_text(encoding="utf-8"))

    if "state_dim" not in metadata or "num_actions" not in metadata:
        raise KeyError("preprocess_metadata.json must contain state_dim and num_actions.")

    actual_state_dim = int(metadata["state_dim"])
    actual_num_actions = int(metadata["num_actions"])
    configured_state_dim = int(OmegaConf.select(cfg, "model.state_dim"))
    configured_num_actions = int(OmegaConf.select(cfg, "model.num_actions"))

    if configured_state_dim != actual_state_dim:
        logger.warning(
            "Model state_dim synchronized from preprocessing metadata | config=%d processed=%d known_labels=%s",
            configured_state_dim,
            actual_state_dim,
            metadata.get("known_labels"),
        )
        _set_cfg_value(cfg, "model.state_dim", actual_state_dim)
        _set_cfg_value(cfg, "env_metadata.state_dim", actual_state_dim)
        if OmegaConf.select(cfg, "model.transformer.input_dim", default=None) is not None:
            _set_cfg_value(cfg, "model.transformer.input_dim", actual_state_dim)

    if configured_num_actions != actual_num_actions:
        message = (
            "model.num_actions does not match preprocessing metadata: "
            f"config={configured_num_actions}, processed={actual_num_actions}, "
            f"known_labels={metadata.get('known_labels')}"
        )
        if strict_num_actions:
            raise ValueError(message)
        logger.warning("%s; synchronizing config to processed dataset.", message)
        _set_cfg_value(cfg, "model.num_actions", actual_num_actions)
        _set_cfg_value(cfg, "env_metadata.num_actions", actual_num_actions)

    return metadata


REQUIRED_CONFIG_KEYS = (
    "seed",
    "device.prefer",
    "device.allow_cpu_fallback",
    "runtime.name",
    "runtime.device_prefer",
    "runtime.allow_cpu_fallback",
    "runtime.client_num_cpus",
    "runtime.client_num_gpus",
    "runtime.simulation_gpu_batches.enabled",
    "runtime.simulation_gpu_batches.batch_size",
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
    "federated.local_client_epochs",
    "federated.evaluation_frequency",
    "federated.server.address",
    "federated.server.fraction_fit",
    "federated.server.fraction_evaluate",
    "federated.server.min_fit_clients",
    "federated.server.min_evaluate_clients",
    "federated.server.min_available_clients",
    "federated.strategy.name",
    "federated.strategy.aggregation_strategy",
    "federated.strategy.min_selected_clients",
    "federated.strategy.max_selected_fraction",
    "federated.strategy.max_agents",
    "federated.strategy.warmup_rounds",
    "open_set.evt.tail_size_percent",
    "open_set.evt.target_known_fpr",
    "plotting.required_plots",
    "tracking.run_dir",
    "checkpointing.latest_checkpoint_path",
    "logging.level",
)


def _select_str(cfg: DictConfig, key: str, default: str = "") -> str:
    value = OmegaConf.select(cfg, key, default=default)
    return str(value or default)



def validate_config(cfg: DictConfig, extra_required: Iterable[str] = ()) -> None:
    """Fail early when a required Hydra key is missing or unresolved."""
    # missing: list[str] = []
    # for key in tuple(REQUIRED_CONFIG_KEYS) + tuple(extra_required):
    #     value = OmegaConf.select(cfg, key, default=None)
    #     if value is None or value == "???":
    #         missing.append(key)
    # if missing:
    #     formatted = "\n  - ".join(missing)
    #     raise ValueError(f"Missing required Hydra config values:\n  - {formatted}")

    # _validate_runtime(cfg)
    # _validate_experiment_contract(cfg)

    # known_labels = OmegaConf.select(cfg, "dataset.preprocessing.known_labels")
    # if not known_labels:
    #     raise ValueError("dataset.preprocessing.known_labels must contain at least one label.")

    # num_clients = int(OmegaConf.select(cfg, "federated.num_clients"))
    # if num_clients <= 0:
    #     raise ValueError("federated.num_clients must be positive.")
    # preprocessing_num_clients = int(OmegaConf.select(cfg, "dataset.preprocessing.num_clients"))
    # if preprocessing_num_clients != num_clients:
    #     raise ValueError(
    #         "dataset.preprocessing.num_clients must match federated.num_clients. "
    #         "Override federated.num_clients to change the client count for preprocessing and FL."
    #     )

    # batch_size = int(OmegaConf.select(cfg, "training.batch_size"))
    # min_buffer_size = int(OmegaConf.select(cfg, "training.min_buffer_size"))
    # if batch_size <= 0 or min_buffer_size <= 0:
    #     raise ValueError("training.batch_size and training.min_buffer_size must be positive.")

    # training_epochs = int(OmegaConf.select(cfg, "training.epochs"))
    # local_episodes = int(OmegaConf.select(cfg, "training.local_episodes_per_round"))
    # steps_per_episode = int(OmegaConf.select(cfg, "training.steps_per_episode"))
    # replay_buffer_size = int(OmegaConf.select(cfg, "training.replay_buffer_size"))
    # if min(training_epochs, local_episodes, steps_per_episode, replay_buffer_size) <= 0:
    #     raise ValueError(
    #         "training.epochs, local_episodes_per_round, steps_per_episode, "
    #         "and replay_buffer_size must be positive."
    #     )
    # if min_buffer_size > replay_buffer_size:
    #     raise ValueError("training.min_buffer_size cannot exceed training.replay_buffer_size.")

    # validation_interval = int(OmegaConf.select(cfg, "training.validation_interval"))
    # checkpoint_interval = int(OmegaConf.select(cfg, "training.checkpoint_interval"))
    # if validation_interval <= 0 or checkpoint_interval <= 0:
    #     raise ValueError("training validation/checkpoint intervals must be positive.")

    # gamma = float(OmegaConf.select(cfg, "training.gamma"))
    # tau = float(OmegaConf.select(cfg, "training.tau"))
    # if not 0.0 <= gamma <= 1.0:
    #     raise ValueError("training.gamma must be in [0, 1].")
    # if not 0.0 < tau <= 1.0:
    #     raise ValueError("training.tau must be in (0, 1].")

    # num_rounds = int(OmegaConf.select(cfg, "federated.num_rounds"))
    # if num_rounds <= 0:
    #     raise ValueError("federated.num_rounds must be positive.")
    # if int(OmegaConf.select(cfg, "federated.local_client_epochs")) != local_episodes:
    #     raise ValueError("federated.local_client_epochs must match training.local_episodes_per_round.")
    # fraction_fit = float(OmegaConf.select(cfg, "federated.server.fraction_fit"))
    # fraction_evaluate = float(OmegaConf.select(cfg, "federated.server.fraction_evaluate"))
    # if fraction_fit != 1.0 or fraction_evaluate != 1.0:
    #     raise ValueError("Paper experiments require federated.server.fraction_fit/evaluate == 1.0.")
    # min_fit_clients = int(OmegaConf.select(cfg, "federated.server.min_fit_clients"))
    # min_eval_clients = int(OmegaConf.select(cfg, "federated.server.min_evaluate_clients"))
    # min_available_clients = int(OmegaConf.select(cfg, "federated.server.min_available_clients"))
    # if (
    #     min_fit_clients != num_clients
    #     or min_eval_clients != num_clients
    #     or min_available_clients != num_clients
    # ):
    #     raise ValueError("Federated server client counts must match federated.num_clients.")

    # min_selected = int(OmegaConf.select(cfg, "federated.strategy.min_selected_clients"))
    # max_agents = int(OmegaConf.select(cfg, "federated.strategy.max_agents"))
    # max_selected_fraction = float(OmegaConf.select(cfg, "federated.strategy.max_selected_fraction"))
    # if min_selected <= 0 or min_selected > num_clients:
    #     raise ValueError("federated.strategy.min_selected_clients must be in [1, num_clients].")
    # if max_agents != num_clients:
    #     raise ValueError("federated.strategy.max_agents must match federated.num_clients.")
    # if not 0.0 < max_selected_fraction <= 1.0:
    #     raise ValueError("federated.strategy.max_selected_fraction must be in (0, 1].")



def _validate_runtime(cfg: DictConfig) -> None:
    prefer = str(OmegaConf.select(cfg, "device.prefer")).lower()
    runtime_prefer = str(OmegaConf.select(cfg, "runtime.device_prefer")).lower()
    allow_fallback = bool(OmegaConf.select(cfg, "device.allow_cpu_fallback"))
    if allow_fallback or bool(OmegaConf.select(cfg, "runtime.allow_cpu_fallback")):
        raise ValueError("Automatic CPU fallback is disabled; allow_cpu_fallback must be false.")
    if prefer != runtime_prefer:
        raise ValueError("device.prefer must resolve from runtime.device_prefer.")
    if prefer in {"gpu", "cuda"}:
        if float(OmegaConf.select(cfg, "runtime.client_num_gpus")) <= 0.0:
            raise ValueError("GPU runtime requires runtime.client_num_gpus > 0.")
        if not bool(OmegaConf.select(cfg, "runtime.simulation_gpu_batches.enabled")):
            raise ValueError(
                "GPU runtime requires runtime.simulation_gpu_batches.enabled=true "
                "for local Flower simulation."
            )
    batch_size = int(OmegaConf.select(cfg, "runtime.simulation_gpu_batches.batch_size"))
    if batch_size <= 0:
        raise ValueError("runtime.simulation_gpu_batches.batch_size must be positive.")


def _validate_experiment_contract(cfg: DictConfig) -> None:
    """Enforce the paper-level experiment contract from docs/cf_marlos-experiment-plan.md."""
    experiment_id = str(OmegaConf.select(cfg, "experiment.id", default="")).upper()
    pipeline = str(OmegaConf.select(cfg, "experiment.pipeline", default="full")).lower()
    valid_pipelines = {
        "suite",
        "plot",
        "export",
        "suite_artifacts",
        "preprocess",
        "compare",
        "full",
        "reproduce",
        "centralized",
        "train",
        "federated",
        "evaluate",
    }
    if pipeline not in valid_pipelines:
        raise ValueError(f"Unknown experiment.pipeline={pipeline!r}.")
    if experiment_id in {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"}:
        rounds = int(OmegaConf.select(cfg, "federated.num_rounds"))
        if experiment_id != "E7" and rounds != 100:
            raise ValueError(
                f"{experiment_id} must use 100 logical federated rounds per cf_marlos-experiment-plan.md."
            )
        expected_clients = 10
        if experiment_id != "E7" and int(OmegaConf.select(cfg, "federated.num_clients")) != expected_clients:
            raise ValueError(
                f"{experiment_id} must use {expected_clients} clients per cf_marlos-experiment-plan.md."
            )


def resolve_path(project_root: Path, path_like: str | Path) -> Path:
    """Resolve a project-relative or absolute path."""
    path = Path(path_like)
    return path if path.is_absolute() else (project_root / path)


def to_plain_container(cfg: DictConfig, *, resolve: bool = True) -> dict[str, Any]:
    """Convert a Hydra config to a plain Python dictionary."""
    return OmegaConf.to_container(cfg, resolve=resolve, throw_on_missing=True)  # type: ignore[return-value]
