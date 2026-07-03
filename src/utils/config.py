from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.openset.evt import resolve_tail_fraction

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
    "open_set.evt.target_known_fpr",
    "plotting.required_plots",
    "tracking.run_dir",
    "checkpointing.latest_checkpoint_path",
    "checkpointing.last_model_path",
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

    _validate_runtime(cfg)
    _validate_experiment_contract(cfg)
    _validate_named_config_contracts(cfg)
    _validate_architecture_contracts(cfg)

    known_labels = OmegaConf.select(cfg, "dataset.preprocessing.known_labels")
    if not known_labels:
        raise ValueError("dataset.preprocessing.known_labels must contain at least one label.")
    num_actions = int(OmegaConf.select(cfg, "model.num_actions"))
    if num_actions != len(known_labels):
        raise ValueError(
            "model.num_actions must match len(dataset.preprocessing.known_labels): "
            f"{num_actions} != {len(known_labels)}."
        )

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

    training_epochs = int(OmegaConf.select(cfg, "training.epochs"))
    local_episodes = int(OmegaConf.select(cfg, "training.local_episodes_per_round"))
    steps_per_episode = int(OmegaConf.select(cfg, "training.steps_per_episode"))
    replay_buffer_size = int(OmegaConf.select(cfg, "training.replay_buffer_size"))
    if min(training_epochs, local_episodes, steps_per_episode, replay_buffer_size) <= 0:
        raise ValueError(
            "training.epochs, local_episodes_per_round, steps_per_episode, "
            "and replay_buffer_size must be positive."
        )
    if min_buffer_size > replay_buffer_size:
        raise ValueError("training.min_buffer_size cannot exceed training.replay_buffer_size.")

    validation_interval = int(OmegaConf.select(cfg, "training.validation_interval"))
    checkpoint_interval = int(OmegaConf.select(cfg, "training.checkpoint_interval"))
    if validation_interval <= 0 or checkpoint_interval <= 0:
        raise ValueError("training validation/checkpoint intervals must be positive.")

    gamma = float(OmegaConf.select(cfg, "training.gamma"))
    tau = float(OmegaConf.select(cfg, "training.tau"))
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("training.gamma must be in [0, 1].")
    if not 0.0 < tau <= 1.0:
        raise ValueError("training.tau must be in (0, 1].")

    num_rounds = int(OmegaConf.select(cfg, "federated.num_rounds"))
    if num_rounds <= 0:
        raise ValueError("federated.num_rounds must be positive.")
    if int(OmegaConf.select(cfg, "federated.local_client_epochs")) != local_episodes:
        raise ValueError("federated.local_client_epochs must match training.local_episodes_per_round.")
    fraction_fit = float(OmegaConf.select(cfg, "federated.server.fraction_fit"))
    fraction_evaluate = float(OmegaConf.select(cfg, "federated.server.fraction_evaluate"))
    if fraction_fit != 1.0 or fraction_evaluate != 1.0:
        raise ValueError("Paper experiments require federated.server.fraction_fit/evaluate == 1.0.")
    min_fit_clients = int(OmegaConf.select(cfg, "federated.server.min_fit_clients"))
    min_eval_clients = int(OmegaConf.select(cfg, "federated.server.min_evaluate_clients"))
    min_available_clients = int(OmegaConf.select(cfg, "federated.server.min_available_clients"))
    if (
        min_fit_clients != num_clients
        or min_eval_clients != num_clients
        or min_available_clients != num_clients
    ):
        raise ValueError("Federated server client counts must match federated.num_clients.")

    min_selected = int(OmegaConf.select(cfg, "federated.strategy.min_selected_clients"))
    max_agents = int(OmegaConf.select(cfg, "federated.strategy.max_agents"))
    max_selected_fraction = float(OmegaConf.select(cfg, "federated.strategy.max_selected_fraction"))
    if min_selected <= 0 or min_selected > num_clients:
        raise ValueError("federated.strategy.min_selected_clients must be in [1, num_clients].")
    if max_agents != num_clients:
        raise ValueError("federated.strategy.max_agents must match federated.num_clients.")
    if not 0.0 < max_selected_fraction <= 1.0:
        raise ValueError("federated.strategy.max_selected_fraction must be in (0, 1].")

    resolve_tail_fraction(OmegaConf.select(cfg, "open_set.evt"))
    target_known_fpr = float(OmegaConf.select(cfg, "open_set.evt.target_known_fpr"))
    target_unknown_tpr = OmegaConf.select(cfg, "open_set.evt.target_unknown_tpr", default=None)
    decision_threshold = float(
        OmegaConf.select(cfg, "open_set.evt.decision_threshold", default=0.5)
    )
    if not 0.0 <= target_known_fpr < 1.0:
        raise ValueError("open_set.evt.target_known_fpr must be in [0, 1).")
    if target_unknown_tpr is not None and not 0.0 < float(target_unknown_tpr) <= 1.0:
        raise ValueError("open_set.evt.target_unknown_tpr must be in (0, 1] when set.")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("open_set.evt.decision_threshold must be in [0, 1].")
    fixed_threshold = float(OmegaConf.select(cfg, "open_set.evt.fixed_threshold", default=0.5))
    if not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("open_set.evt.fixed_threshold must be in [0, 1].")
    threshold_mode = str(
        OmegaConf.select(cfg, "open_set.evt.threshold_mode", default="validation_known_fpr")
    ).lower()
    if threshold_mode not in {"validation_known_fpr", "fixed"}:
        raise ValueError("open_set.evt.threshold_mode must be validation_known_fpr or fixed.")
    score_direction = str(
        OmegaConf.select(cfg, "open_set.evt.score_direction", default="higher_unknown")
    ).lower()
    if score_direction != "higher_unknown":
        raise ValueError("Only score_direction=higher_unknown is currently supported.")
    calibration_protocol = str(
        OmegaConf.select(cfg, "open_set.evt.calibration_protocol", default="validation_only")
    ).lower()
    if calibration_protocol != "validation_only":
        raise ValueError("Only calibration_protocol=validation_only is currently supported.")
    _validate_loss_and_reward_config(cfg)


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


def _validate_loss_and_reward_config(cfg: DictConfig) -> None:
    non_negative_paths = (
        "training.loss_weights.prior_kl",
        "training.loss_weights.q_td",
        "training.loss_weights.bandit_q",
        "training.loss_weights.classification",
        "training.loss_weights.generator_reconstruction",
        "training.loss_weights.proximal",
        "training.auxiliary_losses.supervised_contrastive_lambda",
        "training.auxiliary_losses.center_loss_lambda",
        "training.classification_loss.focal_gamma",
        "training.kl.free_nats",
    )
    for path in non_negative_paths:
        value = float(OmegaConf.select(cfg, path, default=0.0))
        if value < 0.0:
            raise ValueError(f"{path} must be non-negative.")

    correct_reward = float(OmegaConf.select(cfg, "training.reward.correct", default=1.0))
    incorrect_reward = float(OmegaConf.select(cfg, "training.reward.incorrect", default=-1.0))
    class_balance_power = float(
        OmegaConf.select(cfg, "training.reward.class_balance_power", default=1.0)
    )
    if correct_reward <= 0.0:
        raise ValueError("training.reward.correct must be positive.")
    if incorrect_reward >= 0.0:
        raise ValueError("training.reward.incorrect must be negative.")
    if class_balance_power < 0.0:
        raise ValueError("training.reward.class_balance_power must be non-negative.")

    supcon_temperature = float(
        OmegaConf.select(
            cfg,
            "training.auxiliary_losses.supervised_contrastive_temperature",
            default=0.1,
        )
    )
    if supcon_temperature <= 0.0:
        raise ValueError("training.auxiliary_losses.supervised_contrastive_temperature must be positive.")

    classification_loss = str(
        OmegaConf.select(cfg, "training.classification_loss.name", default="focal")
    ).lower()
    if classification_loss not in {"focal", "cross_entropy"}:
        raise ValueError("training.classification_loss.name must be focal or cross_entropy.")

    kl_warmup_steps = int(OmegaConf.select(cfg, "training.kl.warmup_steps", default=0))
    if kl_warmup_steps < 0:
        raise ValueError("training.kl.warmup_steps must be non-negative.")

    mask_value = float(OmegaConf.select(cfg, "training.missing_class_gradient.mask_value", default=-20.0))
    if mask_value >= 0.0:
        raise ValueError("training.missing_class_gradient.mask_value must be negative.")

    imbalance_enabled = bool(OmegaConf.select(cfg, "training.imbalance.enabled", default=False))
    if imbalance_enabled:
        mode = str(
            OmegaConf.select(cfg, "training.imbalance.weight_mode", default="inverse_frequency")
        ).lower()
        if mode not in {
            "none",
            "uniform",
            "inverse",
            "inverse_frequency",
            "effective",
            "effective_number",
        }:
            raise ValueError("training.imbalance.weight_mode is not supported.")
        beta = float(OmegaConf.select(cfg, "training.imbalance.effective_number_beta", default=0.999))
        min_weight = float(OmegaConf.select(cfg, "training.imbalance.min_weight", default=0.2))
        max_weight = float(OmegaConf.select(cfg, "training.imbalance.max_weight", default=5.0))
        if not 0.0 <= beta < 1.0:
            raise ValueError("training.imbalance.effective_number_beta must be in [0, 1).")
        if min_weight <= 0.0 or max_weight <= 0.0 or min_weight > max_weight:
            raise ValueError("training.imbalance min/max weights must satisfy 0 < min <= max.")


def _validate_named_config_contracts(cfg: DictConfig) -> None:
    model_name = str(OmegaConf.select(cfg, "model.name", default="")).lower()
    supported_models = {
        "openset_qchain",
    }
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model.name={model_name!r}.")

    strategy_name = str(OmegaConf.select(cfg, "federated.strategy.name", default="")).lower()
    supported_strategies = {
        "centralized",
        "fedavg",
        "fedprox",
        "fmrl_ava",
        "fedmade",
        "class_aware",
        "class_aware_dynamic",
    }
    if strategy_name not in supported_strategies:
        raise ValueError(f"Unsupported federated.strategy.name={strategy_name!r}.")

    aggregation_strategy = str(
        OmegaConf.select(cfg, "federated.strategy.aggregation_strategy", default=strategy_name)
    ).lower()
    if aggregation_strategy not in supported_strategies:
        raise ValueError(
            f"Unsupported federated.strategy.aggregation_strategy={aggregation_strategy!r}."
        )

    open_set_name = str(OmegaConf.select(cfg, "open_set.name", default="evt")).lower()
    supported_open_set = {
        "evt",
        "openmax_evt",
        "msp",
        "energy",
        "prototype",
        "prototype_distance",
        "mahalanobis",
        "no_rejection",
    }
    if open_set_name not in supported_open_set:
        raise ValueError(f"Unsupported open_set.name={open_set_name!r}.")

    scorer_name = str(OmegaConf.select(cfg, "open_set.scorer.name", default="evt_reconstruction"))
    supported_scorers = {
        "evt_reconstruction",
        "openmax_evt_reconstruction",
        "msp",
        "energy",
        "prototype_distance",
        "mahalanobis_distance",
        "no_rejection",
    }
    if scorer_name.lower() not in supported_scorers:
        raise ValueError(f"Unsupported open_set.scorer.name={scorer_name!r}.")

    monitor_metric = str(OmegaConf.select(cfg, "checkpointing.monitor_metric", default=""))
    if monitor_metric != "combined_validation_score" and not monitor_metric.startswith(
        ("val/", "validation/")
    ):
        raise ValueError(
            "checkpointing.monitor_metric must be validation-prefixed or combined_validation_score."
        )
    monitor_mode = str(OmegaConf.select(cfg, "checkpointing.monitor_mode", default="max")).lower()
    if monitor_mode not in {"min", "max"}:
        raise ValueError("checkpointing.monitor_mode must be 'min' or 'max'.")

    optimizer_name = str(OmegaConf.select(cfg, "optimizer.name", default="adamw")).lower()
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer.name must be adam or adamw.")


def _validate_architecture_contracts(cfg: DictConfig) -> None:
    """Reject combinations that the current run.py pipelines cannot execute."""
    pipeline = str(OmegaConf.select(cfg, "experiment.pipeline", default="full")).lower()
    model_name = str(OmegaConf.select(cfg, "model.name", default="")).lower()
    model_family = str(
        OmegaConf.select(cfg, "model.family", default=_infer_model_family(model_name))
    ).lower()
    open_set_name = str(OmegaConf.select(cfg, "open_set.name", default="evt")).lower()
    scorer_name = str(
        OmegaConf.select(cfg, "open_set.scorer.name", default="evt_reconstruction")
    ).lower()
    strategy_name = str(OmegaConf.select(cfg, "federated.strategy.name", default="")).lower()

    agent_pipelines = {"full", "reproduce", "centralized", "train", "federated", "evaluate"}
    open_set_eval_pipelines = {"full", "reproduce", "centralized", "evaluate"}
    first_class_open_set = {"evt"}
    utility_open_set = {
        "msp",
        "energy",
        "prototype",
        "prototype_distance",
        "mahalanobis",
        "no_rejection",
    }

    if pipeline in agent_pipelines and model_family != "cvae_dqn":
        raise ValueError(
            f"model={model_name!r} is not supported by run.py training/evaluation "
            "pipelines. Supported models are CVAE-DQN models built with "
            "OpenSetQChainModelFactory."
        )

    if pipeline in open_set_eval_pipelines and open_set_name in utility_open_set:
        raise ValueError(
            f"open_set={open_set_name!r} configures a standalone scorer utility. "
            "run.py evaluation currently supports open_set=evt only; MSP, energy, "
            "prototype, Mahalanobis, and no-rejection scorers are tested utilities but "
            "not first-class experiment pipelines."
        )

    if pipeline in open_set_eval_pipelines and open_set_name == "openmax_evt":
        raise ValueError(
            "open_set=openmax_evt is a scaffold alias and is not a real OpenMax "
            "implementation in run.py; use open_set=evt for the implemented EVT path."
        )

    if scorer_name in {"openmax_evt_reconstruction"}:
        raise ValueError(
            "open_set.scorer.name=openmax_evt_reconstruction is not implemented as "
            "OpenMax; use evt_reconstruction or add a real OpenMax evaluator."
        )

    if (
        pipeline in open_set_eval_pipelines
        and bool(OmegaConf.select(cfg, "open_set.evt.enabled", default=False))
        and open_set_name not in first_class_open_set
    ):
        raise ValueError("EVT evaluation must use open_set=evt.")

    if strategy_name in {"fedmade", "class_aware", "class_aware_dynamic"} and pipeline not in {
        "full",
        "reproduce",
        "federated",
        "suite",
    }:
        raise ValueError(
            f"method={strategy_name!r} requires a federated pipeline; "
            "use experiment.pipeline=full or federated."
        )


def _infer_model_family(model_name: str) -> str:
    if model_name == "openset_qchain":
        return "cvae_dqn"
    return "unknown"


def _validate_experiment_contract(cfg: DictConfig) -> None:
    """Enforce the paper-level experiment contract from docs/cf_marlos-experiment-plan.md."""
    experiment_id = str(OmegaConf.select(cfg, "experiment.id", default="")).upper()
    pipeline = str(OmegaConf.select(cfg, "experiment.pipeline", default="full")).lower()
    valid_pipelines = {
        "smoke",
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
