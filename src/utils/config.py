from __future__ import annotations

import json
import logging

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# ``main`` is the canonical publication-stage alias used by the current
# runner. ``paper_final`` and ``reproduction`` remain supported for older
# checkouts and exact reruns.
PUBLICATION_STAGES = {"main", "paper_final", "reproduction"}


def _canonical_dataset_id(value: str) -> str:
    compact = "".join(character for character in str(value).lower() if character.isalnum())
    aliases = {
        "bnat": "bnat",
        "btat": "btat",
        "cicids2017": "cicids2017",
        "toniot": "toniot",
    }
    return aliases.get(compact, compact)



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
    strict_num_classes: bool = True,
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

    if "feature_dim" not in metadata or "num_classes" not in metadata:
        raise KeyError("preprocess_metadata.json must contain feature_dim and num_classes.")

    actual_feature_dim = int(metadata["feature_dim"])
    actual_num_classes = int(metadata["num_classes"])
    configured_feature_dim = int(OmegaConf.select(cfg, "model.feature_dim"))
    configured_num_classes = int(OmegaConf.select(cfg, "model.num_classes"))

    if configured_feature_dim != actual_feature_dim:
        logger.warning(
            "Model feature_dim synchronized from preprocessing metadata | config=%d processed=%d known_labels=%s",
            configured_feature_dim,
            actual_feature_dim,
            metadata.get("known_labels"),
        )
        _set_cfg_value(cfg, "model.feature_dim", actual_feature_dim)
        if OmegaConf.select(cfg, "model.transformer.input_dim", default=None) is not None:
            _set_cfg_value(cfg, "model.transformer.input_dim", actual_feature_dim)

    if configured_num_classes != actual_num_classes:
        message = (
            "model.num_classes does not match preprocessing metadata: "
            f"config={configured_num_classes}, processed={actual_num_classes}, "
            f"known_labels={metadata.get('known_labels')}"
        )
        if strict_num_classes:
            raise ValueError(message)
        logger.warning("%s; synchronizing config to processed dataset.", message)
        _set_cfg_value(cfg, "model.num_classes", actual_num_classes)

    return metadata


REQUIRED_CONFIG_KEYS = (
    "seed",
    "dataset.name",
    "dataset.preprocessing.raw_file",
    "dataset.preprocessing.output_dir",
    "dataset.preprocessing.known_labels",
    "model.num_classes",
    "training.batch_size",
    "training.local_epochs",
    "federated.num_clients",
    "federated.num_rounds",
    "federated.strategy.name",
    "open_set.enabled",
    "tracking.run_dir",
    "checkpointing.latest_checkpoint_path",
    "logging.level",
)


def _select_str(cfg: DictConfig, key: str, default: str = "") -> str:
    value = OmegaConf.select(cfg, key, default=default)
    return str(value or default)



def validate_config(cfg: DictConfig, extra_required: Iterable[str] = ()) -> None:
    """Fail early when a required Hydra key is missing or unresolved."""
    _apply_experiment_protocol(cfg)
    required = (
        "seed",
        "dataset.name",
        "dataset.preprocessing.raw_file",
        "dataset.preprocessing.known_labels",
        "federated.num_clients",
        "dataset.preprocessing.num_clients",
        "model.num_classes",
    ) + tuple(extra_required)
    missing = [
        key
        for key in required
        if OmegaConf.select(cfg, key, default=None) in {None, "???"}
    ]
    if missing:
        raise ValueError("Missing required Hydra values: " + ", ".join(missing))

    _validate_runtime(cfg)
    num_clients = int(OmegaConf.select(cfg, "federated.num_clients"))
    preprocessing_num_clients = int(OmegaConf.select(cfg, "dataset.preprocessing.num_clients"))
    if preprocessing_num_clients != num_clients:
        raise ValueError(
            "dataset.preprocessing.num_clients must match federated.num_clients. "
            "Override federated.num_clients to change both values."
        )
    strategy_name = _select_str(cfg, "federated.strategy.name").lower()
    aggregation_strategy = _select_str(
        cfg, "federated.strategy.aggregation_strategy"
    ).lower()
    if strategy_name == "fedprox":
        proximal_mu = float(
            OmegaConf.select(cfg, "federated.server.proximal_mu", default=0.0)
        )
        if proximal_mu <= 0.0:
            raise ValueError(
                "FedProx requires federated.server.proximal_mu > 0; "
                "a zero value is equivalent to FedAvg."
            )
        if aggregation_strategy != "fedprox":
            raise ValueError(
                "FedProx requires federated.strategy.aggregation_strategy=fedprox."
            )
    _validate_external_experiment_contract(cfg)
    _validate_experiment_contract(cfg)


def _study_prefix(cfg: DictConfig) -> str:
    value = _select_str(cfg, "experiment.id").upper()
    return value.split("-", 1)[0].split("_", 1)[0]


def _apply_experiment_protocol(cfg: DictConfig) -> None:
    """Enforce canonical open/closed-set selector values without legacy aliases."""
    study = _study_prefix(cfg)
    unknown_labels = list(OmegaConf.select(cfg, "dataset.preprocessing.unknown_labels", default=[]) or [])
    open_mode = bool(unknown_labels) or _select_str(cfg, "evaluation.mode").lower() == "open_set"

    if study == "E1":
        open_mode = False
        _set_cfg_value(cfg, "dataset.preprocessing.protocol", "closed_set")
        _set_cfg_value(cfg, "dataset.preprocessing.unknown_labels", [])
        source_labels = list(OmegaConf.select(cfg, "dataset.source_labels", default=[]) or [])
        if source_labels:
            _set_cfg_value(cfg, "dataset.preprocessing.known_labels", source_labels)
    elif open_mode:
        _set_cfg_value(cfg, "dataset.preprocessing.protocol", "open_set")

    _set_cfg_value(cfg, "open_set.enabled", bool(open_mode))
    requested_detector = _select_str(cfg, "open_set.detector", "multicenter_conformal").lower()
    # A4 is the only predeclared study allowed to select a non-canonical detector.
    # Its variants keep the training objective canonical and change only the
    # post-federation detector on a matched seeded representation.
    detector = requested_detector if study == "A4" and requested_detector == "prototype_rank" else (
        "multicenter_conformal" if open_mode else "disabled"
    )
    _set_cfg_value(cfg, "open_set.method", detector)
    _set_cfg_value(cfg, "open_set.detector", detector)
    _set_cfg_value(cfg, "open_set.prototype_rank.enabled", bool(open_mode and detector == "prototype_rank"))
    if OmegaConf.select(cfg, "open_set.prototype_rank.proser.enabled", default=None) is not None:
        _set_cfg_value(cfg, "open_set.prototype_rank.proser.enabled", False)
    if OmegaConf.select(cfg, "open_set.prototype_rank.energy.train_margin_enabled", default=None) is not None:
        _set_cfg_value(cfg, "open_set.prototype_rank.energy.train_margin_enabled", False)


def _validate_external_experiment_contract(cfg: DictConfig) -> None:
    """Validate dataset-wise external-validation invariants without single-seed restrictions."""
    if _study_prefix(cfg) != "E5":
        return
    registry_name = _canonical_dataset_id(_select_str(cfg, "dataset.registry_name"))
    dataset_name = _canonical_dataset_id(_select_str(cfg, "dataset.name"))
    if registry_name and registry_name not in {"bnat", "btat", "toniot", "cicids2017"}:
        raise ValueError(f"Unsupported E5 dataset registry: {registry_name}")
    if dataset_name not in {"bnat", "btat", "toniot", "cicids2017"}:
        raise ValueError(f"E5 dataset-wise validation received unsupported dataset={dataset_name!r}")
    stage = _select_str(cfg, "stage", "development").lower()
    if stage in PUBLICATION_STAGES and int(OmegaConf.select(cfg, "federated.num_clients", default=10)) != 10:
        raise ValueError("E5 requires 10 clients in the canonical dataset-wise protocol.")
    if abs(float(OmegaConf.select(cfg, "dataset.preprocessing.alpha", default=0.5)) - 0.5) > 1e-12:
        raise ValueError("E5 requires Dirichlet alpha=0.5.")
    if bool(OmegaConf.select(cfg, "dataset.preprocessing.iid", default=False)):
        raise ValueError("E5 requires non-IID partitioning.")
    unknown = list(OmegaConf.select(cfg, "dataset.preprocessing.unknown_labels", default=[]) or [])
    if not unknown:
        raise ValueError("E5 requires a predeclared held-out unknown protocol for each dataset.")


def _validate_runtime(cfg: DictConfig) -> None:
    prefer = str(OmegaConf.select(cfg, "device.prefer")).lower()
    runtime_prefer = str(OmegaConf.select(cfg, "runtime.device_prefer")).lower()
    allow_fallback = bool(OmegaConf.select(cfg, "device.allow_cpu_fallback"))
    stage = _select_str(cfg, "stage", "development").lower()
    if (allow_fallback or bool(OmegaConf.select(cfg, "runtime.allow_cpu_fallback"))) and stage in PUBLICATION_STAGES:
        raise ValueError("Automatic CPU fallback is disabled; allow_cpu_fallback must be false for publication stages.")
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
    """Enforce immutable headline-study constraints at publication stages.

    Development/smoke/ablation runs intentionally use smaller budgets, so the hard
    horizon/client constraints apply only to ``paper_final`` and ``reproduction``.
    Study YAML files remain the declarative source of the full matrix; this validator
    is the last line of defense against a manual CLI override quietly changing a
    publication run.
    """
    study = _study_prefix(cfg)
    stage = _select_str(cfg, "stage", "development").lower()
    pipeline = str(OmegaConf.select(cfg, "experiment.pipeline", default="full")).lower()
    valid_pipelines = {
        "full",
        "all",
        "smoke",
        "reproduce",
        "centralized",
        "federated",
        "evaluate",
    }
    if pipeline not in valid_pipelines:
        raise ValueError(f"Unknown experiment.pipeline={pipeline!r}.")
    canonical_studies = {
        "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
        "A1", "A2", "A3", "A4", "A5", "S1",
    }
    if stage in PUBLICATION_STAGES and study not in canonical_studies:
        raise ValueError(
            f"stage={stage!r} requires a canonical E0-E8/A1-A5/S1 study ID; got {study!r}. "
            "Use the declarative study runner instead of publishing an ad-hoc baseline config."
        )

    if stage not in PUBLICATION_STAGES:
        return

    rounds = int(OmegaConf.select(cfg, "federated.num_rounds"))
    if rounds != 100:
        raise ValueError(f"{study} {stage} runs must use the predeclared 100-round horizon.")

    clients = int(OmegaConf.select(cfg, "federated.num_clients"))
    if study == "E6":
        if clients not in {10, 50, 100}:
            raise ValueError("E6 fixed-data scalability requires clients in {10,50,100}.")
    elif clients != 10:
        raise ValueError(f"{study} {stage} runs require 10 clients in the canonical protocol.")

    # All non-scalability publication studies, including ablations and
    # sensitivity analyses, use the same ten-client federated topology.  The
    # E6 exception above is the only declared client-count variation.

    iid = bool(OmegaConf.select(cfg, "dataset.preprocessing.iid", default=False))
    if study in {"E1", "E2"} and not iid:
        raise ValueError(f"{study} is an IID publication study.")
    if study in {"E3", "E4", "E5", "E6", "E7", "E8"} and iid:
        raise ValueError(f"{study} is a non-IID publication study.")

    alpha = float(OmegaConf.select(cfg, "dataset.preprocessing.alpha", default=0.5))
    if study in {"E3", "E4"} and not any(abs(alpha - x) <= 1e-12 for x in (1.0, 0.5, 0.1)):
        raise ValueError(f"{study} requires alpha in {{1.0,0.5,0.1}}.")
    if study in {"E5", "E6", "E7", "E8"} and abs(alpha - 0.5) > 1e-12:
        raise ValueError(f"{study} requires canonical alpha=0.5.")

    if study == "E8":
        unknown = list(OmegaConf.select(cfg, "dataset.preprocessing.unknown_labels", default=[]) or [])
        known = list(OmegaConf.select(cfg, "dataset.preprocessing.known_labels", default=[]) or [])
        if len(unknown) != 1 or str(unknown[0]) not in {"BP", "DoS", "MitM", "FoT"}:
            raise ValueError("E8 requires exactly one held-out attack from {BP,DoS,MitM,FoT}.")
        if "Normal" not in known or str(unknown[0]) in {str(x) for x in known}:
            raise ValueError("E8 requires Normal to remain known and the held-out attack to be absent from known labels.")


def resolve_path(project_root: Path, path_like: str | Path) -> Path:
    """Resolve a project-relative or absolute path."""
    path = Path(path_like)
    return path if path.is_absolute() else (project_root / path)


def to_plain_container(cfg: DictConfig, *, resolve: bool = True) -> dict[str, Any]:
    """Convert a Hydra config to a plain Python dictionary."""
    return OmegaConf.to_container(cfg, resolve=resolve, throw_on_missing=True)  # type: ignore[return-value]
