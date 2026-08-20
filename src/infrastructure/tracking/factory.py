"""Create the single configured interactive experiment tracker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.infrastructure.tracking.base import ExperimentTracker
from src.infrastructure.tracking.null_tracker import NullTracker

logger = logging.getLogger(__name__)


def _get(cfg: DictConfig | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, key, default=default)
        return default if value is None else value
    current: Any = cfg
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return default if current is None else current


def create_tracker(
    cfg: DictConfig | dict[str, Any],
    *,
    run_dir: Path,
    run_id: str,
    human_name: str | None = None,
    study_id: str | None = None,
    stage: str = "development",
    resume: bool = False,
) -> ExperimentTracker:
    """Instantiate W&B or a NullTracker.

    Local filesystem persistence is intentionally *not* performed here; it belongs to
    ``ResultStore``.  This prevents a second pseudo-tracker from becoming a competing
    source of truth.
    """
    backend = str(_get(cfg, "tracking.backend", "wandb")).lower()
    mode = str(_get(cfg, "tracking.mode", "online")).lower()

    if backend in {"none", "null", "disabled"} or mode == "disabled":
        return NullTracker(run_id=run_id)
    if backend != "wandb":
        raise ValueError(
            f"Unsupported tracking backend {backend!r}. Canonical FedTROS-PR supports 'wandb' "
            "or tracking.mode=disabled."
        )

    from src.infrastructure.tracking.wandb_tracker import WandBTracker

    dataset = str(_get(cfg, "dataset.name", "bnat"))
    method = str(_get(cfg, "experiment.method", "FedTROS-PR"))
    seed = int(_get(cfg, "seed", 42))
    alpha = _get(cfg, "dataset.preprocessing.alpha", None)
    clients = _get(cfg, "federated.num_clients", None)
    unknowns = list(_get(cfg, "dataset.preprocessing.unknown_labels", []) or [])
    tags = list(_get(cfg, "tracking.tags", []) or [])
    tags.extend([stage, dataset.lower(), str(method).lower().replace("-", "_")])
    if alpha is not None:
        tags.append(f"alpha_{str(alpha).replace('.', 'p')}")
    if clients is not None:
        tags.append(f"clients_{int(clients)}")
    if unknowns:
        tags.append("open_set")
    else:
        tags.append("closed_set")
    # stable, ordered de-duplication
    tags = list(dict.fromkeys(str(tag) for tag in tags if str(tag)))

    resume_policy: str | bool | None
    if resume:
        resume_policy = "allow"
    else:
        resume_policy = str(_get(cfg, "tracking.resume", "never"))

    return WandBTracker(
        run_dir=run_dir,
        canonical_run_id=run_id,
        project=str(_get(cfg, "tracking.project", "FedTROS-PR")),
        entity=_get(cfg, "tracking.entity", None),
        name=human_name or run_id,
        group=str(_get(cfg, "tracking.group", study_id or "general")),
        job_type=str(_get(cfg, "tracking.job_type", stage)),
        tags=tags,
        mode=mode,
        resume=resume_policy,
        notes=_get(cfg, "tracking.notes", None),
        save_code=bool(_get(cfg, "tracking.save_code", False)),
    )
