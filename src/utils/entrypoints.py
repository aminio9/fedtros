from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.tracking import LocalRunTracker, initialize_run
from src.utils.config import validate_config
from src.utils.utils import resolve_device_from_config, set_seed


@dataclass(frozen=True)
class ScriptContext:
    project_root: Path
    device: torch.device | None
    tracker: LocalRunTracker | None
    seed_settings: dict | None


def prepare_run_context(
    cfg: DictConfig,
    *,
    script_name: str,
    extra_required: Iterable[str] = (),
    seed_offset: int = 0,
    with_device: bool = True,
    with_tracker: bool = True,
) -> ScriptContext:
    """Common setup for Hydra scripts: validate config, tracking, device, and seed."""
    project_root = Path(get_original_cwd())
    validate_config(cfg, extra_required=extra_required)

    device = resolve_device_from_config(cfg) if with_device else None
    tracker = (
        initialize_run(cfg, project_root=project_root, script_name=script_name, device=device)
        if with_tracker
        else None
    )
    seed_settings = set_seed(
        int(cfg.seed) + int(seed_offset),
        deterministic=bool(cfg.device.deterministic),
        benchmark=bool(cfg.device.benchmark),
        use_deterministic_algorithms=bool(cfg.device.use_deterministic_algorithms),
    )
    logging.getLogger(__name__).info("Resolved device for %s: %s", script_name, device)
    if tracker is not None:
        tracker.metadata["seed_settings"] = seed_settings
        tracker.save_metadata()

    return ScriptContext(
        project_root=project_root,
        device=device,
        tracker=tracker,
        seed_settings=seed_settings,
    )
