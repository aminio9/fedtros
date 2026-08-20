#!/usr/bin/env python3
"""Run known-only preprocessing without starting an experiment tracker."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.data import run_preprocessing
from src.infrastructure.logging import configure_logging, get_logger
from src.utils.config import validate_config
from src.utils.utils import set_seed

logger = get_logger("preprocess")


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging()
    validate_config(cfg)
    set_seed(int(cfg.seed), deterministic=True, benchmark=False, use_deterministic_algorithms=False)
    root = Path(get_original_cwd())
    metadata = run_preprocessing(cfg, project_root=root)
    logger.info("Preprocessing complete | dataset=%s | output=%s | samples=%s",
                cfg.dataset.name, cfg.dataset.preprocessing.output_dir,
                metadata.get("num_samples", metadata.get("total_samples", "unknown")))


if __name__ == "__main__":
    main()
