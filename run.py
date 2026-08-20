"""Canonical root entrypoint for FedTROS-PR experiments (Item B15)."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from scripts.run import execute_run
from src.infrastructure.logging import configure_logging, get_logger

logger = get_logger("run")


@hydra.main(config_path="src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    execute_run(cfg, project_root)


if __name__ == "__main__":
    main()
