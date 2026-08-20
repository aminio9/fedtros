#!/usr/bin/env python3
"""Evaluate a FedTROS-PR checkpoint as a fully traceable evaluation run.

The command uses the same run lifecycle, ResultStore, manifests, operational logging,
and W&B tracker abstraction as training.  It never attaches to the legacy local tracker.

Example:
  python scripts/evaluate.py checkpointing.resume_from=/path/to/checkpoint.pt \
      evaluation.mode=open_set tracking.mode=offline
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from scripts.run import execute_run


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.update(cfg, "experiment.pipeline", "evaluate", force_add=True)
    OmegaConf.update(cfg, "experiment.variant", "evaluation_only", force_add=True)
    execute_run(cfg, Path(get_original_cwd()), resume=False)


if __name__ == "__main__":
    main()
