from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.plotting.plots import render_q1_dashboard, render_training_plots
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def generate_plots(
    cfg: DictConfig, *, project_root: Path, run_dir: str | Path | None = None
) -> list[Path]:
    source_run_dir = resolve_path(project_root, run_dir or cfg.run_dir)
    tracking_run_dir = resolve_path(project_root, cfg.tracking.run_dir)
    if (
        source_run_dir != tracking_run_dir
        and str(cfg.plotting.output_dir) == str(cfg.tracking.run_dir) + "/plots"
    ):
        output_dir = source_run_dir / "plots"
    else:
        output_dir = resolve_path(project_root, cfg.plotting.output_dir)
    formats = [str(fmt).lower() for fmt in cfg.plotting.formats]
    dpi = int(cfg.plotting.plot_dpi)
    generated = []
    generated.extend(render_q1_dashboard(source_run_dir, output_dir, formats, dpi))
    generated.extend(render_training_plots(source_run_dir, output_dir, formats, dpi))
    logger.info("Generated %d plot files under %s", len(generated), output_dir)
    return generated
