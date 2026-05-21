from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import DictConfig

from src.plotting.plots import render_required_plots, render_training_plots
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def generate_plots(
    cfg: DictConfig, *, project_root: Path, run_dir: str | Path | None = None
) -> list[Path]:
    source_run_dir = resolve_path(project_root, run_dir or cfg.run_dir)
    tracking_run_dir = resolve_path(project_root, cfg.tracking.run_dir)
    preprocess_output_dir = None
    try:
        from omegaconf import OmegaConf

        preprocess_path = OmegaConf.select(cfg, "dataset.preprocessing.output_dir", default=None)
        if preprocess_path:
            preprocess_output_dir = resolve_path(project_root, preprocess_path)
    except Exception:
        preprocess_output_dir = None
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
    generated.extend(
        render_required_plots(
            source_run_dir,
            output_dir,
            formats,
            dpi,
            preprocess_dir=preprocess_output_dir,
        )
    )
    generated.extend(render_training_plots(source_run_dir, output_dir, formats, dpi))
    manifest = {
        "source_run_dir": str(source_run_dir),
        "output_dir": str(output_dir),
        "formats": formats,
        "dpi": dpi,
        "files": [str(path.relative_to(output_dir)) for path in generated],
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Generated %d plot files under %s", len(generated), output_dir)
    return generated
