import _bootstrap  # noqa: F401
from pathlib import Path
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.plotting import generate_plots
from src.tracking import attach_to_existing_run
from src.utils.config import validate_config


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    validate_config(cfg)
    run_dir = attach_to_existing_run(
        cfg,
        project_root=project_root,
        run_dir=cfg.run_dir,
        script_name="plot.py",
    )
    generate_plots(cfg, project_root=project_root, run_dir=run_dir)


if __name__ == "__main__":
    main()
