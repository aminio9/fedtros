import hydra
from omegaconf import DictConfig

from src.evaluation.compare import compare_runs
from src.utils.entrypoints import prepare_run_context


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    context = prepare_run_context(cfg, script_name="compare_runs.py", with_device=False)
    compare_runs(cfg, project_root=context.project_root)


if __name__ == "__main__":
    main()
