import hydra
from omegaconf import DictConfig

from src.training import run_smoke_test
from src.utils.entrypoints import prepare_run_context


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    context = prepare_run_context(cfg, script_name="smoke_test.py")
    assert context.device is not None
    assert context.tracker is not None
    run_smoke_test(
        cfg,
        project_root=context.project_root,
        device=context.device,
        tracker=context.tracker,
    )


if __name__ == "__main__":
    main()
