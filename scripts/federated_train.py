import hydra
from omegaconf import DictConfig

from src.federated import run_federated_simulation
from src.utils.entrypoints import prepare_run_context


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    context = prepare_run_context(cfg, script_name="federated_train.py")
    assert context.tracker is not None
    run_federated_simulation(
        cfg,
        project_root=context.project_root,
        tracker=context.tracker,
    )


if __name__ == "__main__":
    main()
