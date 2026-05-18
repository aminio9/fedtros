import hydra
from omegaconf import DictConfig

from src.data import run_preprocessing
from src.evaluation import run_evaluation
from src.federated import run_federated_simulation
from src.utils.entrypoints import prepare_run_context


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    context = prepare_run_context(cfg, script_name="reproduce_experiment.py")
    assert context.device is not None
    assert context.tracker is not None
    run_preprocessing(cfg, project_root=context.project_root)
    run_federated_simulation(cfg, project_root=context.project_root, tracker=context.tracker)
    run_evaluation(
        cfg,
        project_root=context.project_root,
        device=context.device,
        tracker=context.tracker,
    )


if __name__ == "__main__":
    main()
