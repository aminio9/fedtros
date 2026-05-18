import hydra
from omegaconf import DictConfig

from src.federated import run_federated_client
from src.utils.entrypoints import prepare_run_context


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    context = prepare_run_context(
        cfg,
        script_name="federated_client.py",
        extra_required=("federated.client_id", "federated.client_data_path"),
        seed_offset=int(cfg.federated.client_id),
    )
    assert context.device is not None
    run_federated_client(cfg, project_root=context.project_root, device=context.device)


if __name__ == "__main__":
    main()
