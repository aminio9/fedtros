import logging
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

try:
    from src.utils import setup_logging
    from src.server import plot_reward_history, run_server
except ImportError as e:
    print(f"Error: Could not import from 'src'. {e}", file=sys.stderr)
    print("Please run this script from the project root directory (dcids_federated/).", file=sys.stderr)
    sys.exit(1)


@hydra.main(config_path="conf", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for the Federated Server."""

    project_root = Path(get_original_cwd())
    log_file = project_root / "logs" / "server.log"
    log_level = str(cfg.get("log_level", "INFO")).upper()
    setup_logging(log_file_path=str(log_file), log_level=log_level)

    logger = logging.getLogger("Run Server")
    logger.info("--- Starting Federated Server ---")
    logger.info(f"Full Configuration:\n{OmegaConf.to_yaml(cfg)}")

    try:
        run_server(cfg)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down gracefully...")
    finally:
        plot_reward_history(cfg)
        logger.info("--- Server finished ---")


if __name__ == "__main__":
    main()
