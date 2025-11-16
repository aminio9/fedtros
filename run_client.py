import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import flwr as fl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

try:
    from src.utils import resolve_device_from_config, setup_logging
    from src.client import FlowerClient
    from src.exceptions import ConfigMismatchError
except ImportError as err:
    print(f"Error: Could not import from 'src'. {err}", file=sys.stderr)
    print("Please run this script from the project root directory (dcids_federated/).", file=sys.stderr)
    sys.exit(1)


def resolve_project_root() -> Path:
    """Return the project root path regardless of Hydra initialization."""
    return Path(__file__).resolve().parent


def load_config(overrides: List[str]) -> DictConfig:
    """Load Hydra/OmegaConf configuration with optional overrides."""
    config_dir = resolve_project_root() / "conf"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config_fl", overrides=overrides)
    return cfg


def run_client(cfg: DictConfig, cid: str, data_path: Path, device: Optional[torch.device]) -> None:
    """Instantiate and start a Flower client."""
    try:
        client = FlowerClient(cid=cid, cfg=cfg, data_path=str(data_path), device=device)
        fl.client.start_client(server_address=cfg.server.address, client=client.to_client())
    except ConfigMismatchError as exc:
        logging.getLogger(__name__).critical("Configuration Error: %s", exc, exc_info=True)
        print(f"\nFATAL: Configuration mismatch. {exc}", file=sys.stderr)
        print("Please check your 'conf/config_fl.yaml' and 'data/processed' files.", file=sys.stderr)
        sys.exit(1)
    except Exception:
        logging.getLogger(__name__).critical("Client %s crashed.", cid, exc_info=True)
        raise


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Flower Client for DC-IDS", add_help=True)
    parser.add_argument("--cid", required=True, help="Client ID (e.g., '1', '2', '3')")
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the client's local data file (e.g., 'data/processed/client_1.pt')",
    )
    return parser.parse_known_args()


def main() -> None:
    client_args, hydra_overrides = parse_args()
    cfg = load_config(hydra_overrides)

    project_root = resolve_project_root()
    data_path = (project_root / client_args.data_path).resolve()
    if not data_path.exists():
        print(f"Error: Data path not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    log_file = project_root / "logs" / f"client_{client_args.cid}.log"
    log_level = str(cfg.get("log_level", "INFO")).upper()
    setup_logging(log_file_path=str(log_file), log_level=log_level)

    logger = logging.getLogger(__name__)
    device = resolve_device_from_config(cfg)
    logger.info("--- Starting Client %s ---", client_args.cid)
    logger.info("Data Path: %s", data_path)
    logger.info("Base Config Loaded. Server: %s", cfg.server.address)
    logger.info("Resolved device: %s", device)

    run_client(cfg, client_args.cid, data_path, device)

    logger.info("--- Client %s finished ---", client_args.cid)


if __name__ == "__main__":
    main()
