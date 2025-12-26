import argparse
import logging
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import flwr as fl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

# Ensure imports work regardless of where the script is run
for candidate in (PROJECT_ROOT, SRC_PATH):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

# Import Project Modules
try:
    from src.utils import resolve_device_from_config, setup_logging
    from src.client import FlowerClient
except ImportError as err:
    print(f"Error importing modules: {err}", file=sys.stderr)
    print("Ensure you are running from the project root.", file=sys.stderr)
    sys.exit(1)


def resolve_project_root() -> Path:
    return PROJECT_ROOT


def load_config(overrides: List[str]) -> DictConfig:
    """Load config using Hydra."""
    config_dir = resolve_project_root() / "conf"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config_fl", overrides=overrides)
    return cfg


def run_single_client(
    cfg: DictConfig, cid: str, data_path: Path, device: Optional[torch.device]
) -> None:
    """Instantiate and start a single Flower client."""
    try:
        # Initialize the Custom Client
        client = FlowerClient(cid=cid, cfg=cfg, data_path=str(data_path), device=device)

        # Start the Flower Client connection
        # NOTE: to_client() converts the NumPyClient to a standard Client
        fl.client.start_client(
            server_address=cfg.server.address, client=client.to_client()
        )
    except Exception as exc:
        logging.getLogger(__name__).critical(
            "Client %s crashed: %s", cid, exc, exc_info=True
        )
        sys.exit(1)


def launch_multiple_clients(
    start: int, end: int, data_path_pattern: str, hydra_overrides: List[str]
):
    """Spawns subprocesses for the specified range of clients."""
    processes = []
    print(f"--- Spawning clients {start} to {end} ---")

    for i in range(start, end + 1):
        cid = str(i)
        # Format the data path (e.g., data/processed/client_1.pt)
        # Handles cases where {cid} is in the pattern
        specific_data_path = data_path_pattern.format(cid=cid)

        # Construct the command to call this script recursively in single mode
        cmd = [
            sys.executable,
            __file__,
            "--cid",
            cid,
            "--data_path",
            specific_data_path,
        ] + hydra_overrides

        # Spawn the process
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"Started Client {cid} (PID: {p.pid})")

        # Stagger starts slightly to avoid hammering the CPU/Disk
        time.sleep(1.0)

    print(f"--- All {len(processes)} clients launched. Waiting for completion... ---")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all clients...")
        for p in processes:
            p.terminate()


def parse_args():
    parser = argparse.ArgumentParser(description="Flower Client for FedOsQ Chain")

    # Multi-client mode argument
    parser.add_argument(
        "--cid_range",
        type=str,
        help="Range of clients to run (e.g., '1-5'). If set, ignores --cid and spawns subprocesses.",
    )

    # Single client mode arguments
    parser.add_argument("--cid", help="Client ID (e.g., '1')")
    parser.add_argument(
        "--data_path",
        default="data/processed/client_{cid}.pt",
        help="Path pattern for data. Use {cid} as placeholder (e.g., 'data/processed/client_{cid}.pt')",
    )
    return parser.parse_known_args()


def main() -> None:
    args, hydra_overrides = parse_args()

    # --- MULTI-CLIENT MODE ---
    if args.cid_range:
        try:
            start_s, end_s = args.cid_range.split("-")
            start, end = int(start_s), int(end_s)
        except ValueError:
            print(
                "Error: --cid_range must be in format 'start-end' (e.g., '1-5')",
                file=sys.stderr,
            )
            sys.exit(1)

        launch_multiple_clients(start, end, args.data_path, hydra_overrides)
        return

    # --- SINGLE CLIENT MODE ---
    if not args.cid:
        print(
            "Error: --cid is required for single client mode (or use --cid_range).",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Load Config
    cfg = load_config(hydra_overrides)
    project_root = resolve_project_root()

    # 2. Resolve Data Path
    # Replace {cid} placeholder if present
    resolved_path_str = args.data_path.format(cid=args.cid)
    data_path = (project_root / resolved_path_str).resolve()

    if not data_path.exists():
        # Fallback check: sometimes filenames differ (e.g. client_1_train.pt vs client_1.pt)
        # You can add custom fallback logic here if needed
        print(f"Error: Data path not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # 3. Setup Logging
    log_file = project_root / "logs" / f"client_{args.cid}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_level = str(cfg.get("log_level", "INFO")).upper()
    setup_logging(log_file_path=str(log_file), log_level=log_level)

    logger = logging.getLogger(__name__)
    device = resolve_device_from_config(cfg)

    logger.info("--- Starting Client %s ---", args.cid)
    logger.info("Server Address: %s", cfg.server.address)
    logger.info("Data Path: %s", data_path)

    run_single_client(cfg, args.cid, data_path, device)


if __name__ == "__main__":
    main()
