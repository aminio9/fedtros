import argparse
import logging
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

import flwr as fl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

# Ensure both the project root and the src/ directory are importable when running as a script
for candidate in (PROJECT_ROOT, SRC_PATH):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from src.utils import resolve_device_from_config, setup_logging
    from src.client import FlowerClient
except ImportError as err:
    print("Error: Could not import project modules.", file=sys.stderr)
    print(f"Root cause: {err}", file=sys.stderr)
    print("Ensure you're running from the repository root and have installed dependencies via 'poetry install'.", file=sys.stderr)
    raise


def resolve_project_root() -> Path:
    """Return the project root path regardless of Hydra initialization."""
    return PROJECT_ROOT


def load_config(overrides: List[str]) -> DictConfig:
    """Load Hydra/OmegaConf configuration with optional overrides."""
    config_dir = resolve_project_root() / "conf"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config_fl", overrides=overrides)
    return cfg


def run_single_client(cfg: DictConfig, cid: str, data_path: Path, device: Optional[torch.device]) -> None:
    """Instantiate and start a Flower client."""
    try:
        client = FlowerClient(cid=cid, cfg=cfg, data_path=str(data_path), device=device)
        fl.client.start_client(server_address=cfg.server.address, client=client.to_client())
    except ValueError as exc:
        logging.getLogger(__name__).critical("Configuration Error: %s", exc, exc_info=True)
        print(f"\nFATAL: Configuration mismatch. {exc}", file=sys.stderr)
        print("Please check your 'conf/config_fl.yaml' and 'data/processed' files.", file=sys.stderr)
        sys.exit(1)
    except Exception:
        logging.getLogger(__name__).critical("Client %s crashed.", cid, exc_info=True)
        raise

def run_multiple_clients(start: int, end: int, data_path_pattern: str, hydra_overrides: List[str]):
    processes = []
    print(f"Initializing clients {start}-{end}...")

    for i in range(start, end+1):
        cid = str(i)
        
        # Format the data path (e.g., data/processed/client_1.pt)
        specific_data_path = data_path_pattern.format(cid=cid)
        
        cmd = [
            sys.executable,
            __file__,
            "--cid", cid,
            "--data_path", specific_data_path,
        ]+ hydra_overrides
        
        #initiate the process
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"Started Client {cid} (PID: {p.pid})")
        
    print(f"--- All {len(processes)} clients launched. Waiting for completion... ---")
    
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all clients...")
        for p in processes:
            p.terminate()



def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Flower Client for FedOsQ Chain", add_help=True)
    
    #for running multiple clients
    parser.add_argument(
        "--cid_range", 
        type=str, 
        help="Range of clients to run (e.g., '0-50'). If set, ignores --cid and spawns subprocesses."
    )
    
    parser.add_argument("--cid", required=True, help="Client ID (e.g., '1', '2', '3')")
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the client's local data file (e.g., 'data/processed/client_1.pt')",
    )
    return parser.parse_known_args()


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
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

# Ensure imports work
for candidate in (PROJECT_ROOT, SRC_PATH):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from src.utils import resolve_device_from_config, setup_logging
    from src.client import FlowerClient
except ImportError as err:
    print(f"Error importing modules: {err}", file=sys.stderr)
    sys.exit(1)


def resolve_project_root() -> Path:
    return PROJECT_ROOT


def load_config(overrides: List[str]) -> DictConfig:
    config_dir = resolve_project_root() / "conf"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config_fl", overrides=overrides)
    return cfg


def run_single_client(cfg: DictConfig, cid: str, data_path: Path, device: Optional[torch.device]) -> None:
    """Instantiate and start a single Flower client."""
    try:
        client = FlowerClient(cid=cid, cfg=cfg, data_path=str(data_path), device=device)
        fl.client.start_client(server_address=cfg.server.address, client=client.to_client())
    except Exception as exc:
        logging.getLogger(__name__).critical("Client %s crashed: %s", cid, exc, exc_info=True)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Flower Client for DC-IDS")
    
    # New argument for running multiple clients
    parser.add_argument(
        "--cid_range", 
        type=str, 
        help="Range of clients to run (e.g., '0-50'). If set, ignores --cid and spawns subprocesses."
    )
    
    parser.add_argument("--cid", help="Client ID (single mode)")
    parser.add_argument(
        "--data_path",
        help="Path pattern for data. Use {cid} as placeholder (e.g., 'data/processed/client_{cid}.pt')",
    )
    return parser.parse_known_args()


def launch_multiple_clients(start: int, end: int, data_path_pattern: str, hydra_overrides: List[str]):
    """Spawns subprocesses for the specified range of clients."""
    processes = []
    print(f"--- Spawning clients {start} to {end} ---")
    
    for i in range(start, end + 1):
        cid = str(i)
        # Format the data path (e.g., data/processed/client_1.pt)
        specific_data_path = data_path_pattern.format(cid=cid)
        
        # Construct the command to call this script recursively in single mode
        cmd = [
            sys.executable, 
            __file__, 
            "--cid", cid, 
            "--data_path", specific_data_path
        ] + hydra_overrides
        
        # Spawn the process
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"Started Client {cid} (PID: {p.pid})")
        
        # Stagger starts slightly to avoid hammering the CPU/Disk all at once
        time.sleep(0.5) 

    print(f"--- All {len(processes)} clients launched. Waiting for completion... ---")
    
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all clients...")
        for p in processes:
            p.terminate()


def main() -> None:
    args, hydra_overrides = parse_args()

    # --- MULTI-CLIENT MODE ---
    if args.cid_range:
        try:
            start_s, end_s = args.cid_range.split("-")
            start, end = int(start_s), int(end_s)
        except ValueError:
            print("Error: --cid_range must be in format 'start-end' (e.g., '0-10')", file=sys.stderr)
            sys.exit(1)

        if not args.data_path:
            # Default fallback if user forgot data_path in multi-mode
            args.data_path = "data/processed/client_{cid}_train.pt"
        
        launch_multiple_clients(start, end, args.data_path, hydra_overrides)
        return

    # --- SINGLE CLIENT MODE (Standard) ---
    if not args.cid or not args.data_path:
        print("Error: --cid and --data_path are required for single client mode.", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(hydra_overrides)
    project_root = resolve_project_root()
    
    # Handle the {cid} placeholder if it was passed explicitly even in single mode
    resolved_data_path_str = args.data_path.format(cid=args.cid)
    data_path = (project_root / resolved_data_path_str).resolve()

    if not data_path.exists():
        print(f"Error: Data path not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Setup Logging
    log_file = project_root / "logs" / f"client_{args.cid}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    log_level = str(cfg.get("log_level", "INFO")).upper()
    setup_logging(log_file_path=str(log_file), log_level=log_level)

    logger = logging.getLogger(__name__)
    device = resolve_device_from_config(cfg)
    
    logger.info("--- Starting Client %s ---", args.cid)
    logger.info("Data Path: %s", data_path)
    run_single_client(cfg, args.cid, data_path, device)
    logger.info("--- Client %s finished ---", args.cid)

if __name__ == "__main__":
    main()
