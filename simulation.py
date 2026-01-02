import sys
import logging
import traceback
from pathlib import Path
from functools import partial

import flwr as fl
from flwr.common import Context  # Import Context
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

# --- 1. SETUP PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_PATH):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Import your project modules
try:
    from src.client import FlowerClient
    from src.server import get_strategy
    from src.utils import resolve_device_from_config
except ImportError as e:
    sys.exit(f"Error importing src modules: {e}")

logger = logging.getLogger("Simulation")

# --- 2. DEFINE CLIENT FUNCTION ---
def client_fn(context: Context) -> fl.client.Client:
    """
    Worker Function.
    Constructs a client based on the context provided by Flower.
    """
    try:
        # Retrieve client ID (partition ID) from context
        cid = context.node_config["partition-id"]
        
        # Retrieve the config dict passed via 'run_config' in start_simulation
        # Note: Flower passes this inside the context in newer versions, 
        # but for legacy compatibility we might need to rely on the bound partial or 
        # reloading. However, strict 'Context' signature means we should extract what we can.
        
        # Robust Config Loading Strategy:
        # Since we can't easily pass the complex Hydra config through the strictly typed 
        # 'Context' without serialization issues in some versions, we re-initialize 
        # Hydra locally. It's safer and cleaner.
        
        with initialize(version_base=None, config_path="conf", job_name=f"worker_{cid}"):
            cfg = compose(config_name="config_fl")

        # Resolve Device
        # Force CPU for workers to prevent OOM in simulation unless explicitly set otherwise
        device = torch.device("cpu") 
        if cfg.device.prefer == "cuda" and torch.cuda.is_available():
             # Ray handles the CUDA_VISIBLE_DEVICES, so we can just say 'cuda' 
             # if we trust Ray's resource allocation. 
             # But for safety against OOM, CPU is often preferred for massive concurrency.
             # You can uncomment this if you trust the resource limits:
             # device = torch.device("cuda")
             pass

        # Resolve Data Path
        data_path_str = f"data/processed/client_{cid}_train.pt"
        data_path = (PROJECT_ROOT / data_path_str).resolve()

        if not data_path.exists():
            # In simulation, partition IDs might be 0-based integers (0, 1, 2...)
            # Your files might be 1-based (client_1_train.pt). Let's try adjusting.
            try_cid = int(cid) + 1
            data_path_alt = (PROJECT_ROOT / f"data/processed/client_{try_cid}_train.pt").resolve()
            if data_path_alt.exists():
                cid = str(try_cid)
                data_path = data_path_alt
            else:
                print(f"!! CRITICAL ERROR: Client {cid} data not found at {data_path}")
                raise FileNotFoundError(f"Data not found: {data_path}")

        # Instantiate Custom Client
        client = FlowerClient(
            cid=str(cid), 
            cfg=cfg, 
            data_path=str(data_path), 
            device=device
        )

        return client.to_client()

    except Exception as e:
        print(f"!!! CLIENT {cid} CRASHED !!!")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    # --- MAIN SIMULATION ENTRY POINT ---
    
    # 1. Load Global Config ONCE for Server Setup
    with initialize(version_base=None, config_path="conf", job_name="sim_master"):
        cfg = compose(config_name="config_fl")
    
    # 2. Define Resources
    # 10 clients parallel = 0.1 GPU each.
    client_resources = {"num_cpus": 1, "num_gpus": 0.0} 
    if cfg.device.prefer == "cuda":
        client_resources["num_gpus"] = 0.1

    print(f"\n{'='*40}")
    print(f"Starting Robust Simulation")
    print(f"Total Clients: {cfg.preprocess.num_clients}")
    print(f"Resources per client: {client_resources}")
    print(f"{'='*40}\n")

    # 3. Start
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.preprocess.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=get_strategy(cfg),
        client_resources=client_resources,
        ray_init_args={
            "include_dashboard": False,
            "log_to_driver": False,  # <--- CHANGE THIS TO FALSE
            "configure_logging": True,
            "logging_level": logging.ERROR,
        }
    )