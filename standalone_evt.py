import logging
import json
import torch
import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset, DataLoader

# Import your project modules
try:
    from models import OpenSetQChainModelFactory
    from evt import save_evt_collection, save_evt_meta
    from agent import Agent
    from openset_eval import fit_evt_models, calibrate_evt_thresholds, evaluate_open_set
except ImportError:
    # Fallback for running from different directory contexts
    from .models import OpenSetQChainModelFactory
    from .evt import save_evt_collection, save_evt_meta
    from .agent import Agent
    from .openset_eval import fit_evt_models, calibrate_evt_thresholds, evaluate_open_set

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneEVT")

def _resolve_path(path_str: str) -> Path:
    """Helper to resolve paths relative to the project root if needed."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    # Assuming the script is run from the project root
    return Path(os.getcwd()) / path

def load_data(path_str: str, device: torch.device):
    """Helper to load .pt feature files."""
    path = _resolve_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
        
    try:
        data = torch.load(path, map_location=device)
        features = data["features"].float().to(device)
        labels = data["labels"].long().to(device)
        logger.info(f"Loaded data from {path.name}: {len(labels)} samples.")
        return features, labels
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {e}")
        raise

def load_global_checkpoint(agent: Agent, checkpoint_path: str, device: torch.device):
    """
    Loads the custom dictionary format saved by server.py into the Agent's sub-networks.
    """
    path = _resolve_path(checkpoint_path)
    if not path.exists():
        logger.warning(f"Checkpoint not found at {path}. Using random weights (Debugging only!)")
        return

    logger.info(f"Loading global model checkpoint from: {path.name}")
    try:
        # Load dictionary
        checkpoint = torch.load(path, map_location=device)
        
        # 1. Load Prior
        if "prior_net" in checkpoint:
            agent.prior_net.load_state_dict(checkpoint["prior_net"])
        else:
            logger.error("Checkpoint missing 'prior_net' key!")

        # 2. Load Recognition
        if "recognition_net" in checkpoint:
            agent.recognition_net.load_state_dict(checkpoint["recognition_net"])
        else:
            logger.error("Checkpoint missing 'recognition_net' key!")

        # 3. Load Main Q
        if "value_net_main" in checkpoint:
            agent.value_net_main.load_state_dict(checkpoint["value_net_main"])
            # Sync target network for consistency
            agent.value_net_target.load_state_dict(checkpoint["value_net_main"])
        else:
            logger.error("Checkpoint missing 'value_net_main' key!")

        # 4. Load Generator (Optional)
        # Check if agent has a generator AND checkpoint has weights for it
        if agent.generation_net is not None:
            if "generation_net" in checkpoint and checkpoint["generation_net"] is not None:
                agent.generation_net.load_state_dict(checkpoint["generation_net"])
                logger.info("Generator weights loaded successfully.")
            else:
                logger.warning("Agent has Generator, but checkpoint did not contain generator weights (or they were None).")
        
        logger.info(f"Global model loaded (Round {checkpoint.get('round', 'Unknown')})")

    except Exception as e:
        logger.exception(f"Critical error loading checkpoint: {e}")
        raise

@hydra.main(config_path="conf", config_name="config_fl", version_base=None)
def main(cfg: DictConfig):
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.device.prefer == "directml":
         try:
             import torch_directml
             device = torch_directml.device()
         except ImportError:
             pass
    logger.info(f"Running on device: {device}")

    # 2. Initialize Models
    logger.info("Initializing Models...")
    model_factory = OpenSetQChainModelFactory(cfg.model)
    
    # Initialize Agent
    # We pass cfg.training because Agent needs it for optimizer init, even for inference
    agent = Agent(model_factory, cfg.training, device=device)

    # 3. Load Pre-trained Weights
    # You can specify this in your config under paths.global_model_checkpoint
    # Or fallback to a hardcoded path if not in config
    checkpoint_path = getattr(cfg.paths, "global_model_checkpoint", "models/federated_global/global_model_latest.pt")
    
    load_global_checkpoint(agent, checkpoint_path, device)

    # Set models to Eval mode
    agent.prior_net.eval()
    agent.recognition_net.eval()
    agent.value_net_main.eval()
    if agent.generation_net:
        agent.generation_net.eval()

    # 4. Load Data
    # A. Calibration Data (Closed Set Test Data)
    logger.info("--- Loading Calibration Data ---")
    calib_data_path = cfg.paths.closed_set_test_data 
    calib_features, calib_labels = load_data(calib_data_path, device)

    # B. Open Set Data (Contains Unknowns)
    logger.info("--- Loading Open Set Data ---")
    open_data_path = cfg.paths.open_set_test_data
    open_features, open_labels = load_data(open_data_path, device)

    # 5. Fit EVT Models
    logger.info("--- Phase 1: Fitting EVT Models ---")
    logger.info(f"EVT Config: Tail={cfg.evt.tail_size_percent}, Threshold={cfg.evt.decision_threshold}")

    try:
        evt_models = fit_evt_models(
            features=calib_features,
            labels=calib_labels,
            batch_size=cfg.training.batch_size,
            evt_cfg=cfg.evt,
            prior_net=agent.prior_net,
            recognition_net=agent.recognition_net,
            value_net_main=agent.value_net_main,
            generation_net=agent.generation_net,
            device=device
        )
        logger.info("EVT Fitting Complete.")
    except Exception as e:
        logger.exception("EVT fitting failed.")
        return

    # 6. Calibrate Thresholds
    logger.info("--- Phase 2: Calibrating Thresholds ---")
    try:
        evt_meta = calibrate_evt_thresholds(
            features=calib_features,
            labels=calib_labels,
            batch_size=cfg.training.batch_size,
            evt_models=evt_models,
            evt_cfg=cfg.evt,
            prior_net=agent.prior_net,
            recognition_net=agent.recognition_net,
            value_net_main=agent.value_net_main,
            generation_net=agent.generation_net,
            device=device
        )
    except Exception as e:
        logger.exception("Calibration failed.")
        return
    
    # Save the Fitted EVT models
    output_dir = _resolve_path("evt_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_evt_collection(evt_models, output_dir / "standalone_evt_models.pkl")
    save_evt_meta(evt_meta, output_dir / "standalone_evt_meta.json")

    # 7. Run Open Set Evaluation
    logger.info("--- Phase 3: Open Set Recognition Evaluation ---")
    
    # Load class names
    class_names = {}
    cn_path = _resolve_path(cfg.paths.class_names)
    if cn_path.exists():
         try:
             with open(cn_path, 'r') as f:
                 class_names = {int(k): v for k, v in json.load(f).items()}
         except Exception:
             logger.warning("Could not load class names json.")

    metrics = evaluate_open_set(
        features=open_features,
        labels=open_labels,
        batch_size=cfg.training.batch_size,
        prior_net=agent.prior_net,
        recognition_net=agent.recognition_net,
        value_net_main=agent.value_net_main,
        generation_net=agent.generation_net,
        evt_models=evt_models,
        evt_meta=evt_meta,
        class_names=class_names,
        output_dir=output_dir,
        device=device,
        report_to_stdout=True
    )

    # 8. Summary
    logger.info(f"{'='*30}")
    logger.info(f"FINAL AUROC: {metrics.get('openset_auroc', 0.0):.4f}")
    logger.info(f"Unknown F1:  {metrics.get('openset_f1_unknown', 0.0):.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*30}")

if __name__ == "__main__":
    main()