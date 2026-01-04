import logging
import json
import torch
import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset, DataLoader

# Adjust based on your folder structure
try:
    from models import OpenSetQChainModelFactory
    from evt import save_evt_collection, save_evt_meta
    from agent import Agent
    from openset_eval import fit_evt_models, calibrate_evt_thresholds, evaluate_open_set
except ImportError:
    from src.models import OpenSetQChainModelFactory
    from src.evt import save_evt_collection, save_evt_meta
    from src.agent import Agent
    from src.openset_eval import fit_evt_models, calibrate_evt_thresholds, evaluate_open_set

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandaloneEVT")

def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(os.getcwd()) / path

def load_data(path_str: str, device: torch.device):
    path = _resolve_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
        
    try:
        # Weights_only=False to allow dictionary load
        data = torch.load(path, map_location=device, weights_only=False)
        features = data["features"].float().to(device)
        labels = data["labels"].long().to(device)
        logger.info(f"Loaded data from {path.name}: {len(labels)} samples.")
        return features, labels
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {e}")
        raise

def load_global_checkpoint(agent: Agent, checkpoint_path: str, device: torch.device):
    path = _resolve_path(checkpoint_path)
    if not path.exists():
        logger.warning(f"Checkpoint not found at {path}. Using random weights.")
        return

    logger.info(f"Loading global model checkpoint from: {path.name}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        if "prior_net" in checkpoint:
            agent.prior_net.load_state_dict(checkpoint["prior_net"])
        
        if "recognition_net" in checkpoint:
            agent.recognition_net.load_state_dict(checkpoint["recognition_net"])

        if "value_net_main" in checkpoint:
            agent.value_net_main.load_state_dict(checkpoint["value_net_main"])
            agent.value_net_target.load_state_dict(checkpoint["value_net_main"])

        if agent.generation_net is not None:
            if "generation_net" in checkpoint and checkpoint["generation_net"] is not None:
                agent.generation_net.load_state_dict(checkpoint["generation_net"])
                logger.info("Generator weights loaded successfully.")
        
        logger.info(f"Global model loaded (Round {checkpoint.get('round', 'Unknown')})")

    except Exception as e:
        logger.exception(f"Critical error loading checkpoint: {e}")
        raise

@hydra.main(config_path="conf", config_name="config_fl", version_base=None)
def main(cfg: DictConfig):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.device.prefer == "directml":
         try:
             import torch_directml
             device = torch_directml.device()
         except ImportError:
             pass
    logger.info(f"Running on device: {device}")

    logger.info("Initializing Models...")
    model_factory = OpenSetQChainModelFactory(cfg.model)
    agent = Agent(model_factory, cfg.training, device=device)

    checkpoint_path = getattr(cfg.paths, "global_model_checkpoint", "models/federated_global/global_model_latest.pt")
    load_global_checkpoint(agent, checkpoint_path, device)

    agent.prior_net.eval()
    agent.recognition_net.eval()
    agent.value_net_main.eval()
    if agent.generation_net:
        agent.generation_net.eval()

    logger.info("--- Loading Calibration Data ---")
    calib_data_path = cfg.paths.closed_set_test_data 
    calib_features, calib_labels = load_data(calib_data_path, device)

    logger.info("--- Loading Open Set Data ---")
    open_data_path = cfg.paths.open_set_test_data
    open_features, open_labels = load_data(open_data_path, device)

    logger.info("--- Phase 1: Fitting EVT Models ---")
    logger.info(f"EVT Config: Tail={cfg.evt.tail_size_percent}")

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
    
    output_dir = _resolve_path("evt_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_evt_collection(evt_models, output_dir / "standalone_evt_models.pkl")
    save_evt_meta(evt_meta, output_dir / "standalone_evt_meta.json")

    logger.info("--- Phase 3: Open Set Recognition Evaluation ---")
    
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

    logger.info(f"{'='*30}")
    logger.info(f"FINAL AUROC: {metrics.get('openset_auroc', 0.0):.4f}")
    logger.info(f"Unknown F1:  {metrics.get('openset_f1_unknown', 0.0):.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*30}")

if __name__ == "__main__":
    main()