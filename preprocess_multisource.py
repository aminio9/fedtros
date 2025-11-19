import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.utils import setup_logging

logger = logging.getLogger("PreprocessMulti")


def resolve_path(path_like: str, project_root: Path) -> Path:
    """Resolves a path relative to the project root if it's not absolute."""
    path = Path(path_like)
    return path if path.is_absolute() else (project_root / path)


def save_as_torch(features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Saves features and labels as a PyTorch tensor file."""
    try:
        data = {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        torch.save(data, path)
        logger.info(f"Saved {len(labels)} samples -> {path.name}")
    except Exception as e:
        logger.error(f"Failed to save data to {path}: {e}", exc_info=True)


def load_source_data(source_map: List[Dict], project_root: Path, required_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Loads all CSV files into a dictionary keyed by client_id."""
    loaded_data = {}
    for src in source_map:
        client_id = src["client_id"]
        file_path = resolve_path(src["path"], project_root)

        if not file_path.exists():
            logger.critical(f"FATAL: Source file not found: {file_path}")
            sys.exit(1)

        try:
            df = pd.read_csv(file_path)
            # Validation
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.critical(f"FATAL: {client_id} ({file_path.name}) missing columns: {missing}")
                sys.exit(1)
            
            loaded_data[client_id] = df
            logger.info(f"Loaded {client_id}: {len(df)} rows from {file_path.name}")
        except Exception as e:
            logger.critical(f"Error reading {file_path}: {e}")
            sys.exit(1)
            
    return loaded_data


def fit_global_processors(
    loaded_data: Dict[str, pd.DataFrame], 
    known_labels: List[str], 
    label_col: str, 
    num_cols: List[str], 
    cat_cols: List[str]
) -> Tuple[MinMaxScaler, OneHotEncoder]:
    """
    Combines KNOWN data from all sources to fit Scaler and Encoder.
    This ensures consistent state dimensions across all clients.
    """
    logger.info("--- Phase 1: Global Fitting ---")
    global_known_list = []

    for client_id, df in loaded_data.items():
        df_known = df[df[label_col].isin(known_labels)]
        global_known_list.append(df_known)
    
    global_known_df = pd.concat(global_known_list, ignore_index=True)
    
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    logger.info(f"Fitting processors on {len(global_known_df)} total known samples...")
    scaler.fit(global_known_df[num_cols])
    encoder.fit(global_known_df[cat_cols])
    
    return scaler, encoder


def process_client(
    client_id: str,
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
    cfg_preprocess: DictConfig,
    label_map: Dict[str, int],
    output_dir: Path
):
    """
    Process a single client's dataframe:
    1. Split into Known/Unknown.
    2. Transform features.
    3. Split Known -> Train + Closed-Test.
    4. Create Open-Test (Closed-Test + Unknowns).
    5. Save all files.
    """
    p_cfg = cfg_preprocess
    label_col = p_cfg.label_column
    num_cols = list(p_cfg.numerical_cols)
    cat_cols = list(p_cfg.categorical_cols)
    known_labels = list(p_cfg.known_labels)
    
    # 1. Separation
    df_known = df[df[label_col].isin(known_labels)].copy()
    df_unknown = df[~df[label_col].isin(known_labels)].copy()

    # Get State Dim for empty fallback
    cat_dim = encoder.get_feature_names_out().shape[0]
    state_dim = len(num_cols) + cat_dim

    # --- PROCESS KNOWN DATA ---
    if len(df_known) > 0:
        # Transform
        feat_num = scaler.transform(df_known[num_cols])
        feat_cat = encoder.transform(df_known[cat_cols])
        features_known = np.concatenate([feat_num, feat_cat], axis=1).astype(np.float32)
        labels_known = df_known[label_col].map(label_map).values

        # Split Train / Test (Stratified if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features_known, labels_known,
                test_size=p_cfg.closed_set_test_size,
                stratify=labels_known,
                random_state=42
            )
        except ValueError:
            logger.warning(f"{client_id}: Cannot stratify (class imbalance). Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                features_known, labels_known,
                test_size=p_cfg.closed_set_test_size,
                random_state=42
            )
        
        # Save Train
        save_as_torch(X_train, y_train, output_dir / f"{client_id}_train.pt")
        # Save Closed Test
        save_as_torch(X_test, y_test, output_dir / f"{client_id}_test_closed.pt")
        
        # Initialize Open Set with Closed Set Test data
        open_features = X_test
        open_labels = y_test
    else:
        logger.warning(f"{client_id}: No KNOWN data found.")
        open_features = np.empty((0, state_dim), dtype=np.float32)
        open_labels = np.empty((0,), dtype=np.int64)

    # --- PROCESS UNKNOWN DATA (Open Set Additions) ---
    if len(df_unknown) > 0:
        feat_num_unk = scaler.transform(df_unknown[num_cols])
        feat_cat_unk = encoder.transform(df_unknown[cat_cols])
        features_unknown = np.concatenate([feat_num_unk, feat_cat_unk], axis=1).astype(np.float32)
        
        # Label -1 for Unknown
        labels_unknown = np.full(len(df_unknown), -1, dtype=np.int64)
        
        # Combine
        open_features = np.concatenate([open_features, features_unknown], axis=0)
        open_labels = np.concatenate([open_labels, labels_unknown], axis=0)
        logger.info(f"{client_id}: Added {len(labels_unknown)} unknown samples to Open Set.")
    
    # Save Open Test
    save_as_torch(open_features, open_labels, output_dir / f"{client_id}_test_open.pt")


@hydra.main(config_path="conf", config_name="config_fl", version_base=None)
def run_preprocessing(cfg: DictConfig):
    project_root = Path(get_original_cwd())
    
    # Logging
    log_file = project_root / "logs" / "preprocess_multi.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_file), str(cfg.get("log_level", "INFO")).upper())

    logger.info("--- ?? Starting Refactored Multi-Source Preprocessing ---")
    
    p_cfg = cfg.preprocess
    output_dir = resolve_path(p_cfg.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    known_labels = list(p_cfg.known_labels)
    label_map = {label: i for i, label in enumerate(known_labels)}
    idx_to_label = {i: label for label, i in label_map.items()}
    
    required_cols = [p_cfg.label_column] + list(p_cfg.numerical_cols) + list(p_cfg.categorical_cols)

    # Save Class Map
    with open(output_dir / "class_names.json", "w") as f:
        json.dump({int(k): v for k, v in idx_to_label.items()}, f, indent=2)

    # Define Sources
    source_map = [
        {"path": "data/raw/w1.csv", "client_id": "client_1"},
        {"path": "data/raw/w2.csv", "client_id": "client_2"},
        {"path": "data/raw/w3.csv", "client_id": "client_3"},
    ]

    # 1. Load Data
    loaded_data = load_source_data(source_map, project_root, required_cols)

    # 2. Global Fit (Ensure unified State Dim)
    scaler, encoder = fit_global_processors(
        loaded_data, known_labels, 
        p_cfg.label_column, 
        list(p_cfg.numerical_cols), 
        list(p_cfg.categorical_cols)
    )
    
    cat_dim = encoder.get_feature_names_out().shape[0]
    state_dim = len(p_cfg.numerical_cols) + cat_dim
    logger.info(f"Global State Dimension established: {state_dim}")

    # 3. Process Each Client
    logger.info("--- Phase 2: Individual Client Processing ---")
    for src in source_map:
        client_id = src["client_id"]
        logger.info(f"Processing {client_id}...")
        process_client(
            client_id=client_id,
            df=loaded_data[client_id],
            scaler=scaler,
            encoder=encoder,
            cfg_preprocess=p_cfg,
            label_map=label_map,
            output_dir=output_dir
        )

    print("\n" + "=" * 60)
    print(f"  Preprocessing Complete.")
    print(f"  Env Metadata (Update config_fl.yaml):")
    print(f"    state_dim: {state_dim}")
    print(f"    num_actions: {len(known_labels)}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_preprocessing()