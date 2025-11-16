import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

try:
    from src.utils import setup_logging
except ImportError:
    from utils import setup_logging

logger = logging.getLogger("Preprocess")


def save_as_torch(features: np.ndarray, labels: np.ndarray, path: Path):
    """Saves features and labels as a PyTorch tensor file."""
    try:
        data = {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        torch.save(data, path)
        logger.info(f"Successfully saved {len(labels)} samples to {path}\n")
    except Exception as e:
        logger.error(f"Failed to save data to {path}: {e}", exc_info=True)


@hydra.main(config_path="conf", config_name="config_fl", version_base=None)
def run_preprocessing(cfg: DictConfig):
    """Main preprocessing pipeline."""

    project_root = Path(get_original_cwd())
    log_file = project_root / "logs" / "preprocess.log"
    log_level = str(cfg.get("log_level", "INFO")).upper()
    setup_logging(log_file_path=str(log_file), log_level=log_level)

    logger.info("--- ?? Starting Preprocessing Pipeline ---")

    def resolve_path(path_like) -> Path:
        path = Path(path_like)
        return path if path.is_absolute() else (project_root / path)

    try:
        p_cfg = cfg.preprocess
        raw_file = resolve_path(p_cfg.raw_file)
        output_dir = resolve_path(p_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(raw_file)
        logger.info(f"Loaded {raw_file} with {len(df)} rows.")
    except FileNotFoundError:
        logger.critical(f"FATAL: Raw data file not found at {raw_file}")
        logger.critical("Please download the data and place it in 'data/raw/'.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: Error loading data or config: {e}", exc_info=True)
        sys.exit(1)

    try:
        label_col = p_cfg.label_column
        cat_cols = list(p_cfg.categorical_cols)
        num_cols = list(p_cfg.numerical_cols)
        all_cols = [label_col] + cat_cols + num_cols

        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            logger.critical(f"FATAL: Columns from config not in CSV: {missing_cols}")
            logger.critical("Please check 'preprocess' section in 'config_fl.yaml'.")
            sys.exit(1)
        logger.info("Column configuration validated.")
    except Exception as e:
        logger.critical(f"FATAL: Config error in column definitions: {e}")
        sys.exit(1)

    known_labels_list = list(p_cfg.known_labels)
    label_map = {label: i for i, label in enumerate(known_labels_list)}
    idx_to_label = {i: label for label, i in label_map.items()}
    num_actions = len(label_map)

    df_known = df[df[label_col].isin(known_labels_list)].copy()
    df_unknown = df[~df[label_col].isin(known_labels_list)].copy()

    logger.info("Total data split:")
    logger.info(f"  -> {len(df_known)} samples for 'Known' (Closed-Set) Training/Testing")
    logger.info(f"  -> {len(df_unknown)} samples for 'Unknown' (Open-Set) Testing")

    if len(df_known) == 0:
        logger.critical("FATAL: No 'Known' samples found. Check 'known_labels' in config.")
        sys.exit(1)

    class_names_path = output_dir / "class_names.json"
    try:
        with open(class_names_path, "w", encoding="utf-8") as f:
            json.dump({int(idx): name for idx, name in idx_to_label.items()}, f, indent=2)
        logger.info(f"Saved class-name mapping to {class_names_path}")
    except Exception as exc:
        logger.error(f"Failed to save class_names.json: {exc}", exc_info=True)

    logger.info("Fitting Scaler and Encoder on KNOWN data...")
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    scaler.fit(df_known[num_cols])
    encoder.fit(df_known[cat_cols])

    cat_feature_count = encoder.get_feature_names_out().shape[0]
    state_dim = len(num_cols) + cat_feature_count
    logger.info(f"Scaler fit on {len(num_cols)} numerical features.")
    logger.info(f"Encoder fit on {len(cat_cols)} categorical features, resulting in {cat_feature_count} one-hot features.")

    logger.info("Processing KNOWN data...")
    num_scaled_known = scaler.transform(df_known[num_cols])
    cat_encoded_known = encoder.transform(df_known[cat_cols])
    features_known = np.concatenate(
        [num_scaled_known, cat_encoded_known],
        axis=1,
    ).astype(np.float32)
    labels_known = df_known[label_col].map(label_map).values

    X_train_known, X_test_closed, y_train_known, y_test_closed = train_test_split(
        features_known,
        labels_known,
        test_size=p_cfg.closed_set_test_size,
        stratify=labels_known,
        random_state=42,
    )

    logger.info("--- ?? Closed-Set Test Data (closed_set_test.pt) ---")
    logger.info(f"Shape: {X_test_closed.shape}")
    if len(y_test_closed) > 0:
        unique_labels, counts = np.unique(y_test_closed, return_counts=True)
        label_dist_str = ", ".join([f"'{idx_to_label[l]}' (ID {l}): {c}" for l, c in zip(unique_labels, counts)])
        logger.info(f"Label Distribution: {label_dist_str}")
    save_as_torch(X_test_closed, y_test_closed, output_dir / "closed_set_test.pt")

    num_clients = p_cfg.num_clients
    logger.info(f"Splitting KNOWN training data for {num_clients} clients...")

    indices = np.random.permutation(len(X_train_known))
    X_train_known, y_train_known = X_train_known[indices], y_train_known[indices]
    client_indices = np.array_split(range(len(X_train_known)), num_clients)

    for i in range(num_clients):
        client_id = i + 1
        client_idx_set = client_indices[i]
        X_client = X_train_known[client_idx_set]
        y_client = y_train_known[client_idx_set]

        logger.info(f"--- ?? Client {client_id} Training Data (client_{client_id}_train.pt) ---")
        logger.info(f"Shape: {X_client.shape}")
        if len(y_client) > 0:
            unique_labels, counts = np.unique(y_client, return_counts=True)
            label_dist_str = ", ".join([f"'{idx_to_label[l]}' (ID {l}): {c}" for l, c in zip(unique_labels, counts)])
            logger.info(f"Label Distribution: {label_dist_str}")
        else:
            logger.warning(f"Client {client_id} has no data.")

        save_path = output_dir / f"client_{client_id}_train.pt"
        save_as_torch(X_client, y_client, save_path)

    open_features = np.copy(X_test_closed)
    open_labels = np.copy(y_test_closed)

    if len(df_unknown) > 0:
        logger.info("Processing UNKNOWN data for open-set evaluation...")
        num_scaled_unknown = scaler.transform(df_unknown[num_cols])
        cat_encoded_unknown = encoder.transform(df_unknown[cat_cols])

        features_unknown = np.concatenate(
            [num_scaled_unknown, cat_encoded_unknown],
            axis=1,
        ).astype(np.float32)
        labels_unknown = np.full(len(df_unknown), -1, dtype=np.int64)

        open_features = np.concatenate([open_features, features_unknown], axis=0)
        open_labels = np.concatenate([open_labels, labels_unknown], axis=0)
        logger.info(
            "Appended %d unknown samples to open-set split (total=%d).",
            len(labels_unknown),
            len(open_labels),
        )
    else:
        logger.warning("No 'Unknown' samples found; open_set_test will contain only closed-set data.")

    logger.info("--- ?? Open-Set Evaluation Data (open_set_test.pt) ---")
    logger.info(f"Shape: {open_features.shape}")
    if len(open_labels) > 0:
        unique_labels, counts = np.unique(open_labels, return_counts=True)
        label_dist_str = ", ".join(
            [f"'{idx_to_label.get(l, 'Unknown')}' (ID {l}): {c}" for l, c in zip(unique_labels, counts)]
        )
        logger.info(f"Label Distribution: {label_dist_str}")
    save_as_torch(open_features, open_labels, output_dir / "open_set_test.pt")
    logger.info("--- ? Preprocessing Pipeline FINISHED ---")
    print("\n" + "=" * 60)
    print("  ACTION REQUIRED: Update 'conf/config_fl.yaml' file!")
    print("  Copy these values into the 'env_metadata' section:")
    print(f"\n    state_dim: {state_dim}")
    print(f"    num_actions: {num_actions}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_preprocessing()

