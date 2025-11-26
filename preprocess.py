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


def _infer_feature_columns(df: pd.DataFrame, label_col: str, numeric_threshold: float = 0.9):
    feature_cols = [c for c in df.columns if c != label_col]
    if not feature_cols:
        raise ValueError("No feature columns remain after excluding the label column.")

    num_cols = []
    cat_cols = []
    for col in feature_cols:
        series = df[col]
        coerced = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        numeric_frac = (coerced.notna().sum() / max(non_null, 1)) if non_null else 0.0
        if numeric_frac >= numeric_threshold:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols


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
        output_dir = resolve_path(p_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_file = resolve_path(p_cfg.raw_file)
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_file}")

        df = pd.read_csv(raw_file, low_memory=False)
        logger.info("Loaded %s with %d rows.", raw_file, len(df))
    except FileNotFoundError as exc:
        logger.critical(f"FATAL: Raw data file not found: {exc}")
        logger.critical("Please download the data and place it in 'data/raw/'.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL: Error loading data or config: {e}", exc_info=True)
        sys.exit(1)

    try:
        label_col = p_cfg.label_column
        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in CSV.")
        num_cols, cat_cols = _infer_feature_columns(df, label_col)
        logger.info(
            "Inferred feature columns | numerical=%d | categorical=%d",
            len(num_cols),
            len(cat_cols),
        )
        logger.debug("Numerical columns: %s", num_cols)
        logger.debug("Categorical columns: %s", cat_cols)
    except Exception as e:
        logger.critical(f"FATAL: Failed to infer columns: {e}")
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
    logger.info(
        "Dataset features inferred: %d numerical, %d categorical, label='%s'",
        len(num_cols),
        len(cat_cols),
        label_col,
    )

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
    scaler = MinMaxScaler() if num_cols else None
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if cat_cols else None

    def _num_array(frame: pd.DataFrame) -> np.ndarray:
        if not num_cols:
            return np.empty((len(frame), 0), dtype=np.float32)
        return (
            frame[num_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

    def _cat_array(frame: pd.DataFrame) -> pd.DataFrame:
        if not cat_cols:
            return pd.DataFrame(index=frame.index)
        return frame[cat_cols].fillna("UNK").astype(str)

    if scaler:
        scaler.fit(_num_array(df_known))
    if encoder:
        encoder.fit(_cat_array(df_known))

    cat_feature_count = encoder.get_feature_names_out().shape[0] if encoder else 0
    state_dim = len(num_cols) + cat_feature_count
    logger.info(f"Scaler fit on {len(num_cols)} numerical features.")
    logger.info(
        f"Encoder fit on {len(cat_cols)} categorical features, resulting in {cat_feature_count} one-hot features."
    )

    logger.info("Processing KNOWN data...")
    num_scaled_known = scaler.transform(_num_array(df_known)) if scaler else _num_array(df_known)
    cat_encoded_known = encoder.transform(_cat_array(df_known)) if encoder else np.empty((len(df_known), 0))
    parts_known = [arr for arr in (num_scaled_known, cat_encoded_known) if arr.shape[1] > 0]
    if not parts_known:
        raise RuntimeError("No usable feature columns found after preprocessing (known split).")
    features_known = np.concatenate(parts_known, axis=1).astype(np.float32)
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
    # Explicit shared alias for clarity across clients
    save_as_torch(X_test_closed, y_test_closed, output_dir / "shared_closed_set_test.pt")

    shard_count = int(getattr(p_cfg, "num_shards", p_cfg.num_clients))
    logger.info(f"Splitting KNOWN training data into {shard_count} client shards...")

    indices = np.random.permutation(len(X_train_known))
    X_train_known, y_train_known = X_train_known[indices], y_train_known[indices]
    client_indices = np.array_split(range(len(X_train_known)), shard_count)

    for i in range(shard_count):
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
        num_scaled_unknown = scaler.transform(_num_array(df_unknown)) if scaler else _num_array(df_unknown)
        cat_encoded_unknown = (
            encoder.transform(_cat_array(df_unknown)) if encoder else np.empty((len(df_unknown), 0))
        )

        parts_unknown = [arr for arr in (num_scaled_unknown, cat_encoded_unknown) if arr.shape[1] > 0]
        if not parts_unknown:
            raise RuntimeError("No usable feature columns found after preprocessing (unknown split).")
        features_unknown = np.concatenate(parts_unknown, axis=1).astype(np.float32)
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
    # Explicit shared alias for clarity across clients
    save_as_torch(open_features, open_labels, output_dir / "shared_open_set_test.pt")
    logger.info("--- ? Preprocessing Pipeline FINISHED ---")
    print("\n" + "=" * 60)
    print("  ACTION REQUIRED: Update 'conf/config_fl.yaml' file!")
    print("  Copy these values into the 'env_metadata' section:")
    print(f"\n    state_dim: {state_dim}")
    print(f"    num_actions: {num_actions}")
    print("\n  Shared eval tensors:")
    print("    closed_set_test.pt  (alias: shared_closed_set_test.pt)")
    print("    open_set_test.pt    (alias: shared_open_set_test.pt)")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_preprocessing()

