from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    label_col: str,
    numeric_threshold: float,
) -> tuple[list[str], list[str]]:
    feature_cols = [col for col in df.columns if col != label_col]
    if not feature_cols:
        raise ValueError("No feature columns remain after excluding the label column.")

    numerical: list[str] = []
    categorical: list[str] = []
    for col in feature_cols:
        series = df[col]
        coerced = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        numeric_frac = coerced.notna().sum() / max(non_null, 1) if non_null else 0.0
        if numeric_frac >= numeric_threshold:
            numerical.append(col)
        else:
            categorical.append(col)
    return numerical, categorical


def _feature_columns(df: pd.DataFrame, cfg: DictConfig) -> tuple[list[str], list[str]]:
    if cfg.numerical_cols is not None or cfg.categorical_cols is not None:
        numerical = list(cfg.numerical_cols or [])
        categorical = list(cfg.categorical_cols or [])
        missing = [col for col in numerical + categorical if col not in df.columns]
        if missing:
            raise KeyError(f"Configured feature columns are missing from raw data: {missing}")
        return numerical, categorical
    return infer_feature_columns(
        df,
        label_col=str(cfg.label_column),
        numeric_threshold=float(cfg.numeric_threshold),
    )


def _numeric_array(frame: pd.DataFrame, numerical: list[str]) -> np.ndarray:
    if not numerical:
        return np.empty((len(frame), 0), dtype=np.float32)
    return (
        frame[numerical]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _categorical_frame(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    if not categorical:
        return pd.DataFrame(index=frame.index)
    return frame[categorical].fillna("UNK").astype(str)


def _transform_features(
    frame: pd.DataFrame,
    *,
    numerical: list[str],
    categorical: list[str],
    scaler: MinMaxScaler | None,
    encoder: OneHotEncoder | None,
) -> np.ndarray:
    if frame.empty:
        categorical_dim = (
            sum(len(categories) for categories in encoder.categories_) if encoder else 0
        )
        feature_dim = len(numerical) + categorical_dim
        if feature_dim <= 0:
            raise RuntimeError("No usable feature columns found after preprocessing.")
        return np.empty((0, feature_dim), dtype=np.float32)

    numeric_part = (
        scaler.transform(_numeric_array(frame, numerical))
        if scaler
        else _numeric_array(frame, numerical)
    )
    categorical_part = (
        encoder.transform(_categorical_frame(frame, categorical))
        if encoder
        else np.empty((len(frame), 0), dtype=np.float32)
    )
    parts = [part for part in (numeric_part, categorical_part) if part.shape[1] > 0]
    if not parts:
        raise RuntimeError("No usable feature columns found after preprocessing.")
    return np.concatenate(parts, axis=1).astype(np.float32)


def _save_tensor_dataset(features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        },
        path,
    )
    logger.info("Saved tensor dataset | path=%s | samples=%d", path, len(labels))


def dirichlet_split(
    labels: np.ndarray,
    *,
    num_clients: int,
    alpha: float,
    num_classes: int,
    rng: np.random.Generator,
    iid: bool,
) -> dict[int, list[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if labels.shape[0] < num_clients:
        raise ValueError(
            f"Cannot create {num_clients} non-empty clients from {labels.shape[0]} training samples."
        )

    client_indices = {client_id: [] for client_id in range(num_clients)}
    if iid:
        shuffled = rng.permutation(np.arange(labels.shape[0]))
        for client_id, shard in enumerate(np.array_split(shuffled, num_clients)):
            client_indices[client_id].extend(int(idx) for idx in shard)
        return client_indices

    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive when iid=false.")

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        shards = np.split(class_indices, split_points)
        for client_id, shard in enumerate(shards):
            client_indices[client_id].extend(int(idx) for idx in shard)
    return _ensure_non_empty_clients(client_indices, rng=rng)


def _ensure_non_empty_clients(
    client_indices: dict[int, list[int]], *, rng: np.random.Generator
) -> dict[int, list[int]]:
    empty_clients = [client_id for client_id, indices in client_indices.items() if not indices]
    if not empty_clients:
        return client_indices

    donor_clients = [client_id for client_id, indices in client_indices.items() if len(indices) > 1]
    for empty_client in empty_clients:
        if not donor_clients:
            raise ValueError("Cannot rebalance split: no donor clients have more than one sample.")
        donor = int(rng.choice(donor_clients))
        donor_indices = client_indices[donor]
        moved_position = int(rng.integers(0, len(donor_indices)))
        client_indices[empty_client].append(donor_indices.pop(moved_position))
        if len(donor_indices) <= 1:
            donor_clients.remove(donor)

    return client_indices


def _split_known_data(
    df_known: pd.DataFrame,
    labels: np.ndarray,
    *,
    closed_set_test_size: float,
    validation_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    try:
        train_val_df, test_df, train_val_y, test_y = train_test_split(
            df_known,
            labels,
            test_size=closed_set_test_size,
            stratify=labels,
            random_state=seed,
        )
    except ValueError as exc:
        logger.warning(
            "Stratified known/test split failed (%s). Falling back to random split.", exc
        )
        train_val_df, test_df, train_val_y, test_y = train_test_split(
            df_known,
            labels,
            test_size=closed_set_test_size,
            random_state=seed,
        )

    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError("dataset.preprocessing.validation_split must be in [0, 1).")
    if validation_split == 0.0:
        empty_val_df = train_val_df.iloc[0:0].copy()
        empty_val_y = np.empty(0, dtype=train_val_y.dtype)
        return train_val_df, empty_val_df, test_df, train_val_y, empty_val_y, test_y

    try:
        train_df, val_df, train_y, val_y = train_test_split(
            train_val_df,
            train_val_y,
            test_size=validation_split,
            stratify=train_val_y,
            random_state=seed,
        )
    except ValueError as exc:
        logger.warning(
            "Stratified train/validation split failed (%s). Falling back to random split.", exc
        )
        train_df, val_df, train_y, val_y = train_test_split(
            train_val_df,
            train_val_y,
            test_size=validation_split,
            random_state=seed,
        )
    return train_df, val_df, test_df, train_y, val_y, test_y


def run_preprocessing(cfg: DictConfig, *, project_root: Path) -> dict[str, Any]:
    p_cfg = cfg.dataset.preprocessing
    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)

    output_dir = resolve_path(project_root, p_cfg.output_dir)
    raw_file = resolve_path(project_root, p_cfg.raw_file)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")

    df = pd.read_csv(raw_file, low_memory=False)
    label_col = str(p_cfg.label_column)
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col!r} not found in {raw_file}.")

    numerical, categorical = _feature_columns(df, p_cfg)
    known_labels = list(p_cfg.known_labels)
    label_map = {label: idx for idx, label in enumerate(known_labels)}
    idx_to_label = {idx: label for label, idx in label_map.items()}
    num_actions = len(label_map)
    if num_actions == 0:
        raise ValueError("dataset.preprocessing.known_labels must not be empty.")

    df_known = df[df[label_col].isin(known_labels)].copy()
    df_unknown = df[~df[label_col].isin(known_labels)].copy()
    if df_known.empty:
        raise ValueError("No known-label samples found. Check known_labels and label_column.")

    labels_known = df_known[label_col].map(label_map).to_numpy(dtype=np.int64)
    train_df, val_df, test_df, train_y, val_y, test_y = _split_known_data(
        df_known,
        labels_known,
        closed_set_test_size=float(p_cfg.closed_set_test_size),
        validation_split=float(p_cfg.validation_split),
        seed=seed,
    )

    scaler = MinMaxScaler() if numerical else None
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if categorical else None
    if scaler:
        scaler.fit(_numeric_array(train_df, numerical))
    if encoder:
        encoder.fit(_categorical_frame(train_df, categorical))

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = _transform_features(
        train_df,
        numerical=numerical,
        categorical=categorical,
        scaler=scaler,
        encoder=encoder,
    )
    X_val = _transform_features(
        val_df,
        numerical=numerical,
        categorical=categorical,
        scaler=scaler,
        encoder=encoder,
    )
    X_test = _transform_features(
        test_df,
        numerical=numerical,
        categorical=categorical,
        scaler=scaler,
        encoder=encoder,
    )

    _save_tensor_dataset(X_train, train_y, output_dir / "known_train.pt")
    _save_tensor_dataset(X_val, val_y, output_dir / "validation.pt")
    _save_tensor_dataset(X_test, test_y, output_dir / "closed_set_test.pt")
    _save_tensor_dataset(X_test, test_y, output_dir / "shared_closed_set_test.pt")

    num_clients = int(p_cfg.num_clients)
    client_indices = dirichlet_split(
        train_y,
        num_clients=num_clients,
        alpha=float(p_cfg.alpha),
        num_classes=num_actions,
        rng=rng,
        iid=bool(p_cfg.iid),
    )
    partition_records: list[dict[str, Any]] = []
    class_distribution_rows: list[dict[str, Any]] = []
    for zero_based_id, indices in client_indices.items():
        client_id = zero_based_id + 1
        client_indices_array = np.asarray(indices, dtype=np.int64)
        rng.shuffle(client_indices_array)
        y_client = train_y[client_indices_array]
        _save_tensor_dataset(
            X_train[client_indices_array],
            y_client,
            output_dir / f"client_{client_id}_train.pt",
        )
        counts = np.bincount(y_client, minlength=num_actions)[:num_actions]
        class_distribution_rows.append(
            {
                "client_id": client_id,
                **{
                    idx_to_label[class_id]: int(counts[class_id]) for class_id in range(num_actions)
                },
            }
        )
        for local_idx in client_indices_array.tolist():
            partition_records.append(
                {
                    "client_id": client_id,
                    "sample_index": int(local_idx),
                    "label": int(train_y[local_idx]),
                    "label_name": idx_to_label[int(train_y[local_idx])],
                    "split": "train",
                    "seed": seed,
                    "alpha": float(p_cfg.alpha),
                }
            )

    open_features = X_test
    open_labels = test_y
    if not df_unknown.empty:
        X_unknown = _transform_features(
            df_unknown,
            numerical=numerical,
            categorical=categorical,
            scaler=scaler,
            encoder=encoder,
        )
        unknown_labels = np.full(len(df_unknown), int(p_cfg.unknown_label_id), dtype=np.int64)
        open_features = np.concatenate([open_features, X_unknown], axis=0)
        open_labels = np.concatenate([open_labels, unknown_labels], axis=0)
    _save_tensor_dataset(open_features, open_labels, output_dir / "open_set_test.pt")
    _save_tensor_dataset(open_features, open_labels, output_dir / "shared_open_set_test.pt")

    class_names_path = output_dir / "class_names.json"
    class_names_path.write_text(
        json.dumps({idx: label for idx, label in idx_to_label.items()}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "partition_manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in partition_records),
        encoding="utf-8",
    )
    pd.DataFrame(class_distribution_rows).to_csv(
        output_dir / "client_class_distribution.csv",
        index=False,
    )
    if scaler:
        joblib.dump(scaler, output_dir / "scaler.joblib")
    if encoder:
        joblib.dump(encoder, output_dir / "encoder.joblib")

    state_dim = int(X_train.shape[1])
    metadata = {
        "dataset": str(cfg.dataset.name),
        "raw_file": str(raw_file),
        "seed": seed,
        "label_column": label_col,
        "known_labels": known_labels,
        "unknown_label_id": int(p_cfg.unknown_label_id),
        "num_actions": num_actions,
        "state_dim": state_dim,
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "num_known_samples": len(df_known),
        "num_unknown_samples": len(df_unknown),
        "num_train_samples": len(train_y),
        "num_validation_samples": len(val_y),
        "num_closed_test_samples": len(test_y),
        "num_open_test_samples": len(open_labels),
        "num_clients": num_clients,
        "alpha": float(p_cfg.alpha),
        "iid": bool(p_cfg.iid),
        "config_snapshot": OmegaConf.to_container(cfg.dataset, resolve=True),
    }
    metadata_path = output_dir / "preprocess_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "Preprocessing complete | state_dim=%d | num_actions=%d | output=%s",
        state_dim,
        num_actions,
        output_dir,
    )
    return metadata
