from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def _effective_max_samples_per_class(cfg: DictConfig) -> int | None:
    configured = getattr(cfg, "max_samples_per_class", None)
    cap = int(configured) if configured is not None else None
    if bool(getattr(cfg, "smoke", False)):
        smoke_cap = int(getattr(cfg, "smoke_max_samples_per_class", 96))
        if smoke_cap <= 0:
            raise ValueError("smoke_max_samples_per_class must be positive")
        cap = smoke_cap if cap is None else min(cap, smoke_cap)
    return cap


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
    numeric = frame[numerical].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.to_numpy(dtype=np.float64)


def _categorical_frame(
    frame: pd.DataFrame,
    categorical: list[str],
    *,
    allowed_categories: list[set[str]] | None = None,
) -> pd.DataFrame:
    if not categorical:
        return pd.DataFrame(index=frame.index)
    result = frame[categorical].fillna("__UNK__").astype(str).copy()
    if allowed_categories is not None:
        for column, allowed in zip(categorical, allowed_categories, strict=True):
            result.loc[~result[column].isin(allowed), column] = "__UNK__"
    return result




def _categorical_schema_frame(
    *,
    df: pd.DataFrame,
    df_known: pd.DataFrame,
    train_df: pd.DataFrame,
    categorical: list[str],
    cfg: DictConfig,
) -> pd.DataFrame:
    """Return the frame used only to define one-hot categorical schema.

    Numeric scalers are still fitted on known training data only.  This helper
    controls the vocabulary of categorical one-hot columns so label-wise
    open-set runs do not silently drop columns when a held-out label removes
    one categorical value from the known training split.
    """
    if not categorical:
        return pd.DataFrame(index=train_df.index)

    scope = str(getattr(cfg, "categorical_schema_scope", "known_train")).lower()
    if scope in {"known_train", "train", "known-train"}:
        schema_df = train_df
    elif scope in {"known", "known_all", "known-all"}:
        schema_df = df_known
    elif scope in {"source", "all", "all_source", "all-source"}:
        schema_df = df
    else:
        raise ValueError(
            "dataset.preprocessing.categorical_schema_scope must be one of "
            "'known_train', 'known', or 'source'."
        )

    logger.info(
        "Categorical schema scope | scope=%s | rows=%d | categorical_cols=%d",
        scope,
        len(schema_df),
        len(categorical),
    )
    return _categorical_frame(schema_df, categorical)


def _transform_features(
    frame: pd.DataFrame,
    *,
    numerical: list[str],
    categorical: list[str],
    imputer: SimpleImputer | None,
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

    numeric_raw = _numeric_array(frame, numerical)
    numeric_imputed = imputer.transform(numeric_raw) if imputer else numeric_raw
    numeric_part = scaler.transform(numeric_imputed) if scaler else numeric_imputed
    allowed_categories = (
        [set(str(value) for value in values.tolist()) for values in encoder.categories_]
        if encoder
        else None
    )
    categorical_part = (
        encoder.transform(
            _categorical_frame(
                frame,
                categorical,
                allowed_categories=allowed_categories,
            )
        )
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
    min_samples_per_client: int = 1,
    max_attempts: int = 100,
) -> dict[int, list[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if labels.shape[0] < num_clients:
        raise ValueError(
            f"Cannot create {num_clients} non-empty clients from {labels.shape[0]} training samples."
        )

    if min_samples_per_client <= 0:
        raise ValueError("min_samples_per_client must be positive.")
    if labels.shape[0] < num_clients * min_samples_per_client:
        raise ValueError(
            f"Cannot allocate at least {min_samples_per_client} samples to {num_clients} "
            f"clients from {labels.shape[0]} training samples."
        )

    client_indices = {client_id: [] for client_id in range(num_clients)}
    if iid:
        shuffled = rng.permutation(np.arange(labels.shape[0]))
        for client_id, shard in enumerate(np.array_split(shuffled, num_clients)):
            client_indices[client_id].extend(int(idx) for idx in shard)
        return client_indices

    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive when iid=false.")

    for _ in range(max_attempts):
        client_indices = {client_id: [] for client_id in range(num_clients)}
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            shards = np.split(class_indices, split_points)
            for client_id, shard in enumerate(shards):
                client_indices[client_id].extend(int(idx) for idx in shard)
        if min(len(indices) for indices in client_indices.values()) >= min_samples_per_client:
            return client_indices

    sizes = {client_id: len(indices) for client_id, indices in client_indices.items()}
    raise ValueError(
        "Unable to satisfy the minimum client size after deterministic Dirichlet retries: "
        f"minimum={min_samples_per_client}, attempts={max_attempts}, sizes={sizes}."
    )


def _label_vector_sha256(labels: np.ndarray) -> str:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _validate_paired_partition_payload(
    payload: dict[str, Any],
    *,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    iid: bool,
    seed: int,
    known_labels: list[str],
    unknown_labels: list[str],
) -> dict[int, list[int]]:
    """Validate an immutable paired partition before reusing it across methods."""
    if payload.get("schema_name") != "fedtros_paired_partition" or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported paired-partition schema")
    expected = {
        "num_clients": int(num_clients),
        "seed": int(seed),
        "iid": bool(iid),
        "train_size": int(len(labels)),
        "train_labels_sha256": _label_vector_sha256(labels),
        "known_labels": [str(x) for x in known_labels],
        "unknown_labels": [str(x) for x in unknown_labels],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Paired partition mismatch for {key}: stored={payload.get(key)!r}, expected={value!r}"
            )
    if not iid and abs(float(payload.get("alpha", -1.0)) - float(alpha)) > 1e-12:
        raise ValueError(
            f"Paired partition alpha mismatch: stored={payload.get('alpha')}, expected={alpha}"
        )
    raw_clients = payload.get("client_indices")
    if not isinstance(raw_clients, dict):
        raise ValueError("Paired partition has no client_indices mapping")
    clients: dict[int, list[int]] = {}
    flattened: list[int] = []
    for cid in range(num_clients):
        raw = raw_clients.get(str(cid), raw_clients.get(cid))
        if raw is None:
            raise ValueError(f"Paired partition missing client {cid}")
        indices = [int(x) for x in raw]
        if not indices:
            raise ValueError(f"Paired partition client {cid} is empty")
        if min(indices) < 0 or max(indices) >= len(labels):
            raise ValueError(f"Paired partition client {cid} contains out-of-range indices")
        clients[cid] = indices
        flattened.extend(indices)
    if len(flattened) != len(labels) or len(set(flattened)) != len(labels):
        raise ValueError(
            "Paired partition must assign every known-training sample exactly once "
            f"(assigned={len(flattened)}, unique={len(set(flattened))}, expected={len(labels)})"
        )
    return clients


def load_or_create_paired_partition(
    *,
    partition_path: Path | None,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    num_classes: int,
    rng: np.random.Generator,
    iid: bool,
    min_samples_per_client: int,
    max_attempts: int,
    seed: int,
    known_labels: list[str],
    unknown_labels: list[str],
) -> dict[int, list[int]]:
    """Reuse one seed/condition partition across all matched methods.

    The persisted relative indices refer to the deterministic known-training split.
    A label-vector hash and the full known/unknown protocol prevent accidental reuse
    across E3/E4/E8 populations that happen to share dataset/alpha/seed values.
    """
    if partition_path is not None and partition_path.exists():
        payload = json.loads(partition_path.read_text(encoding="utf-8"))
        clients = _validate_paired_partition_payload(
            payload,
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            iid=iid,
            seed=seed,
            known_labels=known_labels,
            unknown_labels=unknown_labels,
        )
        logger.info("Reusing paired client partition: %s", partition_path)
        return clients

    clients = dirichlet_split(
        labels,
        num_clients=num_clients,
        alpha=alpha,
        num_classes=num_classes,
        rng=rng,
        iid=iid,
        min_samples_per_client=min_samples_per_client,
        max_attempts=max_attempts,
    )
    if partition_path is None:
        return clients

    partition_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": "fedtros_paired_partition",
        "schema_version": 1,
        "dataset_protocol": {
            "known_labels": [str(x) for x in known_labels],
            "unknown_labels": [str(x) for x in unknown_labels],
        },
        "known_labels": [str(x) for x in known_labels],
        "unknown_labels": [str(x) for x in unknown_labels],
        "seed": int(seed),
        "alpha": float(alpha),
        "iid": bool(iid),
        "num_clients": int(num_clients),
        "train_size": int(len(labels)),
        "train_labels_sha256": _label_vector_sha256(labels),
        "client_indices": {str(cid): [int(x) for x in indices] for cid, indices in clients.items()},
    }
    tmp = partition_path.with_name(f".{partition_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, partition_path)
    logger.info("Created paired client partition: %s", partition_path)
    return clients


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


def _prepare_raw_frame(
    df: pd.DataFrame,
    *,
    label_col: str,
    cfg: DictConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Apply deterministic, label-preserving cleanup before any split."""
    drop_columns = [str(value) for value in list(getattr(cfg, "drop_columns", []) or [])]
    present_drop_columns = [column for column in drop_columns if column in df.columns]
    if present_drop_columns:
        df = df.drop(columns=present_drop_columns)

    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip()
    before_counts = {
        str(label): int(count) for label, count in df[label_col].value_counts().sort_index().items()
    }

    if bool(getattr(cfg, "drop_duplicates", False)):
        df = df.drop_duplicates(keep="first")

    source_labels = [str(value) for value in list(getattr(cfg, "source_labels", []) or [])]
    if source_labels:
        actual_labels = set(str(value) for value in df[label_col].dropna().unique().tolist())
        unexpected = sorted(actual_labels - set(source_labels))
        missing = sorted(set(source_labels) - actual_labels)
        if unexpected:
            raise ValueError(f"Raw data contains labels outside dataset.source_labels: {unexpected}")
        if missing and bool(getattr(cfg, "require_all_source_labels", False)):
            raise ValueError(f"Raw data is missing configured source labels: {missing}")

    max_per_class = _effective_max_samples_per_class(cfg)
    if max_per_class is not None and int(max_per_class) > 0:
        capped_parts: list[pd.DataFrame] = []
        for _, group in df.groupby(label_col, sort=True):
            if len(group) > int(max_per_class):
                group = group.sample(n=int(max_per_class), random_state=seed, replace=False)
            capped_parts.append(group)
        df = pd.concat(capped_parts, axis=0).sort_index()

    after_counts = {
        str(label): int(count) for label, count in df[label_col].value_counts().sort_index().items()
    }
    return df, before_counts, after_counts


def _categorical_fit_frame(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    prepared = _categorical_frame(frame, categorical)
    if not categorical:
        return prepared
    unknown_row = pd.DataFrame(
        [{column: "__UNK__" for column in categorical}],
        columns=categorical,
    )
    return pd.concat([prepared, unknown_row], ignore_index=True)


def _read_raw_csv(
    path: Path,
    *,
    cfg: DictConfig,
    label_col: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int] | None]:
    """Read normally or keep deterministic bottom-k rows per class in streaming mode."""
    chunksize = getattr(cfg, "read_chunksize", None)
    max_per_class = _effective_max_samples_per_class(cfg)
    if chunksize is None or max_per_class is None or int(max_per_class) <= 0:
        return pd.read_csv(path, low_memory=False), None

    reservoirs: dict[str, pd.DataFrame] = {}
    source_counts: Counter[str] = Counter()
    for chunk in pd.read_csv(path, chunksize=int(chunksize), low_memory=False):
        if label_col not in chunk.columns:
            raise KeyError(f"Label column {label_col!r} not found in {path}.")
        chunk[label_col] = chunk[label_col].astype(str).str.strip()
        source_counts.update(str(value) for value in chunk[label_col])
        for label, group in chunk.groupby(label_col, sort=False):
            label = str(label)
            candidate = pd.concat([reservoirs.get(label, group.iloc[0:0]), group], axis=0)
            if bool(getattr(cfg, "drop_duplicates", False)):
                candidate = candidate.drop_duplicates(keep="first")
            if len(candidate) > int(max_per_class):
                hashes = pd.util.hash_pandas_object(candidate, index=False).to_numpy(dtype=np.uint64)
                salted = hashes ^ np.uint64(seed)
                positions = np.argpartition(salted, int(max_per_class) - 1)[
                    : int(max_per_class)
                ]
                candidate = candidate.iloc[np.sort(positions)]
            reservoirs[label] = candidate
    if not reservoirs:
        raise ValueError(f"Raw CSV contains no rows: {path}")
    return pd.concat(reservoirs.values(), ignore_index=True), dict(sorted(source_counts.items()))


def run_preprocessing(cfg: DictConfig, *, project_root: Path) -> dict[str, Any]:
    p_cfg = cfg.dataset.preprocessing
    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)

    output_dir = resolve_path(project_root, p_cfg.output_dir)
    raw_file = resolve_path(project_root, p_cfg.raw_file)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")

    label_col = str(p_cfg.label_column)
    df, streamed_source_counts = _read_raw_csv(
        raw_file,
        cfg=p_cfg,
        label_col=label_col,
        seed=seed,
    )
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col!r} not found in {raw_file}.")

    df, source_class_counts, experiment_class_counts = _prepare_raw_frame(
        df,
        label_col=label_col,
        cfg=p_cfg,
        seed=seed,
    )
    if streamed_source_counts is not None:
        source_class_counts = streamed_source_counts

    numerical, categorical = _feature_columns(df, p_cfg)
    known_labels = list(p_cfg.known_labels)
    configured_unknown_labels = [
        str(value) for value in list(getattr(p_cfg, "unknown_labels", []) or [])
    ]
    source_labels = [str(value) for value in list(getattr(p_cfg, "source_labels", []) or [])]
    if source_labels:
        complement = [label for label in source_labels if label not in known_labels]
        if configured_unknown_labels and set(complement) != set(configured_unknown_labels):
            raise ValueError(
                "Configured unknown_labels must exactly match source_labels minus known_labels: "
                f"expected={complement}, configured={configured_unknown_labels}"
            )
    label_map = {label: idx for idx, label in enumerate(known_labels)}
    idx_to_label = {idx: label for label, idx in label_map.items()}
    num_classes = len(label_map)
    if num_classes == 0:
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

    imputer = SimpleImputer(strategy="median", keep_empty_features=True) if numerical else None
    scaler = MinMaxScaler() if numerical else None
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if categorical else None
    numeric_train = _numeric_array(train_df, numerical)
    if imputer:
        imputer.fit(numeric_train)
    if scaler:
        scaler.fit(imputer.transform(numeric_train) if imputer else numeric_train)
    if encoder:
        schema_frame = (
            _categorical_schema_frame(
                df=df,
                df_known=df_known,
                train_df=train_df,
                categorical=categorical,
                cfg=p_cfg,
            )
        )
        encoder.fit(_categorical_fit_frame(schema_frame, categorical))

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = _transform_features(
        train_df,
        numerical=numerical,
        categorical=categorical,
        imputer=imputer,
        scaler=scaler,
        encoder=encoder,
    )
    X_val = _transform_features(
        val_df,
        numerical=numerical,
        categorical=categorical,
        imputer=imputer,
        scaler=scaler,
        encoder=encoder,
    )
    X_test = _transform_features(
        test_df,
        numerical=numerical,
        categorical=categorical,
        imputer=imputer,
        scaler=scaler,
        encoder=encoder,
    )

    _save_tensor_dataset(X_train, train_y, output_dir / "known_train.pt")
    _save_tensor_dataset(X_val, val_y, output_dir / "validation.pt")
    _save_tensor_dataset(X_test, test_y, output_dir / "closed_set_test.pt")
    _save_tensor_dataset(X_test, test_y, output_dir / "shared_closed_set_test.pt")

    num_clients = int(p_cfg.num_clients)
    partition_value = OmegaConf.select(cfg, "dataset.partition_file", default=None)
    partition_path: Path | None = None
    if partition_value:
        partition_path = Path(str(partition_value))
        if not partition_path.is_absolute():
            partition_path = project_root / partition_path
    client_indices = load_or_create_paired_partition(
        partition_path=partition_path,
        labels=train_y,
        num_clients=num_clients,
        alpha=float(p_cfg.alpha),
        num_classes=num_classes,
        rng=rng,
        iid=bool(p_cfg.iid),
        min_samples_per_client=(
            int(getattr(p_cfg, "smoke_min_samples_per_client", 8))
            if bool(getattr(p_cfg, "smoke", False))
            else int(getattr(p_cfg, "min_samples_per_client", 1))
        ),
        max_attempts=int(getattr(p_cfg, "partition_max_attempts", 100)),
        seed=seed,
        known_labels=known_labels,
        unknown_labels=configured_unknown_labels,
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
        counts = np.bincount(y_client, minlength=num_classes)[:num_classes]
        class_distribution_rows.append(
            {
                "client_id": client_id,
                **{
                    idx_to_label[class_id]: int(counts[class_id]) for class_id in range(num_classes)
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

    max_unknown_ratio = getattr(p_cfg, "max_unknown_test_ratio", None)
    if (
        not df_unknown.empty
        and max_unknown_ratio is not None
        and float(max_unknown_ratio) >= 0.0
    ):
        max_unknown = int(len(test_df) * float(max_unknown_ratio))
        if len(df_unknown) > max_unknown:
            if max_unknown <= 0:
                df_unknown = df_unknown.iloc[0:0].copy()
            else:
                counts = df_unknown[label_col].value_counts()
                stratify = (
                    df_unknown[label_col]
                    if max_unknown >= len(counts) and int(counts.min()) >= 2
                    else None
                )
                df_unknown, _ = train_test_split(
                    df_unknown,
                    train_size=max_unknown,
                    stratify=stratify,
                    random_state=seed,
                )

    open_features = X_test
    open_labels = test_y
    if not df_unknown.empty:
        X_unknown = _transform_features(
            df_unknown,
            numerical=numerical,
            categorical=categorical,
            imputer=imputer,
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
    pd.DataFrame(
        [
            {
                "label": label,
                "source_count": int(source_class_counts.get(label, 0)),
                "experiment_count": int(experiment_class_counts.get(label, 0)),
                "role": "known" if label in known_labels else "unknown",
            }
            for label in (source_labels or sorted(experiment_class_counts))
        ]
    ).to_csv(output_dir / "class_support.csv", index=False)
    split_rows: list[dict[str, Any]] = []
    for split_name, frame in (
        ("known_train", train_df),
        ("validation", val_df),
        ("known_test", test_df),
        ("unknown_test", df_unknown),
    ):
        split_rows.extend(
            {
                "source_index": int(source_index),
                "split": split_name,
                "label": str(row[label_col]),
            }
            for source_index, row in frame.iterrows()
        )
    pd.DataFrame(split_rows).to_csv(output_dir / "split_manifest.csv", index=False)

    feature_names = list(numerical)
    if encoder:
        feature_names.extend(str(value) for value in encoder.get_feature_names_out(categorical))
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_dim": len(feature_names),
                "feature_names": feature_names,
                "numerical_columns": numerical,
                "categorical_columns": categorical,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if imputer:
        joblib.dump(imputer, output_dir / "numeric_imputer.joblib")
    if scaler:
        joblib.dump(scaler, output_dir / "scaler.joblib")
    if encoder:
        joblib.dump(encoder, output_dir / "encoder.joblib")

    feature_dim = int(X_train.shape[1])
    for split_name, features in (
        ("known_train", X_train),
        ("validation", X_val),
        ("known_test", X_test),
        ("open_test", open_features),
    ):
        if not np.isfinite(features).all():
            raise RuntimeError(f"Non-finite feature values remain in {split_name} after preprocessing.")
    expected_feature_dim = getattr(p_cfg, "expected_feature_dim", None)
    if expected_feature_dim is not None and int(expected_feature_dim) != feature_dim:
        raise RuntimeError(
            f"Preprocessed feature_dim={feature_dim}, but "
            f"dataset.preprocessing.expected_feature_dim={int(expected_feature_dim)}. "
            "This usually means the categorical schema changed. Use "
            "categorical_schema_scope=source for the fixed BNaT schema, or update "
            "the expected_feature_dim if the raw feature schema intentionally changed."
        )
    metadata = {
        "dataset": str(cfg.dataset.name),
        "raw_file": str(raw_file),
        "seed": seed,
        "label_column": label_col,
        "known_labels": known_labels,
        "unknown_labels": sorted(str(value) for value in df_unknown[label_col].dropna().unique().tolist()),
        "configured_unknown_labels": configured_unknown_labels,
        "source_class_counts": source_class_counts,
        "experiment_class_counts": experiment_class_counts,
        "open_test_unknown_class_counts": {
            str(label): int(count)
            for label, count in df_unknown[label_col].value_counts().sort_index().items()
        },
        "unknown_label_id": int(p_cfg.unknown_label_id),
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "categorical_schema_scope": str(getattr(p_cfg, "categorical_schema_scope", "known_train")),
        "categorical_category_counts": (
            [int(len(categories)) for categories in encoder.categories_] if encoder else []
        ),
        "categorical_categories": (
            [[str(value) for value in categories.tolist()] for categories in encoder.categories_]
            if encoder
            else []
        ),
        "num_known_samples": len(df_known),
        "num_unknown_samples": len(df_unknown),
        "num_train_samples": len(train_y),
        "num_validation_samples": len(val_y),
        "num_closed_test_samples": len(test_y),
        "num_open_test_samples": len(open_labels),
        "num_clients": num_clients,
        "min_samples_per_client": (
            int(getattr(p_cfg, "smoke_min_samples_per_client", 8))
            if bool(getattr(p_cfg, "smoke", False))
            else int(getattr(p_cfg, "min_samples_per_client", 1))
        ),
        "alpha": float(p_cfg.alpha),
        "iid": bool(p_cfg.iid),
        "config_snapshot": OmegaConf.to_container(cfg.dataset, resolve=True),
    }
    metadata_path = output_dir / "preprocess_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        "Preprocessing complete | feature_dim=%d | num_classes=%d | output=%s",
        feature_dim,
        num_classes,
        output_dir,
    )
    return metadata
