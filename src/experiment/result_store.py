"""Canonical local scientific result store for FedTROS-PR.

The ResultStore is deliberately separate from the interactive experiment tracker.
It is the durable scientific source of truth used for reproducibility, aggregation,
and publication export.  The default tabular contract is CSV to avoid requiring a
Parquet engine on every execution host; callers may still save arbitrary artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf


class ResultStore:
    """Persist canonical run outputs independently of W&B or any other tracker."""

    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.config_dir = self.run_dir / "config"
        self.data_dir = self.run_dir / "data"
        self.logs_dir = self.run_dir / "logs"
        self.metrics_dir = self.run_dir / "metrics"
        self.predictions_dir = self.run_dir / "predictions"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metadata_dir = self.run_dir / "metadata"
        for directory in (
            self.config_dir,
            self.data_dir,
            self.logs_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.artifacts_dir,
            self.checkpoints_dir,
            self.metadata_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._round_records: list[dict[str, Any]] = []
        self._load_existing_round_records()

    def _load_existing_round_records(self) -> None:
        path = self.metrics_dir / "round_metrics.csv"
        if not path.exists():
            legacy = self.metrics_dir / "metrics_round.csv"
            path = legacy if legacy.exists() else path
        if path.exists():
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    self._round_records = frame.where(pd.notna(frame), None).to_dict("records")
            except Exception:
                # A damaged historic metrics file must not prevent the run from starting;
                # the caller will still have the original file for forensic inspection.
                self._round_records = []

    @staticmethod
    def _json_payload(value: Any) -> str:
        return json.dumps(value, indent=2, sort_keys=True, default=str)

    def save_config(self, config: DictConfig | dict[str, Any]) -> None:
        """Persist raw and resolved configuration in canonical locations."""
        if isinstance(config, DictConfig):
            raw = OmegaConf.to_yaml(config, resolve=False)
            resolved = OmegaConf.to_yaml(config, resolve=True)
            (self.config_dir / "raw_config.yaml").write_text(raw, encoding="utf-8")
            (self.config_dir / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
            (self.run_dir / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
        else:
            payload = self._json_payload(config)
            (self.config_dir / "config.json").write_text(payload, encoding="utf-8")

    def save_resume_config(
        self,
        config: DictConfig | dict[str, Any],
        *,
        resumed_from_round: int,
    ) -> Path:
        """Persist resume-time runtime overrides without rewriting the frozen run config.

        Exact continuation necessarily changes operational values such as the number of
        Flower rounds remaining.  Those values are evidence about *how continuation was
        executed*, not a new scientific experiment definition.  The original
        ``resolved_config.yaml`` therefore stays immutable and a dedicated resume config
        is written beside it.
        """
        stem = f"resume_from_round_{int(resumed_from_round):04d}"
        if isinstance(config, DictConfig):
            path = self.config_dir / f"{stem}.yaml"
            path.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
            return path
        path = self.config_dir / f"{stem}.json"
        path.write_text(self._json_payload(config), encoding="utf-8")
        return path

    def append_round_metrics(self, metrics: dict[str, Any], step: int | None = None) -> Path:
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **({"round": int(step)} if step is not None else {}),
            **metrics,
        }
        self._round_records.append(record)
        frame = pd.DataFrame(self._round_records)
        path = self.metrics_dir / "round_metrics.csv"
        frame.to_csv(path, index=False)
        return path

    def save_final_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.metrics_dir / "final_metrics.json"
        path.write_text(self._json_payload(metrics), encoding="utf-8")
        return path

    def save_client_metrics(self, records: Iterable[dict[str, Any]] | pd.DataFrame) -> Path | None:
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
        if frame.empty:
            return None
        path = self.metrics_dir / "client_metrics.csv"
        frame.to_csv(path, index=False)
        return path

    def save_class_metrics(self, records: Iterable[dict[str, Any]] | pd.DataFrame) -> Path | None:
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
        if frame.empty:
            return None
        path = self.metrics_dir / "class_metrics.csv"
        frame.to_csv(path, index=False)
        return path

    def save_dataframe(self, relative_path: str | Path, frame: pd.DataFrame) -> Path:
        path = self.run_dir / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
        else:
            if path.suffix.lower() not in {".csv", ".tsv"}:
                path = path.with_suffix(".csv")
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            frame.to_csv(path, index=False, sep=sep)
        return path

    def save_predictions(self, frame: pd.DataFrame, filename: str = "predictions.csv") -> Path:
        return self.save_dataframe(self.predictions_dir.relative_to(self.run_dir) / filename, frame)

    def save_open_set_scores(self, frame: pd.DataFrame) -> Path:
        return self.save_dataframe("predictions/open_set_scores.csv", frame)

    def save_numpy(self, relative_path: str | Path, array: np.ndarray) -> Path:
        path = self.run_dir / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".npz":
            np.savez_compressed(path, data=array)
        else:
            if path.suffix.lower() != ".npy":
                path = path.with_suffix(".npy")
            np.save(path, array)
        return path

    def write_json(self, filename: str | Path, payload: dict[str, Any] | list[Any]) -> Path:
        """Compatibility-safe JSON writer.

        A bare filename is written at the run root because several scientific modules
        historically expect those files there. New infrastructure should prefer explicit
        subdirectories such as ``metadata/foo.json`` or ``artifacts/foo.json``.
        """
        path = self.run_dir / Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._json_payload(payload), encoding="utf-8")
        return path

    def copy_artifact(self, source: str | Path, relative_target: str | Path | None = None) -> Path:
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(src)
        target = self.artifacts_dir / (Path(relative_target) if relative_target else src.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
        return target

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def artifact_inventory(self) -> dict[str, dict[str, Any]]:
        inventory: dict[str, dict[str, Any]] = {}
        for path in sorted(self.run_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.run_dir).as_posix()
            # Logs remain operationally useful but are append-only until the final
            # logger flush.  Manifests are rewritten during finalization.  Neither
            # belongs in an immutable scientific-artifact checksum inventory.
            if rel.startswith("logs/") or path.name in {"result_manifest.json", "run_manifest.json"}:
                continue
            inventory[rel] = {
                "size_bytes": path.stat().st_size,
                "sha256": self._sha256(path),
            }
        return inventory

    def finalize_result_manifest(
        self,
        *,
        status: str,
        final_metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "schema_name": "fedtros_run_results",
            "schema_version": 3,
            "run_id": self.run_id,
            "status": str(status),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tabular_format": "csv",
            "final_metrics": final_metrics or {},
            "artifacts": self.artifact_inventory(),
        }
        if extra:
            payload.update(extra)
        path = self.metadata_dir / "result_manifest.json"
        path.write_text(self._json_payload(payload), encoding="utf-8")
        (self.run_dir / "result_manifest.json").write_text(
            self._json_payload(payload), encoding="utf-8"
        )
        return path
