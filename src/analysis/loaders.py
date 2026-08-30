"""Load canonical and legacy FedTROS run directories for non-visual analysis/export."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _json(path: Path) -> dict[str, Any]:
    if not path.exists(): return {}
    try:
        value=json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        logger.debug("Could not load %s: %s", path, exc); return {}


def _csv(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            try: return pd.read_csv(path)
            except Exception as exc: logger.debug("Could not load %s: %s", path, exc)
    return pd.DataFrame()


def _first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


@dataclass
class RunRecord:
    run_id: str
    run_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    config: Any = None
    study: str = "unknown"
    stage: str = "unknown"
    method: str = "unknown"
    dataset: str = "unknown"
    alpha: float = 0.0
    iid: bool = False
    seed: int = 0
    num_clients: int = 0
    unknown_labels: list[str] = field(default_factory=list)
    status: str = "UNKNOWN"
    metrics: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    split_hash: str = ""
    git_commit: str = ""
    timestamp_utc: str = ""
    variant: str = "canonical"
    _history_df: pd.DataFrame | None = None
    _scores_df: pd.DataFrame | None = None
    _roc_df: pd.DataFrame | None = None
    _pr_df: pd.DataFrame | None = None
    _client_df: pd.DataFrame | None = None
    _class_df: pd.DataFrame | None = None
    _communication_df: pd.DataFrame | None = None
    _runtime_df: pd.DataFrame | None = None
    _client_distribution_df: pd.DataFrame | None = None
    _client_support_df: pd.DataFrame | None = None

    @property
    def history(self) -> pd.DataFrame:
        if self._history_df is None:
            self._history_df=_csv(
                self.run_dir/"metrics"/"round_metrics.csv",
                self.run_dir/"federated_history.csv",
                self.run_dir/"metrics.csv",
            )
        return self._history_df

    @property
    def scalability(self) -> pd.DataFrame:
        return _csv(self.run_dir/"metrics"/"scalability_round_metrics.csv", self.run_dir/"scalability_round_metrics.csv", self.run_dir/"metrics"/"scalability.csv")

    @property
    def scores(self) -> pd.DataFrame:
        if self._scores_df is None:
            self._scores_df=_csv(
                self.run_dir/"predictions"/"open_set_scores.csv",
                self.run_dir/"open_set_scores.csv",
                self.run_dir/"prototype_rank_scores.csv", self.run_dir/"osr"/"test_scores.csv",
            )
        return self._scores_df

    @property
    def roc_curve(self) -> pd.DataFrame:
        if self._roc_df is None:
            self._roc_df = _csv(
                self.run_dir/"artifacts"/"open_set_roc_curve.csv",
                self.run_dir/"open_set_roc_curve.csv",
            )
        return self._roc_df

    @property
    def pr_curve(self) -> pd.DataFrame:
        if self._pr_df is None:
            self._pr_df = _csv(
                self.run_dir/"artifacts"/"open_set_pr_curve.csv",
                self.run_dir/"open_set_pr_curve.csv",
            )
        return self._pr_df

    @property
    def client_metrics(self) -> pd.DataFrame:
        if self._client_df is None:
            self._client_df=_csv(self.run_dir/"metrics"/"client_metrics.csv", self.run_dir/"client_eval_metrics.csv")
        return self._client_df

    @property
    def class_metrics(self) -> pd.DataFrame:
        if self._class_df is None:
            self._class_df=_csv(self.run_dir/"metrics"/"class_metrics.csv")
        return self._class_df

    @property
    def communication(self) -> pd.DataFrame:
        if self._communication_df is None:
            self._communication_df=_csv(self.run_dir/"metrics"/"communication_round.csv", self.run_dir/"metrics"/"communication.csv", self.run_dir/"communication_metrics.csv")
        return self._communication_df

    @property
    def runtime(self) -> pd.DataFrame:
        if self._runtime_df is None:
            self._runtime_df=_csv(self.run_dir/"metrics"/"timing_round.csv", self.run_dir/"metrics"/"runtime.csv", self.run_dir/"timing_round.csv")
        return self._runtime_df

    @property
    def client_distribution(self) -> pd.DataFrame:
        if self._client_distribution_df is None:
            self._client_distribution_df=_csv(
                self.run_dir/"metadata"/"client_class_distribution.csv",
                self.run_dir/"data"/"client_class_distribution.csv",
                self.run_dir/"processed"/"client_class_distribution.csv",
                self.run_dir/"client_class_distribution.csv",
            )
        return self._client_distribution_df

    @property
    def client_support(self) -> pd.DataFrame:
        if self._client_support_df is None:
            self._client_support_df = _csv(
                self.run_dir / "client_support.csv",
                self.run_dir / "metadata" / "client_support.csv",
                self.run_dir / "metrics" / "client_support.csv",
            )
        return self._client_support_df

    @property
    def confusion_before_path(self) -> Path | None:
        return _first_existing(self.run_dir/"artifacts"/"confusion_closed.npy", self.run_dir/"before_osr_confusion_matrix.csv")

    @property
    def confusion_after_path(self) -> Path | None:
        return _first_existing(self.run_dir/"artifacts"/"confusion_open.npy", self.run_dir/"after_osr_confusion_matrix.csv")

    @property
    def confusion_before(self) -> pd.DataFrame:
        p=self.confusion_before_path
        if p is None: return pd.DataFrame()
        if p.suffix==".npy": return pd.DataFrame(np.load(p, allow_pickle=False))
        return pd.read_csv(p, index_col=0)

    @property
    def confusion_after(self) -> pd.DataFrame:
        p=self.confusion_after_path
        if p is None: return pd.DataFrame()
        if p.suffix==".npy": return pd.DataFrame(np.load(p, allow_pickle=False))
        return pd.read_csv(p, index_col=0)

    def get_metric(self, candidate_keys: Iterable[str], default: float | None=None) -> float | None:
        for key in candidate_keys:
            if key not in self.metrics: continue
            try:
                val=float(self.metrics[key])
                if np.isfinite(val): return val
            except (TypeError, ValueError): pass
        return default


def _manifest(run_dir: Path) -> dict[str, Any]:
    for p in (run_dir/"metadata"/"run_manifest.json", run_dir/"run_manifest.json", run_dir/"metadata.json"):
        data=_json(p)
        if data: return data
    return {}


def _metrics(run_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any]={}
    for p in (
        run_dir/"metrics"/"final_metrics.json", run_dir/"metrics"/"evaluation_metrics.json",
        run_dir/"metrics"/"open_set_metrics.json", run_dir/"metrics"/"test_metrics.json",
        run_dir/"evaluation_metrics.json", run_dir/"open_set_metrics.json", run_dir/"training_summary.json", run_dir/"federated_summary.json",
        run_dir/"result_manifest.json",
    ):
        data=_json(p)
        if p.name=="result_manifest.json": data=data.get("final_metrics", {}) if isinstance(data.get("final_metrics", {}), dict) else {}
        merged.update(data)
    return merged


def _cfg(path: Path) -> Any:
    for p in (path/"config"/"resolved_config.yaml", path/"resolved_config.yaml", path/"config.yaml"):
        if p.exists():
            try: return OmegaConf.load(p)
            except Exception: pass
    return None


def _sel(config: Any, path: str, default: Any=None) -> Any:
    if config is None: return default
    try:
        value=OmegaConf.select(config, path, default=default)
        return default if value in (None,"???") else value
    except Exception: return default


def _normalize_method(value: str) -> str:
    low=value.lower()
    if "fedtros" in low: return "FedTROS-PR"
    if "fedprox" in low: return "FedProx-Student"
    if "fedavg" in low: return "FedAvg-Student"
    if "scaffold" in low: return "SCAFFOLD-Student"
    return value


def is_run_completed(run_dir: Path) -> bool:
    return str(_manifest(run_dir).get("status", "")).upper()=="COMPLETED"


def _infer_study_from_name(name: str) -> str:
    low = name.lower()
    if low.startswith("e0"): return "E0-VERIFY"
    if low.startswith("e1"): return "E1-IID-CS"
    if low.startswith("e2"): return "E2-IID-OSR"
    if low.startswith("e3"): return "E3-NIID-CS"
    if low.startswith("e4"): return "E4-NIID-FOSR"
    if low.startswith("e5"): return "E5-DATASET"
    if low.startswith("e6"): return "E6-SCALE"
    if low.startswith("e7"): return "E7-EFFICIENCY"
    if low.startswith("e8"): return "E8-LOAO"
    if low.startswith("a1"): return "A1-TEACHER"
    if low.startswith("a2"): return "A2-ANCHOR"
    if low.startswith("a3"): return "A3-TRANSFER"
    if low.startswith("a4"): return "A4-PR"
    if low.startswith("a5"): return "A5-FEATURE"
    if low.startswith("s1"): return "S1-SENSITIVITY"
    return "unknown"


def load_run(run_dir: str|Path) -> RunRecord:
    p=Path(run_dir).resolve()
    if not p.is_dir(): raise FileNotFoundError(p)
    manifest=_manifest(p); config=_cfg(p); metrics=_metrics(p)
    study_raw = manifest.get("study_id") or manifest.get("study") or _sel(config,"experiment.id",None)
    study = str(study_raw) if study_raw not in (None, "None", "unknown", "") else _infer_study_from_name(p.name)
    stage=str(manifest.get("stage") or _sel(config,"stage","unknown"))
    raw_method = str(manifest.get("method") or _sel(config,"experiment.method",_sel(config,"federated.strategy.name","unknown")))
    canonical_method = bool(_sel(config, "method.canonical", False))
    method = "FedTROS-MC" if canonical_method and "fedtros" in raw_method.lower() else _normalize_method(raw_method)
    dataset=str(manifest.get("dataset") or _sel(config,"dataset.name","unknown"))
    alpha=float(manifest.get("alpha", _sel(config,"dataset.preprocessing.alpha",0.0)) or 0.0)
    iid=bool(manifest.get("iid", _sel(config,"dataset.preprocessing.iid",False)))
    seed=int(manifest.get("seed", _sel(config,"seed",0)) or 0)
    clients=int(manifest.get("num_clients", _sel(config,"federated.num_clients",0)) or 0)
    unknown=list(manifest.get("unknown_labels") or manifest.get("held_out_unknown") or _sel(config,"dataset.preprocessing.unknown_labels",[]) or [])
    status=str(manifest.get("status", "COMPLETED" if (p/"result_manifest.json").exists() else "INCOMPLETE")).upper()
    variant_raw = manifest.get("variant") or _sel(config,"experiment.variant",None)
    variant = str(variant_raw) if variant_raw not in (None, "None", "") else "canonical"
    return RunRecord(
        run_id=str(manifest.get("run_id") or p.name), run_dir=p, metadata=manifest, config=config,
        study=study, stage=stage, method=method, dataset=dataset, alpha=alpha, iid=iid,
        seed=seed, num_clients=clients, unknown_labels=[str(x) for x in unknown], status=status,
        metrics=metrics, config_hash=str(manifest.get("config_hash", "")),
        split_hash=str(manifest.get("split_hash", manifest.get("dataset_split_hash", ""))),
        git_commit=str(manifest.get("git_commit", manifest.get("code_commit", ""))),
        timestamp_utc=str(manifest.get("started_at", manifest.get("created_at", ""))),
        variant=variant,
    )

    @property
    def client_support(self) -> pd.DataFrame:
        if self._client_support_df is None:
            self._client_support_df = _csv(self.run_dir / "client_support.csv")
        return self._client_support_df
