"""Canonical Manifests and Output Contract for FedTROS-PR experiments."""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.infrastructure.hardware import get_full_environment_provenance

logger = logging.getLogger(__name__)


class RunStatus(str, enum.Enum):
    """Lifecycle states of an experiment run."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    RESUMED = "RESUMED"
    CANCELLED = "CANCELLED"


def compute_file_sha256(path: Path) -> str:
    """Compute sha256 checksum for a file if it exists."""
    if not path.exists() or not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info(project_root: Path) -> tuple[str, bool]:
    """Retrieve git commit hash and dirty status."""
    commit = "unknown_commit"
    dirty = False
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        dirty = len(status_output) > 0
    except Exception:
        pass
    return commit, dirty


def initialize_run_directories(run_dir: Path) -> dict[str, Path]:
    """Create and return standard output directory layout.

    outputs/runs/<run_id>/
    ├── config/
    ├── data/
    ├── logs/
    ├── metrics/
    ├── checkpoints/
    ├── predictions/
    ├── artifacts/
    └── metadata/
    """
    subdirs = {
        "config": run_dir / "config",
        "data": run_dir / "data",
        "logs": run_dir / "logs",
        "metrics": run_dir / "metrics",
        "checkpoints": run_dir / "checkpoints",
        "predictions": run_dir / "predictions",
        "artifacts": run_dir / "artifacts",
        "metadata": run_dir / "metadata",
    }
    for directory in subdirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return subdirs


@dataclass
class RunManifest:
    """Canonical run manifest conforming to Schema Version 2."""

    schema_version: int = 2
    run_id: str = ""
    study_id: str = ""
    stage: str = "development"
    status: str = RunStatus.CREATED.value

    method: str = "FedTROS-PR"
    method_id: str = "fedtros_pr"
    teacher_type: str = "variational_classifier"
    open_set_method: str = "multicenter_conformal"

    dataset: str = "bnat"
    known_labels: list[Any] = field(default_factory=list)
    unknown_labels: list[Any] = field(default_factory=list)

    num_clients: int = 10
    num_rounds: int = 100
    local_training_budget: dict[str, Any] = field(default_factory=lambda: {"local_epochs": 1, "batch_size": 32})

    partition_type: str = "dirichlet"
    alpha: float = 0.1
    iid: bool = False

    seed: int = 42
    split_seed: int = 42
    partition_seed: int = 42

    config_hash: str = ""
    split_hash: str = ""

    git_commit: str = ""
    git_dirty: bool = False

    hostname: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)

    started_at: str = ""
    finished_at: str | None = None
    runtime: float = 0.0

    tracker_run_id: str | None = None
    artifact_inventory: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, run_dir: Path) -> Path:
        target = run_dir / "metadata" / "run_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        # Also maintain a backward-compatible root symlink/copy if needed
        (run_dir / "run_manifest.json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return target


def create_initial_run_manifest(
    cfg: DictConfig | dict[str, Any],
    *,
    run_id: str,
    study_id: str,
    stage: str,
    config_hash: str,
    project_root: Path,
    tracker_run_id: str | None = None,
) -> RunManifest:
    """Create an initial RunManifest in CREATED status."""
    env = get_full_environment_provenance()
    git_commit, git_dirty = get_git_info(project_root)

    def get_val(key: str, default: Any = None) -> Any:
        if isinstance(cfg, DictConfig):
            val = OmegaConf.select(cfg, key, default=default)
            return val if val is not None else default
        parts = key.split(".")
        curr = cfg
        for p in parts:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
            else:
                return default
        return curr if curr is not None else default

    manifest = RunManifest(
        schema_version=2,
        run_id=run_id,
        study_id=study_id,
        stage=stage,
        status=RunStatus.CREATED.value,
        method=str(get_val("experiment.method", "FedTROS-PR")),
        method_id=str(get_val("federated.strategy.name", get_val("strategy.name", "fedtros_pr"))),
        teacher_type=str(get_val("model.teacher_type", "variational_classifier")),
        open_set_method=str(get_val("open_set.detector", "multicenter_conformal")),
        dataset=str(get_val("dataset.name", "bnat")),
        known_labels=list(get_val("dataset.preprocessing.known_labels", [])),
        unknown_labels=list(get_val("dataset.preprocessing.unknown_labels", [])),
        num_clients=int(get_val("federated.num_clients", get_val("dataset.preprocessing.num_clients", 10))),
        num_rounds=int(get_val("federated.num_rounds", 100)),
        local_training_budget={
            "local_epochs": int(get_val("training.local_epochs", 1)),
            "batch_size": int(get_val("training.batch_size", 32)),
            "learning_rate": float(get_val("training.learning_rate", 0.001)),
        },
        partition_type="iid" if bool(get_val("dataset.preprocessing.iid", False)) else "dirichlet",
        alpha=float(get_val("dataset.preprocessing.alpha", 0.1)),
        iid=bool(get_val("dataset.preprocessing.iid", False)),
        seed=int(get_val("seed", 42)),
        split_seed=int(get_val("seed", 42)),
        partition_seed=int(get_val("seed", 42)),
        config_hash=config_hash,
        split_hash="",
        git_commit=git_commit,
        git_dirty=git_dirty,
        hostname=env["hostname"],
        hardware={"cpu_ram": env["cpu_ram"], "gpu": env["gpu"]},
        software={"python": env["python"], "packages": env["packages"]},
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        runtime=0.0,
        tracker_run_id=tracker_run_id,
        artifact_inventory={},
    )
    return manifest


def update_run_manifest_status(
    run_dir: Path,
    status: RunStatus | str,
    *,
    error: str | None = None,
    tracker_run_id: str | None = None,
) -> RunManifest | None:
    """Update status, finish time, and runtime in the run_manifest.json."""
    manifest_path = run_dir / "metadata" / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["status"] = status.value if isinstance(status, RunStatus) else str(status)
        if tracker_run_id:
            data["tracker_run_id"] = tracker_run_id
        if error:
            data["error_message"] = error

        if status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.INTERRUPTED}:
            now_iso = datetime.now(timezone.utc).isoformat()
            data["finished_at"] = now_iso
            started = data.get("started_at")
            if started:
                try:
                    start_dt = datetime.fromisoformat(started)
                    end_dt = datetime.fromisoformat(now_iso)
                    data["runtime"] = round((end_dt - start_dt).total_seconds(), 2)
                except Exception:
                    pass

        # Scan artifact inventory
        inventory: dict[str, dict[str, Any]] = {}
        for sub in ("config", "metrics", "checkpoints", "predictions", "artifacts", "metadata"):
            subdir = run_dir / sub
            if subdir.exists() and subdir.is_dir():
                for p in subdir.glob("*"):
                    if p.is_file() and p.name not in {"run_manifest.json", "result_manifest.json"}:
                        rel_path = p.relative_to(run_dir).as_posix()
                        inventory[rel_path] = {
                            "size_bytes": p.stat().st_size,
                            "sha256": compute_file_sha256(p),
                        }
        data["artifact_inventory"] = inventory

        manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        if (run_dir / "run_manifest.json").exists():
            (run_dir / "run_manifest.json").write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
    except Exception as exc:
        logger.warning("Could not update run manifest status: %s", exc)
    return None


def write_canonical_manifests(
    run_dir: Path,
    *,
    cfg: DictConfig | dict[str, Any],
    data_manifest: dict[str, Any] | None = None,
    partition_manifest: dict[str, Any] | None = None,
    model_manifest: dict[str, Any] | None = None,
    seed_manifest: dict[str, Any] | None = None,
    feature_manifest: dict[str, Any] | None = None,
    result_manifest: dict[str, Any] | None = None,
) -> None:
    """Persist all required sub-manifests to outputs/runs/<run_id>/metadata/."""
    meta_dir = run_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_map = {
        "data_manifest.json": data_manifest,
        "partition_manifest.json": partition_manifest,
        "model_manifest.json": model_manifest,
        "seed_manifest.json": seed_manifest,
        "feature_manifest.json": feature_manifest,
        "result_manifest.json": result_manifest,
    }
    for filename, content in manifest_map.items():
        if content is not None:
            (meta_dir / filename).write_text(
                json.dumps(content, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            # Root copy for convenience
            (run_dir / filename).write_text(
                json.dumps(content, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
