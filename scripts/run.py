"""Authoritative single-experiment runner for FedTROS-PR.

This is the only canonical one-run entry point.  It composes the resolved Hydra
configuration, generates an immutable run identity, initializes local ResultStore and
W&B (or NullTracker), executes the scientific pipeline, and finalizes provenance.
It never renders publication figures.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.data import run_preprocessing
from src.evaluation import run_evaluation, run_prototype_rank_evaluation
from src.experiment import create_run_services
from src.federated import run_federated_simulation
from src.infrastructure.logging import configure_logging, get_logger
from src.infrastructure.manifests import (
    RunStatus,
    get_git_info,
    create_initial_run_manifest,
    initialize_run_directories,
    update_run_manifest_status,
    write_canonical_manifests,
)
from src.infrastructure.run_id import generate_run_id, validate_run_collision
from src.training import run_training
from src.utils.config import sync_model_dimensions_from_preprocessing, validate_config
from src.utils.utils import resolve_device_from_config, set_seed

logger = get_logger("run")


def validate_experiment_config(cfg: DictConfig) -> None:
    """Fail fast on invalid scientific/configuration combinations."""
    known_labels = list(OmegaConf.select(cfg, "dataset.preprocessing.known_labels", default=[]) or [])
    unknown_labels = list(OmegaConf.select(cfg, "dataset.preprocessing.unknown_labels", default=[]) or [])
    overlap = set(known_labels) & set(unknown_labels)
    if overlap:
        raise ValueError(
            f"Unknown labels overlap with known labels: {sorted(overlap)}. "
            "Known and unknown label sets must be strictly disjoint!"
        )

    if int(OmegaConf.select(cfg, "federated.num_clients", default=1)) <= 0:
        raise ValueError("federated.num_clients must be > 0")
    if int(OmegaConf.select(cfg, "federated.num_rounds", default=1)) <= 0:
        raise ValueError("federated.num_rounds must be > 0")
    alpha = float(OmegaConf.select(cfg, "dataset.preprocessing.alpha", default=1.0))
    iid = bool(OmegaConf.select(cfg, "dataset.preprocessing.iid", default=False))
    if not iid and alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0 for non-IID studies")

    validate_config(cfg)


def _final_metrics(run_dir: Path) -> dict[str, Any]:
    """Load final scalar metrics for tracker summary/result manifest."""
    merged: dict[str, Any] = {}
    for candidate in (
        run_dir / "metrics" / "final_metrics.json",
        run_dir / "metrics" / "evaluation_metrics.json",
        run_dir / "metrics" / "open_set_metrics.json",
        run_dir / "metrics" / "test_metrics.json",
        # Legacy fallbacks retained only for historical-result import.
        run_dir / "evaluation_metrics.json",
        run_dir / "open_set_metrics.json",
        run_dir / "training_summary.json",
        run_dir / "federated_summary.json",
    ):
        if not candidate.exists():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _materialize_runtime_paths(cfg: DictConfig, run_id: str, run_dir: Path) -> None:
    """Materialize all *non-scientific* runtime paths inside the immutable run.

    The original project wrote every experiment's preprocessed tensors into the
    shared ``data/processed`` directory.  That is unsafe once ``run_study.py``
    dispatches independent cells concurrently: two methods/seeds can overwrite
    one another's tensors and manifests while training.  Each run therefore gets
    its own preprocessing directory under ``<run_dir>/data``.

    ``dataset.preprocessing.output_dir`` is intentionally *not* part of the
    scientific configuration hash.  Moving generated files into an isolated
    run directory changes storage location, not the experiment definition.  A
    separately persisted paired-partition file remains the scientific mechanism
    that makes matched methods reuse the exact same client assignment.
    """
    OmegaConf.update(cfg, "tracking.run_id", run_id, force_add=True)
    OmegaConf.update(cfg, "tracking.run_dir", str(run_dir.resolve()), force_add=True)
    OmegaConf.update(cfg, "run_dir", str(run_dir.resolve()), force_add=True)
    data_dir = (run_dir / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(
        cfg,
        "dataset.preprocessing.output_dir",
        str(data_dir),
        force_add=True,
    )




def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_preprocessing_artifacts(cfg: DictConfig, project_root: Path, run_dir: Path, metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Freeze split/partition/feature evidence inside this immutable run."""
    data_dir = Path(str(OmegaConf.select(cfg, "dataset.preprocessing.output_dir", default="data/processed")))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    meta_dir = run_dir / "metadata"
    copied: dict[str, str] = {}
    for name in (
        "preprocess_metadata.json", "partition_manifest.jsonl", "client_class_distribution.csv",
        "class_support.csv", "split_manifest.csv", "feature_schema.json", "class_names.json",
    ):
        src = data_dir / name
        if src.exists():
            dst = meta_dir / name
            shutil.copy2(src, dst)
            copied[name] = _sha256(dst)
    feature_manifest: dict[str, Any] = {}
    fp = meta_dir / "feature_schema.json"
    if fp.exists():
        try: feature_manifest = json.loads(fp.read_text(encoding="utf-8"))
        except Exception: pass
    partition_manifest = {
        "num_clients": metadata.get("num_clients"), "alpha": metadata.get("alpha"), "iid": metadata.get("iid"),
        "partition_records_sha256": copied.get("partition_manifest.jsonl", ""),
        "client_class_distribution_sha256": copied.get("client_class_distribution.csv", ""),
    }
    split_hash = copied.get("split_manifest.csv", "")
    return partition_manifest, feature_manifest, split_hash


def _set_manifest_split_hash(run_dir: Path, split_hash: str) -> None:
    if not split_hash: return
    for p in (run_dir / "metadata" / "run_manifest.json", run_dir / "run_manifest.json"):
        if not p.exists(): continue
        try:
            data = json.loads(p.read_text(encoding="utf-8")); data["split_hash"] = split_hash
            p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except Exception: pass



@contextmanager
def _time_stage(stage: str, timings: dict[str, float]):
    """Measure a top-level pipeline phase without contaminating per-round timing data."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[f"runtime/{stage}_seconds"] = float(time.perf_counter() - start)


def _write_pipeline_timing(run_dir: Path, timings: dict[str, float], total_seconds: float) -> None:
    """Persist whole-run phase timing separately from E6/E7 per-round instrumentation."""
    payload = dict(timings)
    payload["runtime/total_seconds"] = float(total_seconds)
    path = run_dir / "metrics" / "pipeline_timing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def execute_run(cfg: DictConfig, project_root: Path, *, resume: bool = False) -> Path:
    validate_experiment_config(cfg)

    study_id = str(
        OmegaConf.select(
            cfg,
            "experiment.id",
            default=OmegaConf.select(cfg, "experiment.name", default="E0-VERIFY"),
        )
    )
    raw_stage = OmegaConf.select(cfg, "stage", default="development")
    if isinstance(raw_stage, (dict, DictConfig)):
        stage = str(raw_stage.get("stage", "development"))
    else:
        stage = str(raw_stage)
    run_id, human_name, config_hash = generate_run_id(cfg, study_id=study_id)

    outputs_root = Path(str(OmegaConf.select(cfg, "output.root_dir", default="outputs")))
    if not outputs_root.is_absolute():
        outputs_root = project_root / outputs_root
    run_dir = outputs_root / "runs" / run_id
    validate_run_collision(run_dir, config_hash)
    initialize_run_directories(run_dir)
    _materialize_runtime_paths(cfg, run_id, run_dir)
    git_commit, git_dirty = get_git_info(project_root)
    OmegaConf.update(cfg, "experiment.config_hash", config_hash, force_add=True)
    OmegaConf.update(cfg, "experiment.git_commit", git_commit, force_add=True)
    OmegaConf.update(cfg, "experiment.git_dirty", bool(git_dirty), force_add=True)
    OmegaConf.update(cfg, "experiment.run_id", run_id, force_add=True)
    OmegaConf.update(cfg, "experiment.stage", stage, force_add=True)

    configure_logging(
        run_dir=run_dir,
        console_level=str(OmegaConf.select(cfg, "logging.level", default="INFO")),
        file_level=str(OmegaConf.select(cfg, "logging.debug_level", default="DEBUG")),
    )
    logger.info("=" * 88)
    logger.info("FedTROS-PR run starting | %s", human_name)
    logger.info("study=%s | stage=%s | run_id=%s | config=%s", study_id, stage, run_id, config_hash[:12])
    logger.info("run_dir=%s", run_dir)
    logger.info("=" * 88)

    services = create_run_services(
        cfg,
        run_dir=run_dir,
        run_id=run_id,
        human_name=human_name,
        study_id=study_id,
        stage=stage,
        resume=resume,
    )
    services.log_config(cfg)

    manifest = create_initial_run_manifest(
        cfg,
        run_id=run_id,
        study_id=study_id,
        stage=stage,
        config_hash=config_hash,
        project_root=project_root,
        tracker_run_id=services.tracker_run_id,
    )
    manifest.status = RunStatus.RUNNING.value
    manifest.save(run_dir)
    services.set_status(RunStatus.RUNNING.value)

    seed = int(OmegaConf.select(cfg, "seed", default=42))
    seed_settings = set_seed(
        seed,
        deterministic=bool(OmegaConf.select(cfg, "device.deterministic", default=True)),
        benchmark=bool(OmegaConf.select(cfg, "device.benchmark", default=False)),
        use_deterministic_algorithms=bool(
            OmegaConf.select(cfg, "device.use_deterministic_algorithms", default=False)
        ),
    )
    device = resolve_device_from_config(cfg)
    logger.info("Resolved compute device=%s | seed=%d", device, seed)

    pipeline_timings: dict[str, float] = {}
    pipeline = str(OmegaConf.select(cfg, "experiment.pipeline", default="full")).lower()
    start_time = time.perf_counter()

    try:
        with _time_stage("preprocessing", pipeline_timings):
            metadata = run_preprocessing(cfg, project_root=project_root)
            sync_model_dimensions_from_preprocessing(
                cfg, project_root=project_root, metadata=metadata
            )

        partition_manifest, feature_manifest, split_hash = _snapshot_preprocessing_artifacts(
            cfg, project_root, run_dir, metadata
        )
        write_canonical_manifests(
            run_dir,
            cfg=cfg,
            data_manifest=metadata,
            partition_manifest=partition_manifest,
            seed_manifest={"seed": seed, "seed_settings": seed_settings},
            model_manifest={
                "model_name": str(OmegaConf.select(cfg, "model.name", default="fedtros")),
                "method": str(OmegaConf.select(cfg, "experiment.method", default="FedTROS-PR")),
                "teacher_type": "variational_classifier",
                "open_set_method": "prototype_rank",
            },
            feature_manifest=feature_manifest,
        )
        _set_manifest_split_hash(run_dir, split_hash)

        if pipeline in {"full", "federated", "reproduce"}:
            with _time_stage("federated_execution", pipeline_timings):
                run_federated_simulation(cfg, project_root=project_root, tracker=services)

            eval_mode = str(OmegaConf.select(cfg, "evaluation.mode", default="closed_set")).lower()
            open_enabled = bool(OmegaConf.select(cfg, "open_set.enabled", default=False))
            if eval_mode == "open_set" or open_enabled:
                with _time_stage("open_set_eval", pipeline_timings):
                    run_prototype_rank_evaluation(
                        cfg,
                        project_root=project_root,
                        device=device,
                        tracker=services,
                    )
            else:
                with _time_stage("closed_set_eval", pipeline_timings):
                    run_evaluation(
                        cfg,
                        project_root=project_root,
                        device=device,
                        tracker=services,
                    )
        elif pipeline == "centralized":
            with _time_stage("centralized_training", pipeline_timings):
                run_training(cfg, project_root=project_root, device=device, tracker=services)
            open_enabled = bool(OmegaConf.select(cfg, "open_set.enabled", default=False)) or str(OmegaConf.select(cfg, "evaluation.mode", default="closed_set")).lower() == "open_set"
            with _time_stage("centralized_eval", pipeline_timings):
                if open_enabled:
                    run_prototype_rank_evaluation(cfg, project_root=project_root, device=device, tracker=services)
                else:
                    run_evaluation(cfg, project_root=project_root, device=device, tracker=services)
        elif pipeline == "evaluate":
            open_enabled = bool(OmegaConf.select(cfg, "open_set.enabled", default=False)) or str(OmegaConf.select(cfg, "evaluation.mode", default="closed_set")).lower() == "open_set"
            with _time_stage("evaluation_only", pipeline_timings):
                if open_enabled:
                    run_prototype_rank_evaluation(cfg, project_root=project_root, device=device, tracker=services)
                else:
                    run_evaluation(cfg, project_root=project_root, device=device, tracker=services)
        else:
            raise ValueError(f"Unknown experiment.pipeline={pipeline!r}")

        total_runtime = time.perf_counter() - start_time
        _write_pipeline_timing(run_dir, pipeline_timings, total_runtime)

        summary = _final_metrics(run_dir)
        summary["runtime/total_seconds"] = float(total_runtime)
        services.set_summary(summary)
        update_run_manifest_status(
            run_dir,
            RunStatus.COMPLETED,
            tracker_run_id=services.tracker_run_id,
        )
        services.set_status(RunStatus.COMPLETED.value)
        services.finish(status=RunStatus.COMPLETED.value)
        logger.info("Experiment completed in %.2fs | run_id=%s", total_runtime, run_id)
        return run_dir

    except KeyboardInterrupt:
        total_runtime = time.perf_counter() - start_time
        _write_pipeline_timing(run_dir, pipeline_timings, total_runtime)
        logger.warning("Experiment interrupted by user | runtime=%.2fs", total_runtime)
        update_run_manifest_status(
            run_dir,
            RunStatus.INTERRUPTED,
            tracker_run_id=services.tracker_run_id,
        )
        services.set_status(RunStatus.INTERRUPTED.value)
        services.finish(status=RunStatus.INTERRUPTED.value)
        raise
    except Exception as exc:
        total_runtime = time.perf_counter() - start_time
        _write_pipeline_timing(run_dir, pipeline_timings, total_runtime)
        logger.exception("Experiment failed after %.2fs: %s", total_runtime, exc)
        update_run_manifest_status(
            run_dir,
            RunStatus.FAILED,
            error=str(exc),
            tracker_run_id=services.tracker_run_id,
        )
        services.set_status(RunStatus.FAILED.value)
        services.finish(status=RunStatus.FAILED.value)
        raise


def _render_hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(repr(x) if isinstance(x, str) else str(x) for x in value) + "]"
    return str(value)


def _normalize_method_alias(value: str) -> str:
    token = value.lower().replace("-", "_")
    return {"fedtros": "fedtros_pr", "fedtros_pr": "fedtros_pr", "fedavg_student": "fedavg", "fedprox_student": "fedprox"}.get(token, token)


def _resolve_direct_study_args(raw_args: list[str]) -> list[str] | None:
    """Resolve ``study=...`` into exactly one declarative study cell.

    ``run.py`` is intentionally a *single-run* interface.  If the supplied study
    dimensions still expand to more than one cell, the user is directed to
    ``run_study.py`` rather than silently picking a seed/method/holdout.
    """
    values: dict[str, str] = {}
    for arg in raw_args:
        if "=" in arg and not arg.startswith("--"):
            key, value = arg.split("=", 1)
            values[key.strip().lower()] = value
    study_name = values.get("study")
    if not study_name:
        return None

    from src.infrastructure.study import expand_study_matrix, load_study_config

    stage = values.get("stage", "development")
    seed_filter = [int(values["seed"])] if "seed" in values else None
    study = load_study_config(study_name, _ROOT)
    plans = expand_study_matrix(study, stage=stage, seeds=seed_filter, project_root=_ROOT)

    def match(plan: Any) -> bool:
        if "alpha" in values and abs(plan.alpha - float(values["alpha"])) > 1e-9:
            return False
        client_value = values.get("clients", values.get("num_clients"))
        if client_value is not None and plan.num_clients != int(client_value):
            return False
        if "method" in values and _normalize_method_alias(plan.method) != _normalize_method_alias(values["method"]):
            return False
        if "dataset" in values and plan.dataset.lower() != values["dataset"].lower():
            return False
        if "variant" in values and plan.variant.lower() != values["variant"].lower():
            return False
        if "unknown" in values:
            requested = [x.strip() for x in values["unknown"].split(",") if x.strip()]
            if [x.lower() for x in plan.unknown_labels] != [x.lower() for x in requested]:
                return False
        return True

    candidates = [p for p in plans if match(p)]
    if len(candidates) != 1:
        sample = "\n".join(f"  - {p.run_id}" for p in candidates[:8]) or "  (none)"
        raise SystemExit(
            f"Direct study execution must resolve to exactly one run, but {len(candidates)} cells match {study_name}.\n"
            f"Add dimensions such as seed=42, method=fedtros_pr, alpha=0.5, variant=... or use:\n"
            f"  python scripts/run_study.py {study_name} --dry-run\n"
            f"Matching cells:\n{sample}"
        )

    selected = candidates[0]
    selection_keys = {"study", "seed", "alpha", "clients", "num_clients", "method", "dataset", "variant", "unknown", "stage"}
    # The study cell supplies all scientific overrides.  Preserve only explicit
    # infrastructure/runtime overrides not used to select the cell.
    passthrough = []
    for arg in raw_args:
        if "=" not in arg or arg.startswith("--"):
            passthrough.append(arg)
            continue
        key, value = arg.split("=", 1)
        normalized_key = key.strip().lower()
        if normalized_key in selection_keys:
            continue
        if normalized_key in {"tracking_mode", "wandb_mode"}:
            passthrough.append(f"tracking.mode={value}")
        else:
            passthrough.append(arg)
    resolved = [f"{key}={_render_hydra_value(value)}" for key, value in selected.overrides.items()]
    return [*resolved, *passthrough]


def normalize_cli_args() -> None:
    """Translate research-facing CLI into one validated canonical Hydra run."""
    raw_args = list(sys.argv[1:])
    study_args = _resolve_direct_study_args(raw_args)
    if study_args is not None:
        sys.argv = [sys.argv[0], *study_args]
        return

    remapped_args: list[str] = []
    for arg in raw_args:
        if "=" in arg and not arg.startswith("--"):
            key, value = arg.split("=", 1)
            k = key.lower().strip()
            if k == "alpha":
                remapped_args.append(f"dataset.preprocessing.alpha={value}")
            elif k in {"clients", "num_clients"}:
                remapped_args.extend(
                    [f"federated.num_clients={value}", f"dataset.preprocessing.num_clients={value}"]
                )
            elif k in {"rounds", "num_rounds"}:
                remapped_args.append(f"federated.num_rounds={value}")
            elif k == "method":
                token = _normalize_method_alias(value)
                remapped_args.append(f"method={token}")
                if token == "fedtros_pr":
                    remapped_args.append("model=fedtros")
            elif k in {"tracking_mode", "wandb_mode"}:
                remapped_args.append(f"tracking.mode={value}")
            else:
                remapped_args.append(arg)
        else:
            remapped_args.append(arg)
    sys.argv = [sys.argv[0], *remapped_args]


@hydra.main(config_path="../src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    execute_run(cfg, Path(get_original_cwd()))


if __name__ == "__main__":
    normalize_cli_args()
    main()
