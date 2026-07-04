from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.artifacts.suite import build_suite_artifacts
from src.data import run_preprocessing
from src.evaluation import run_evaluation
from src.evaluation.compare import compare_runs
from src.federated import run_federated_simulation
from src.plotting import generate_plots
from src.tracking import attach_to_existing_run, initialize_run
from src.training import run_smoke_test, run_training
from src.utils.config import resolve_path, validate_config
from src.utils.entrypoints import prepare_run_context


logger = logging.getLogger(__name__)


def _log_checkpoint_evaluation_summary(cfg: DictConfig, project_root: Path) -> None:
    best_metrics_path = resolve_path(project_root, Path(str(cfg.checkpointing.dir)) / "best_metrics.json")
    latest_path = resolve_path(project_root, cfg.checkpointing.latest_checkpoint_path)
    best_path = resolve_path(project_root, cfg.checkpointing.best_model_path)
    checkpoint_used = best_path if bool(OmegaConf.select(cfg, "evaluation.use_best_checkpoint", default=True)) and best_path.exists() else resolve_path(project_root, cfg.evaluation.checkpoint_path)

    best_round = None
    best_value = None
    latest_value = None
    if best_metrics_path.exists():
        try:
            payload = json.loads(best_metrics_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            best_round = metadata.get("round")
            best_value = payload.get("selected_metric_value")
            latest_value = metrics.get("val/macro_f1")
        except Exception:
            logger.exception("Failed to read best checkpoint metadata from %s", best_metrics_path)

    logger.info(
        "Checkpoint summary before evaluation | best_round=%s | best_validation_metric=%s | "
        "latest_checkpoint=%s | evaluation_checkpoint=%s",
        best_round,
        best_value,
        latest_path,
        checkpoint_used,
    )
    if best_value is not None and latest_value is not None:
        try:
            if float(best_value) - float(latest_value) > 0.05:
                logger.warning(
                    "Latest validation macro-F1 appears worse than best by >0.05 | best=%s | latest=%s",
                    best_value,
                    latest_value,
                )
        except (TypeError, ValueError):
            pass


def _pipeline(cfg: DictConfig) -> str:
    value = OmegaConf.select(cfg, "experiment.pipeline", default="full")
    return str(value or "full").lower()


def _suite_commands(cfg: DictConfig) -> Iterable[list[str]]:
    commands = OmegaConf.select(cfg, "experiment.suite_commands", default=[])
    for command in commands or []:
        yield [str(part) for part in command]


def _sync_model_shape_from_metadata(cfg: DictConfig, metadata: dict) -> None:
    if "state_dim" in metadata:
        cfg.model.state_dim = int(metadata["state_dim"])
    if "num_actions" in metadata:
        cfg.model.num_actions = int(metadata["num_actions"])


def _sync_tracker_shape_metadata(tracker, cfg: DictConfig) -> None:
    tracker.metadata["state_dim"] = int(cfg.model.state_dim)
    tracker.metadata["num_actions"] = int(cfg.model.num_actions)


def _run_suite(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    validate_config(cfg)
    for command in _suite_commands(cfg):
        child = [sys.executable, str(project_root / "run.py"), *command]
        subprocess.run(child, cwd=project_root, check=True)


def _run_plot(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    validate_config(cfg)
    run_dir = attach_to_existing_run(
        cfg,
        project_root=project_root,
        run_dir=cfg.run_dir,
        script_name="run.py:plot",
    )
    generate_plots(cfg, project_root=project_root, run_dir=run_dir)


def _run_export(cfg: DictConfig) -> None:
    project_root = Path(get_original_cwd())
    if not cfg.runs:
        raise ValueError("Provide runs=[outputs/run1,outputs/run2,...] for export.")
    tracker = initialize_run(cfg, project_root=project_root, script_name="run.py:export")
    compare_runs(cfg, project_root=project_root)
    generated = build_suite_artifacts(
        run_dirs=[resolve_path(project_root, run) for run in cfg.runs],
        output_dir=tracker.run_dir,
        project_root=project_root,
    )
    tracker.write_json(
        "suite_artifacts.json",
        {
            "input_runs": [str(run) for run in cfg.runs],
            "generated_files": {name: str(path) for name, path in sorted(generated.items())},
        },
    )


@hydra.main(config_path="src/configs", config_name="config_fl", version_base=None)
def main(cfg: DictConfig) -> None:
    pipeline = _pipeline(cfg)

    if pipeline == "suite":
        _run_suite(cfg)
        return
    if pipeline == "plot":
        _run_plot(cfg)
        return
    if pipeline in {"export", "suite_artifacts"}:
        _run_export(cfg)
        return

    extra_required = ("evaluation.checkpoint_path",) if pipeline == "evaluate" else ()
    context = prepare_run_context(
        cfg,
        script_name=f"run.py:{pipeline}",
        extra_required=extra_required,
        with_device=pipeline not in {"preprocess", "compare"},
        with_tracker=pipeline not in {"plot"},
    )

    if pipeline == "smoke":
        assert context.device is not None
        assert context.tracker is not None
        run_smoke_test(
            cfg,
            project_root=context.project_root,
            device=context.device,
            tracker=context.tracker,
        )
        return

    if pipeline == "preprocess":
        assert context.tracker is not None
        metadata = run_preprocessing(cfg, project_root=context.project_root)
        _sync_model_shape_from_metadata(cfg, metadata)
        _sync_tracker_shape_metadata(context.tracker, cfg)
        context.tracker.write_json("preprocess_metadata.json", metadata)
        return

    if pipeline == "compare":
        compare_runs(cfg, project_root=context.project_root)
        return

    if pipeline in {"full", "reproduce"}:
        assert context.device is not None
        assert context.tracker is not None
        metadata = run_preprocessing(cfg, project_root=context.project_root)
        _sync_model_shape_from_metadata(cfg, metadata)
        _sync_tracker_shape_metadata(context.tracker, cfg)
        context.tracker.save_config()
        context.tracker.save_metadata()
        run_federated_simulation(
            cfg,
            project_root=context.project_root,
            tracker=context.tracker,
        )
        _log_checkpoint_evaluation_summary(cfg, context.project_root)
        run_evaluation(
            cfg,
            project_root=context.project_root,
            device=context.device,
            tracker=context.tracker,
        )
        return

    if pipeline == "centralized":
        assert context.device is not None
        assert context.tracker is not None
        metadata = run_preprocessing(cfg, project_root=context.project_root)
        _sync_model_shape_from_metadata(cfg, metadata)
        _sync_tracker_shape_metadata(context.tracker, cfg)
        context.tracker.save_config()
        context.tracker.save_metadata()
        run_training(
            cfg,
            project_root=context.project_root,
            device=context.device,
            tracker=context.tracker,
        )
        run_evaluation(
            cfg,
            project_root=context.project_root,
            device=context.device,
            tracker=context.tracker,
        )
        return

    if pipeline == "train":
        assert context.device is not None
        assert context.tracker is not None
        run_training(
            cfg,
            project_root=context.project_root,
            device=context.device,
            tracker=context.tracker,
        )
        return

    if pipeline == "federated":
        assert context.tracker is not None
        run_federated_simulation(
            cfg,
            project_root=context.project_root,
            tracker=context.tracker,
        )
        return

    if pipeline == "evaluate":
        assert context.device is not None
        assert context.tracker is not None
        run_evaluation(
            cfg,
            project_root=context.project_root,
            device=context.device,
            tracker=context.tracker,
        )
        return

    raise ValueError(
        f"Unknown experiment.pipeline={pipeline!r}. "
        "Use full, smoke, centralized, preprocess, train, federated, evaluate, plot, compare, or export."
    )


if __name__ == "__main__":
    main()
