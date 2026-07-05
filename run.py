from __future__ import annotations

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
from src.evaluation import run_dkd_fedos_student_open_set_evaluation, run_evaluation
from src.evaluation.compare import compare_runs
from src.federated import run_federated_simulation
from src.plotting import generate_plots
from src.tracking import attach_to_existing_run, initialize_run
from src.training import run_training
from src.utils.config import resolve_path, validate_config
from src.utils.entrypoints import prepare_run_context




def _enforce_dkd_fedos_v5_contract(cfg: DictConfig) -> None:
    """Fail fast if DKD-FedOS is not running the student-anchor setup.

    This prevents silent Hydra/default fallbacks where logs look like DKD-FedOS
    but the anchor, stronger student, or reliability aggregation are not active.
    """
    strategy_name = str(OmegaConf.select(cfg, "strategy.name", default="")).lower()
    if strategy_name != "dkd_fedos":
        return

    logger = logging.getLogger("run")
    aggregation_mode = str(
        OmegaConf.select(cfg, "federated.strategy.student_aggregation_mode", default="")
        or OmegaConf.select(cfg, "strategy.student_aggregation_mode", default="")
    ).lower()
    anchor_weight = float(OmegaConf.select(cfg, "training.dkd_global_anchor_weight", default=0.0))
    hidden_dims_raw = OmegaConf.select(cfg, "training.dkd_student_hidden_dims", default=[])
    hidden_dims = [int(v) for v in list(hidden_dims_raw)]
    expected_hidden = [512, 256, 128]
    t2s_start = int(OmegaConf.select(cfg, "training.dkd_teacher_to_student_start_round", default=999))
    align_start = int(OmegaConf.select(cfg, "training.dkd_alignment_start_round", default=999))
    s2t_enabled = bool(OmegaConf.select(cfg, "training.dkd_update_teacher_from_student", default=False))

    if aggregation_mode != "reliability_weighted_average":
        raise ValueError(
            "DKD-FedOS V5 STUDENT-ANCHOR is not active: "
            f"student_aggregation_mode must be reliability_weighted_average, got {aggregation_mode!r}."
        )
    if anchor_weight <= 0.0:
        raise ValueError(
            "DKD-FedOS V5 STUDENT-ANCHOR is not active: "
            "training.dkd_global_anchor_weight must be > 0."
        )
    if hidden_dims != expected_hidden:
        raise ValueError(
            "DKD-FedOS V5 STUDENT-ANCHOR is not active: "
            f"student_hidden_dims must be {expected_hidden}, got {hidden_dims}."
        )
    if t2s_start > 1:
        raise ValueError(
            "DKD-FedOS V5 STUDENT-ANCHOR contract violation: "
            f"dkd_teacher_to_student_start_round must be 1, got {t2s_start}."
        )
    if align_start > 2:
        raise ValueError(
            "DKD-FedOS V5 STUDENT-ANCHOR contract violation: "
            f"dkd_alignment_start_round must be <= 2, got {align_start}."
        )
    if s2t_enabled:
        raise ValueError(
            "DKD-FedOS V5 STUDENT-ANCHOR contract violation: "
            "training.dkd_update_teacher_from_student must stay false for RL safety."
        )

    evt_enabled = bool(OmegaConf.select(cfg, "open_set.evt.enabled", default=False))
    evt_backend = str(OmegaConf.select(cfg, "open_set.evt.backend", default="teacher_generator")).lower()
    if evt_enabled and evt_backend not in {"student_feature_evt", "student_feature", "feature_evt", "dual_boundary_evt", "dual", "dual_evt"}:
        raise ValueError(
            "DKD-FedOS v7 supports only open_set.evt.backend=student_feature_evt "
            "or dual_boundary_evt for DKD-FedOS open-set runs."
        )
    logger.info(
        "DKD-FedOS V5 STUDENT-ANCHOR ACTIVE | "
        "student_aggregation_mode=%s | global_anchor_enabled=%s | "
        "global_anchor_weight=%.3f | student_hidden_dims=%s | "
        "t2s_start_round=%d | alignment_start_round=%d | "
        "open_set_evt_enabled=%s | open_set_evt_backend=%s",
        aggregation_mode,
        bool(anchor_weight > 0.0),
        anchor_weight,
        hidden_dims,
        t2s_start,
        align_start,
        evt_enabled,
        evt_backend,
    )


def _pipeline(cfg: DictConfig) -> str:
    value = OmegaConf.select(cfg, "experiment.pipeline", default="full")
    return str(value or "full").lower()


def _suite_commands(cfg: DictConfig) -> Iterable[list[str]]:
    commands = OmegaConf.select(cfg, "experiment.suite_commands", default=[])
    for command in commands or []:
        yield [str(part) for part in command]


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

    _enforce_dkd_fedos_v5_contract(cfg)

    if pipeline == "preprocess":
        assert context.tracker is not None
        metadata = run_preprocessing(cfg, project_root=context.project_root)
        context.tracker.write_json("preprocess_metadata.json", metadata)
        return

    if pipeline == "compare":
        compare_runs(cfg, project_root=context.project_root)
        return

    if pipeline in {"full", "reproduce"}:
        assert context.device is not None
        assert context.tracker is not None
        run_preprocessing(cfg, project_root=context.project_root)
        run_federated_simulation(
            cfg,
            project_root=context.project_root,
            tracker=context.tracker,
        )
        if str(OmegaConf.select(cfg, "strategy.name", default="")).lower() == "dkd_fedos":
            # DKD-FedOS globally aggregates only the compact student.  For
            # closed-set experiments the client-side Flower reports are enough.
            # For open-set E2/E4, evaluate the aggregated global student with
            # class-wise Feature-EVT. The optional local generator branch is
            # client-side only and is not uploaded to the server.
            if bool(OmegaConf.select(cfg, "open_set.evt.enabled", default=False)):
                run_dkd_fedos_student_open_set_evaluation(
                    cfg,
                    project_root=context.project_root,
                    device=context.device,
                    tracker=context.tracker,
                )
            else:
                print("DKD-FedOS: skipping standard full-agent evaluation; global object is student-only.")
            return
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
        run_preprocessing(cfg, project_root=context.project_root)
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
        "Use full, centralized, preprocess, train, federated, evaluate, plot, compare, or export."
    )


if __name__ == "__main__":
    main()

