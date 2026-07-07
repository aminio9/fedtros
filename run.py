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
from src.utils.config import resolve_path, sync_model_dimensions_from_preprocessing, validate_config
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
    if evt_enabled and evt_backend not in {"fed_digos", "digos", "student_digos"}:
        raise ValueError(
            "DKD-FedOS open-set runs now use Fed-DiGOS only. Set "
            "open_set.evt.backend=fed_digos and training.dkd_student_osr_enabled=true."
        )
    if evt_enabled and evt_backend in {"fed_digos", "digos", "student_digos"}:
        osr_enabled = bool(OmegaConf.select(cfg, "training.dkd_student_osr_enabled", default=False))
        if not osr_enabled:
            raise ValueError(
                "Fed-DiGOS requires training.dkd_student_osr_enabled=true."
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


def _open_set_evaluation_requested(cfg: DictConfig) -> bool:
    """Return True when the experiment contract requires open-set evaluation."""
    evaluation_mode = str(OmegaConf.select(cfg, "evaluation.mode", default="closed_set") or "").lower()
    family = str(OmegaConf.select(cfg, "experiment.family", default="") or "").lower()
    dependencies = OmegaConf.select(cfg, "experiment.dependencies", default=[]) or []
    if evaluation_mode == "open_set":
        return True
    if "open_set" in family or "openset" in family:
        return True
    return any("open_set" in str(dep).lower() for dep in dependencies)


def _require_dkd_fedos_open_set_config(cfg: DictConfig) -> None:
    """Fail loudly when an open-set DKD-FedOS experiment is not wired to Fed-DiGOS."""
    if not _open_set_evaluation_requested(cfg):
        return
    if str(OmegaConf.select(cfg, "strategy.name", default="")).lower() != "dkd_fedos":
        return

    evt_enabled = bool(OmegaConf.select(cfg, "open_set.evt.enabled", default=False))
    evt_backend = str(OmegaConf.select(cfg, "open_set.evt.backend", default="") or "").lower()
    fed_digos_enabled = bool(OmegaConf.select(cfg, "open_set.fed_digos.enabled", default=False))
    osr_enabled = bool(OmegaConf.select(cfg, "training.dkd_student_osr_enabled", default=False))
    open_set_head_enabled = bool(OmegaConf.select(cfg, "training.dkd_student_open_set_enabled", default=False))

    if not evt_enabled:
        raise ValueError(
            "Open-set DKD-FedOS run is misconfigured: evaluation.mode=open_set but "
            "open_set.evt.enabled=false. Set open_set.evt.enabled=true."
        )
    if evt_backend not in {"fed_digos", "digos", "student_digos"}:
        raise ValueError(
            "Open-set DKD-FedOS run is misconfigured: backend must be fed_digos, "
            f"got {evt_backend!r}. Set open_set.evt.backend=fed_digos."
        )
    if not fed_digos_enabled:
        raise ValueError(
            "Open-set DKD-FedOS run is misconfigured: open_set.fed_digos.enabled=false. "
            "Set open_set.fed_digos.enabled=true."
        )
    if not osr_enabled:
        raise ValueError(
            "Open-set DKD-FedOS run is misconfigured: training.dkd_student_osr_enabled=false. "
            "Set training.dkd_student_osr_enabled=true."
        )
    if not open_set_head_enabled:
        raise ValueError(
            "Open-set DKD-FedOS run is misconfigured: training.dkd_student_open_set_enabled=false. "
            "Set training.dkd_student_open_set_enabled=true."
        )


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
        metadata = run_preprocessing(cfg, project_root=context.project_root)
        sync_model_dimensions_from_preprocessing(
            cfg, project_root=context.project_root, metadata=metadata
        )
        run_federated_simulation(
            cfg,
            project_root=context.project_root,
            tracker=context.tracker,
        )
        if str(OmegaConf.select(cfg, "strategy.name", default="")).lower() == "dkd_fedos":
            # DKD-FedOS globally aggregates only the compact student. Closed-set
            # experiments use the Flower shared-test reports. Open-set experiments
            # such as E2/E4/E8 must run one final server-side Fed-DiGOS evaluation
            # on the aggregated global student after FL.
            if _open_set_evaluation_requested(cfg):
                _require_dkd_fedos_open_set_config(cfg)
                logging.getLogger("run").info(
                    "DKD-FedOS open-set final evaluation requested | experiment=%s | mode=%s | backend=%s",
                    OmegaConf.select(cfg, "experiment.id", default="?"),
                    OmegaConf.select(cfg, "evaluation.mode", default="?"),
                    OmegaConf.select(cfg, "open_set.evt.backend", default="?"),
                )
                run_dkd_fedos_student_open_set_evaluation(
                    cfg,
                    project_root=context.project_root,
                    device=context.device,
                    tracker=context.tracker,
                )
            elif bool(OmegaConf.select(cfg, "open_set.evt.enabled", default=False)):
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
        metadata = run_preprocessing(cfg, project_root=context.project_root)
        sync_model_dimensions_from_preprocessing(
            cfg, project_root=context.project_root, metadata=metadata
        )
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
        sync_model_dimensions_from_preprocessing(cfg, project_root=context.project_root)
        run_training(
            cfg,
            project_root=context.project_root,
            device=context.device,
            tracker=context.tracker,
        )
        return

    if pipeline == "federated":
        assert context.tracker is not None
        sync_model_dimensions_from_preprocessing(cfg, project_root=context.project_root)
        run_federated_simulation(
            cfg,
            project_root=context.project_root,
            tracker=context.tracker,
        )
        return

    if pipeline == "evaluate":
        assert context.device is not None
        assert context.tracker is not None
        sync_model_dimensions_from_preprocessing(cfg, project_root=context.project_root)
        if (
            str(OmegaConf.select(cfg, "strategy.name", default="")).lower() == "dkd_fedos"
            and _open_set_evaluation_requested(cfg)
        ):
            _require_dkd_fedos_open_set_config(cfg)
            run_dkd_fedos_student_open_set_evaluation(
                cfg,
                project_root=context.project_root,
                device=context.device,
                tracker=context.tracker,
            )
        else:
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

