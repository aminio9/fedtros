from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.agents.agent import Agent
from src.artifacts.embeddings import export_latent_embeddings
from src.checkpointing.checkpoints import load_agent_checkpoint
from src.data.io import load_tensor_dataset
from src.evaluation.closed_set import evaluate_closed_set, load_class_names
from src.evaluation.openset_eval import (
    calibrate_evt_thresholds,
    evaluate_open_set,
    fit_evt_models,
)
from src.models.models import OpenSetQChainModelFactory
from src.openset.evt import save_evt_collection, save_evt_meta
from src.openset.feature_evt import (
    evaluate_feature_evt_open_set,
    fit_student_feature_evt_models,
    save_feature_evt_collection,
)
from src.tracking.local import LocalRunTracker
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def build_agent(cfg: DictConfig, device: torch.device) -> Agent:
    return Agent(OpenSetQChainModelFactory(cfg.model), cfg.training, device=device)


def _latent_export_tensor(
    cfg: DictConfig,
    *,
    project_root: Path,
    closed_features: torch.Tensor,
    closed_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    evaluation_mode = str(cfg.evaluation.mode).lower()
    open_set_data_path = resolve_path(project_root, cfg.evaluation.open_set_data)
    if evaluation_mode in {"open_set", "export_only"} and open_set_data_path.exists():
        open_features, open_labels = load_tensor_dataset(
            open_set_data_path,
            map_location="cpu",
        )
        return open_features, open_labels, "open_set"
    return closed_features, closed_labels, "closed_set"


def run_evaluation(
    cfg: DictConfig,
    *,
    project_root: Path,
    device: torch.device,
    tracker: LocalRunTracker | None = None,
) -> dict[str, Any]:
    output_dir = resolve_path(project_root, cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(cfg, device)
    checkpoint_path = resolve_path(project_root, cfg.evaluation.checkpoint_path)
    load_agent_checkpoint(
        agent,
        checkpoint_path,
        device,
        strict=bool(cfg.checkpointing.strict_load),
        load_optimizers=False,
    )

    class_names = load_class_names(
        resolve_path(project_root, cfg.evaluation.class_names),
        int(cfg.model.num_actions),
    )
    closed_features, closed_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.closed_set_data),
        map_location="cpu",
    )
    closed_metrics = evaluate_closed_set(
        agent,
        closed_features,
        closed_labels,
        batch_size=int(cfg.evaluation.batch_size),
        device=device,
        class_names=class_names,
        output_dir=output_dir,
        prefix="test",
        save_predictions=bool(cfg.evaluation.save_predictions),
    )
    if tracker:
        tracker.log_metrics(closed_metrics)

    all_metrics: dict[str, Any] = dict(closed_metrics)
    evt_cfg = cfg.open_set.evt
    if bool(evt_cfg.enabled):
        calibration_data_path = resolve_path(project_root, cfg.evaluation.validation_data)
        if not calibration_data_path.exists():
            raise FileNotFoundError(
                "EVT calibration requires validation-only data. Missing: "
                f"{calibration_data_path}"
            )
        calibration_features, calibration_labels = load_tensor_dataset(
            calibration_data_path,
            map_location="cpu",
        )

        evt_output_dir = output_dir / "evt"
        evt_output_dir.mkdir(parents=True, exist_ok=True)
        evt_models = fit_evt_models(
            features=calibration_features,
            labels=calibration_labels,
            batch_size=int(cfg.evaluation.batch_size),
            evt_cfg=evt_cfg,
            prior_net=agent.prior_net,
            recognition_net=agent.recognition_net,
            value_net_main=agent.value_net_main,
            generation_net=agent.generation_net,
            student_model=agent.student_model,
            device=device,
        )
        evt_meta = calibrate_evt_thresholds(
            features=calibration_features,
            labels=calibration_labels,
            batch_size=int(cfg.evaluation.batch_size),
            evt_models=evt_models,
            evt_cfg=evt_cfg,
            prior_net=agent.prior_net,
            recognition_net=agent.recognition_net,
            value_net_main=agent.value_net_main,
            generation_net=agent.generation_net,
            student_model=agent.student_model,
            device=device,
        )
        save_evt_collection(evt_models, evt_output_dir / "evt_models.pkl")
        save_evt_meta(evt_meta, evt_output_dir / "evt_meta.json")
        open_features, open_labels = load_tensor_dataset(
            resolve_path(project_root, cfg.evaluation.open_set_data),
            map_location="cpu",
        )
        open_metrics = evaluate_open_set(
            features=open_features,
            labels=open_labels,
            batch_size=int(cfg.evaluation.batch_size),
            prior_net=agent.prior_net,
            recognition_net=agent.recognition_net,
            value_net_main=agent.value_net_main,
            generation_net=agent.generation_net,
            student_model=agent.student_model,
            evt_models=evt_models,
            evt_meta=evt_meta,
            class_names=class_names,
            output_dir=output_dir,
            device=device,
            evt_cfg=evt_cfg,
            report_to_stdout=bool(cfg.evaluation.report_to_stdout),
        )
        all_metrics.update(open_metrics)
        if tracker:
            tracker.log_metrics(open_metrics)

    if bool(cfg.evaluation.export_latent_embeddings):
        latent_features, latent_labels, source = _latent_export_tensor(
            cfg,
            project_root=project_root,
            closed_features=closed_features,
            closed_labels=closed_labels,
        )
        logger.info("Exporting latent embeddings from the %s evaluation tensor.", source)

        export_latent_embeddings(
            prior_net=agent.prior_net,
            features=latent_features,
            labels=latent_labels,
            class_names=class_names,
            output_path=resolve_path(project_root, cfg.evaluation.latent_embeddings_output),
            batch_size=int(cfg.evaluation.latent_embeddings_batch_size),
            max_points=int(cfg.evaluation.latent_embeddings_max_points),
            source=source,
        )

    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Evaluation complete. Metrics saved under %s", output_dir)
    return all_metrics


def run_dkd_fedos_student_open_set_evaluation(
    cfg: DictConfig,
    *,
    project_root: Path,
    device: torch.device,
    tracker: LocalRunTracker | None = None,
) -> dict[str, Any]:
    """Evaluate the DKD-FedOS global student with the configured open-set backend.

    DKD-FedOS uploads/aggregates only the compact student.  The default open-set
    backend is ``student_feature_evt``: class-wise EVT over Mahalanobis
    distances in the global-student feature space.

    ``dual_boundary_evt`` is supported when local generator boundaries are fitted
    in client-side evaluation.  In this server-side global-student evaluator, the
    local generator branch is unavailable by design, so ``dual_boundary_evt``
    runs the primary global feature EVT boundary and records that the local
    branch was skipped.
    """
    evt_cfg = cfg.open_set.evt
    if not bool(getattr(evt_cfg, "enabled", False)):
        logger.info("DKD-FedOS open-set evaluation skipped: EVT disabled.")
        return {}

    backend = str(getattr(evt_cfg, "backend", "student_feature_evt")).lower()
    backend_aliases = {
        "feature": "student_feature_evt",
        "student_feature": "student_feature_evt",
        "student_feature_evt": "student_feature_evt",
        "feature_evt": "student_feature_evt",
        "dual": "dual_boundary_evt",
        "dual_evt": "dual_boundary_evt",
        "dual_boundary_evt": "dual_boundary_evt",
    }
    backend = backend_aliases.get(backend, backend)
    if backend not in {"student_feature_evt", "dual_boundary_evt"}:
        logger.info("DKD-FedOS open-set evaluation skipped: unsupported evt.backend=%s", backend)
        return {}

    output_dir = resolve_path(project_root, cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(cfg, device)

    student_ckpt = resolve_path(
        project_root,
        Path(str(cfg.checkpointing.dir)) / "dkd_fedos_student_latest.pt",
    )
    if not student_ckpt.exists():
        raise FileNotFoundError(f"DKD-FedOS student checkpoint not found: {student_ckpt}")
    payload = torch.load(student_ckpt, map_location=device, weights_only=False)
    state = payload.get("student_model")
    if state is None:
        raise KeyError(f"Missing 'student_model' in DKD-FedOS checkpoint: {student_ckpt}")
    agent.student_model.load_state_dict(state, strict=True)
    agent.student_model.to(device).eval()

    class_names = load_class_names(
        resolve_path(project_root, cfg.evaluation.class_names),
        int(cfg.model.num_actions),
    )
    calibration_features, calibration_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.validation_data),
        map_location="cpu",
    )
    open_features, open_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.open_set_data),
        map_location="cpu",
    )

    num_unknown_train = int((calibration_labels < 0).sum().item())
    unknown_label_id = int(getattr(evt_cfg, "unknown_label_id", -1))
    num_unknown_test = int((open_labels == unknown_label_id).sum().item())
    logger.info(
        "OPEN-SET PROTOCOL ACTIVE | known_classes=%s | heldout_unknown=%s | "
        "unknown_label_id=%s | num_actions=%d | calibration_samples=%d | "
        "calibration_unknown_samples=%d | open_test_samples=%d | open_test_unknown_samples=%d | "
        "backend=%s",
        [class_names[k] for k in sorted(class_names)],
        "FoT",
        unknown_label_id,
        int(cfg.model.num_actions),
        int(calibration_labels.numel()),
        num_unknown_train,
        int(open_labels.numel()),
        num_unknown_test,
        backend,
    )
    if num_unknown_train != 0:
        raise ValueError("EVT calibration data must contain known classes only; found unknown labels.")

    evt_output_dir = output_dir / "evt"
    evt_output_dir.mkdir(parents=True, exist_ok=True)

    if backend in {"student_feature_evt", "dual_boundary_evt"}:
        if backend == "dual_boundary_evt":
            logger.warning(
                "DUAL-BOUNDARY EVT requested in server-side DKD-FedOS evaluation. "
                "Local teacher/generator checkpoints are not uploaded, so the final server metric uses "
                "the primary global student Feature-EVT boundary only. Enable client_eval_enabled=true "
                "for client-side local-generator ablation."
            )
        logger.info(
            "GLOBAL STUDENT FEATURE-EVT ACTIVE | score=mahalanobis_feature_distance | "
            "threshold_method=%s | target_known_fpr=%.4f",
            str(getattr(evt_cfg, "threshold_method", "mef")),
            float(getattr(evt_cfg, "target_known_fpr", 0.05)),
        )
        feature_boundaries, evt_meta, calibration_df = fit_student_feature_evt_models(
            features=calibration_features,
            labels=calibration_labels,
            batch_size=int(cfg.evaluation.batch_size),
            student_model=agent.student_model,
            evt_cfg=evt_cfg,
            device=device,
            logger_=logger,
        )
        # If the user selected dual_boundary_evt, record the requested backend but
        # keep the same global boundaries.  The evaluator will not try to use local
        # generator boundaries unless they are explicitly provided.
        if backend == "dual_boundary_evt":
            evt_meta["backend"] = "dual_boundary_evt"
            evt_meta["local_generator_available"] = False
        save_feature_evt_collection(feature_boundaries, evt_output_dir / "feature_evt_models.pkl", logger_=logger)
        (evt_output_dir / "feature_evt_meta.json").write_text(
            json.dumps(evt_meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        calibration_df.to_csv(output_dir / "student_feature_distances_calibration.csv", index=False)
        logger.info(
            "Feature-EVT fitted | classes=%s | thresholds=%s",
            sorted(feature_boundaries.keys()),
            {int(k): float(v.threshold) for k, v in feature_boundaries.items()},
        )
        metrics = evaluate_feature_evt_open_set(
            features=open_features,
            labels=open_labels,
            batch_size=int(cfg.evaluation.batch_size),
            student_model=agent.student_model,
            feature_boundaries=feature_boundaries,
            evt_meta=evt_meta,
            class_names=class_names,
            output_dir=output_dir,
            device=device,
            evt_cfg=evt_cfg,
            report_to_stdout=bool(cfg.evaluation.report_to_stdout),
            logger_=logger,
        )


    if tracker:
        tracker.log_metrics(metrics)
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("DKD-FedOS open-set evaluation complete: %s", output_dir)
    return metrics
