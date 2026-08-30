from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from src.models.bundle import FedTROSModelBundle as Agent
from src.artifacts.embeddings import export_latent_embeddings, export_prototype_rank_projection
from src.checkpointing.checkpoints import load_agent_checkpoint
from src.data.io import load_tensor_dataset
from src.evaluation.closed_set import evaluate_closed_set, load_class_names
from src.models.models import FedTROSModelFactory
from src.openset.prototype_rank_pipeline import calibrate_prototype_rank, evaluate_prototype_rank
from src.experiment.run_services import MetricsSink
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def _compose_prototype_rank_runtime_config(open_set_cfg: DictConfig) -> DictConfig:
    """Attach protocol-level calibration to the detector runtime subtree."""
    detector_cfg = getattr(open_set_cfg, "prototype_rank", None)
    if detector_cfg is None:
        return OmegaConf.create({})
    protocol_cfg = OmegaConf.create({
        "unknown_label_id": int(getattr(open_set_cfg, "unknown_label_id", -1)),
        "open_set_label_id": int(getattr(open_set_cfg, "open_set_label_id", 99)),
        "calibration": OmegaConf.to_container(
            getattr(open_set_cfg, "calibration", OmegaConf.create({})), resolve=True
        ),
    })
    return OmegaConf.merge(protocol_cfg, detector_cfg)


def _resolve_prototype_rank_checkpoint(cfg: DictConfig, *, project_root: Path) -> Path:
    """Select the method-specific student checkpoint, then the canonical fallback."""
    checkpoint_dir = Path(str(cfg.checkpointing.dir))
    candidates = [
        resolve_path(project_root, checkpoint_dir / "fedtros_pr_student_latest.pt"),
        resolve_path(project_root, cfg.evaluation.checkpoint_path),
        resolve_path(project_root, checkpoint_dir / "global_model_latest.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No student checkpoint found for Prototype-Rank evaluation; tried: {attempted}")


def build_agent(cfg: DictConfig, device: torch.device) -> Agent:
    from src.models.bundle import FedTROSModelBundle

    return FedTROSModelBundle(FedTROSModelFactory(cfg.model), cfg.training, device=device)


def _open_set_unknown_names(cfg: DictConfig, *, project_root: Path) -> list[str]:
    """Load held-out unknown label names from preprocessing metadata when available."""
    try:
        metadata_path = resolve_path(project_root, cfg.dataset.preprocessing.output_dir) / "preprocess_metadata.json"
        if not metadata_path.exists():
            return []
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = metadata.get("unknown_labels", [])
        if isinstance(values, list):
            return [str(value) for value in values]
    except Exception:
        logger.debug("Could not read unknown label names from preprocessing metadata.", exc_info=True)
    return []


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
    tracker: MetricsSink | None = None,
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
        int(cfg.model.num_classes),
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
    if bool(cfg.evaluation.export_latent_embeddings):
        latent_features, latent_labels, source = _latent_export_tensor(
            cfg,
            project_root=project_root,
            closed_features=closed_features,
            closed_labels=closed_labels,
        )
        logger.info("Exporting latent embeddings from the %s evaluation tensor.", source)

        export_latent_embeddings(
            model=agent.student_model,
            features=latent_features,
            labels=latent_labels,
            class_names=class_names,
            output_path=resolve_path(project_root, cfg.evaluation.latent_embeddings_output),
            batch_size=int(cfg.evaluation.latent_embeddings_batch_size),
            max_points=int(cfg.evaluation.latent_embeddings_max_points),
            source=source,
        )

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "evaluation_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Evaluation complete. Metrics saved under %s", output_dir)
    return all_metrics


def run_open_set_evaluation(
    cfg,
    *,
    project_root,
    device,
    tracker = None,
    output_dir = None,
    server_round = None,
    save_scores = None,
    append_round_metrics = False,
):
    open_set_cfg = cfg.open_set
    if not bool(getattr(open_set_cfg, "enabled", False)):
        logger.info("Open-set evaluation skipped: open_set.enabled=false.")
        return {}

    detector = str(getattr(open_set_cfg, "detector", getattr(open_set_cfg, "method", "multicenter_conformal"))).lower()
    logger.info("Open-set detector=%s", detector)
    canonical = str(getattr(getattr(cfg, "method", None), "canonical", "false")).lower() == "true"
    
    if canonical and detector != "multicenter_conformal":
        raise ValueError("CanonicalConfigurationError: canonical=true MUST resolve to backend=multicenter_conformal.")

    base_output_dir = resolve_path(project_root, cfg.evaluation.output_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_scores is None: save_scores = True

    agent = build_agent(cfg, device)
    student_ckpt = _resolve_prototype_rank_checkpoint(cfg, project_root=project_root)
    payload = torch.load(student_ckpt, map_location=device, weights_only=False)
    state = payload.get("student_model")
    if state is None:
        raise KeyError(f"Missing 'student_model' in checkpoint: {student_ckpt}")
    agent.student_model.load_state_dict(state, strict=True)
    agent.student_model.to(device).eval()

    class_names = load_class_names(resolve_path(project_root, cfg.evaluation.class_names), int(cfg.model.num_classes))
    calibration_features, calibration_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.validation_data), map_location="cpu"
    )
    open_features, open_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.open_set_data), map_location="cpu"
    )

    if detector == "multicenter_conformal":
        logger.info("Evaluating Multicenter Conformal | canonical=%s", canonical)
        from src.openset.multicenter_conformal_pipeline import calibrate_multicenter_conformal, evaluate_multicenter_conformal
        conformal_meta, df_calib, meta = calibrate_multicenter_conformal(
            calibration_features, calibration_labels,
            student_model=agent.student_model, batch_size=int(cfg.evaluation.batch_size),
            device=device, cfg=open_set_cfg, logger_=logger, output_dir=output_dir
        )
        
        pr_output_dir = output_dir / "artifacts" / "multicenter_conformal"
        pr_output_dir.mkdir(parents=True, exist_ok=True)
        import json
        meta_json = json.dumps(meta, indent=2, sort_keys=True)
        (pr_output_dir / "mc_meta.json").write_text(meta_json, encoding="utf-8")
        
        metrics = evaluate_multicenter_conformal(
            open_features, open_labels,
            student_model=agent.student_model, batch_size=int(cfg.evaluation.batch_size),
            device=device, cfg=open_set_cfg, class_names=class_names,
            output_dir=output_dir, conformal_meta=conformal_meta, logger_=logger
        )
        all_metrics = metrics
    else:
        # Legacy Prototype Rank
        logger.info("Evaluating Legacy Prototype-Rank")
        from src.openset.prototype_rank_pipeline import calibrate_prototype_rank, evaluate_prototype_rank, export_prototype_rank_projection
        prototype_rank_cfg = _compose_prototype_rank_runtime_config(open_set_cfg)
        prototype_bank, calibration_df, meta = calibrate_prototype_rank(
            calibration_features, calibration_labels,
            student_model=agent.student_model, batch_size=int(cfg.evaluation.batch_size),
            device=device, cfg=prototype_rank_cfg, logger_=logger
        )
        
        pr_output_dir = output_dir / "artifacts" / "prototype_rank"
        pr_output_dir.mkdir(parents=True, exist_ok=True)
        import json
        meta_json = json.dumps(meta, indent=2, sort_keys=True)
        (pr_output_dir / "prototype_rank_meta.json").write_text(meta_json, encoding="utf-8")
        
        metrics = evaluate_prototype_rank(
            open_features, open_labels,
            student_model=agent.student_model, batch_size=int(cfg.evaluation.batch_size),
            device=device, cfg=prototype_rank_cfg, class_names=class_names,
            output_dir=output_dir, prototype_bank=prototype_bank, calibration_df=calibration_df,
            logger_=logger, report_to_stdout=bool(cfg.evaluation.report_to_stdout), meta=meta
        )
        all_metrics = metrics

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    import json
    (metrics_dir / "evaluation_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Evaluation complete. Metrics saved under %s", output_dir)
    return all_metrics


