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


def run_prototype_rank_evaluation(
    cfg: DictConfig,
    *,
    project_root: Path,
    device: torch.device,
    tracker: MetricsSink | None = None,
    output_dir: Path | None = None,
    server_round: int | None = None,
    save_scores: bool | None = None,
    append_round_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate FedTROS-PR with the known-only Prototype-Rank rejection stage.

    The private VCT remains client-local. Prototype-Rank consumes the configured
    student feature source and never uses final unknown samples for fitting or
    threshold calibration.
    """
    open_set_cfg = cfg.open_set
    prototype_rank_cfg = _compose_prototype_rank_runtime_config(open_set_cfg)
    if not bool(getattr(open_set_cfg, "enabled", False)):
        logger.info("FedTROS-PR open-set evaluation skipped: open_set.enabled=false.")
        return {}
    detector = str(getattr(open_set_cfg, "detector", getattr(open_set_cfg, "method", "prototype_rank"))).lower()
    if detector != "prototype_rank" or not bool(getattr(prototype_rank_cfg, "enabled", False)):
        raise ValueError(
            "FedTROS-PR open-set evaluation requires open_set.detector=prototype_rank "
            "and open_set.prototype_rank.enabled=true."
        )

    base_output_dir = resolve_path(project_root, cfg.evaluation.output_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_scores is None:
        save_scores = True

    agent = build_agent(cfg, device)
    student_ckpt = _resolve_prototype_rank_checkpoint(cfg, project_root=project_root)
    payload = torch.load(student_ckpt, map_location=device, weights_only=False)
    state = payload.get("student_model")
    if state is None:
        raise KeyError(f"Missing 'student_model' in checkpoint: {student_ckpt}")
    agent.student_model.load_state_dict(state, strict=True)
    agent.student_model.to(device).eval()
    feature_source = str(getattr(getattr(prototype_rank_cfg, "prototype", None), "feature_source", "student_embedding")).lower()
    if feature_source in {"osr_mu", "osr_embedding"} and not bool(getattr(agent.student_model, "osr_enabled", False)):
        raise RuntimeError(
            "Prototype-Rank feature_source=osr_mu requires the optional student OSR branch. "
            "Use feature_source=student_embedding or enable the branch for A5 evaluation."
        )

    class_names = load_class_names(resolve_path(project_root, cfg.evaluation.class_names), int(cfg.model.num_classes))
    calibration_features, calibration_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.validation_data), map_location="cpu"
    )
    open_features, open_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.open_set_data), map_location="cpu"
    )
    unknown_label_id = int(getattr(open_set_cfg, "unknown_label_id", -1))
    num_unknown_train = int((calibration_labels == unknown_label_id).sum().item())
    num_unknown_test = int((open_labels == unknown_label_id).sum().item())
    unknown_label_names = _open_set_unknown_names(cfg, project_root=project_root)
    logger.info(
        "FEDTROS-PR PROTOTYPE-RANK ACTIVE | known_classes=%s | unknown_labels=%s | unknown_label_id=%s | "
        "num_classes=%d | calibration_samples=%d | calibration_unknown=%d | open_test_samples=%d | "
        "open_test_unknown=%d | osr_latent_dim=%d | boundary_samples=%s | energy=%s | prototype=%s | server_round=%s",
        [class_names[k] for k in sorted(class_names)],
        unknown_label_names or ["held_out_unknown"],
        unknown_label_id,
        int(cfg.model.num_classes),
        int(calibration_labels.numel()),
        num_unknown_train,
        int(open_labels.numel()),
        num_unknown_test,
        int(getattr(agent.student_model, "osr_latent_dim", 0)),
        bool(getattr(getattr(prototype_rank_cfg, "boundary_samples", None), "enabled", True)),
        bool(getattr(getattr(prototype_rank_cfg, "energy", None), "enabled", True)),
        bool(getattr(getattr(prototype_rank_cfg, "prototype", None), "enabled", True)),
        "final" if server_round is None else int(server_round),
    )
    if num_unknown_train != 0:
        raise ValueError("Prototype-Rank calibration data must contain known classes only; found unknown labels.")

    prototype_bank, calibration_df, meta = calibrate_prototype_rank(
        calibration_features,
        calibration_labels,
        student_model=agent.student_model,
        batch_size=int(cfg.evaluation.batch_size),
        device=device,
        cfg=prototype_rank_cfg,
        logger_=logger,
    )
    pr_output_dir = output_dir / "artifacts" / "prototype_rank"
    pr_output_dir.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta, indent=2, sort_keys=True)
    (pr_output_dir / "prototype_rank_meta.json").write_text(meta_json, encoding="utf-8")
    metrics = evaluate_prototype_rank(
        open_features,
        open_labels,
        student_model=agent.student_model,
        batch_size=int(cfg.evaluation.batch_size),
        device=device,
        cfg=prototype_rank_cfg,
        class_names=class_names,
        output_dir=output_dir,
        prototype_bank=prototype_bank,
        calibration_df=calibration_df,
        logger_=logger,
        report_to_stdout=bool(cfg.evaluation.report_to_stdout),
    )

    if bool(getattr(cfg.evaluation, "export_latent_embeddings", True)):
        scores_path = output_dir / "predictions" / "open_set_scores.csv"
        if not scores_path.exists():
            scores_path = output_dir / "open_set_scores.csv"
        if not scores_path.exists():
            raise FileNotFoundError(
                "Prototype-Rank evaluation did not write open_set_scores.csv required for "
                "the latent projection artifact."
            )
        export_prototype_rank_projection(
            model=agent.student_model,
            features=open_features,
            labels=open_labels,
            class_names=class_names,
            prototype_bank=prototype_bank,
            scores=pd.read_csv(scores_path),
            output_path=output_dir / "artifacts" / "prototype_rank_latent_projection.csv",
            batch_size=int(cfg.evaluation.latent_embeddings_batch_size),
            max_points=int(cfg.evaluation.latent_embeddings_max_points),
            feature_source=feature_source,
        )

    if server_round is not None:
        metrics["server_round"] = int(server_round)
        metrics["open_set/server_round"] = float(server_round)

    if append_round_metrics and server_round is not None:
        curve_path = base_output_dir / "metrics" / "open_set_round_metrics.csv"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        round_row = {
            "round": int(server_round),
            "backend": "prototype_rank",
            "openset_known_acc": float(metrics.get("openset_known_acc", 0.0)),
            "openset_unknown_recall": float(metrics.get("openset_unknown_recall", 0.0)),
            "openset_unknown_f1": float(metrics.get("openset_unknown_f1", 0.0)),
            "openset_f1_macro": float(metrics.get("openset_f1_macro", 0.0)),
            "openset_overall_acc": float(metrics.get("openset_overall_acc", 0.0)),
            "openset_auroc": float(metrics.get("openset_auroc", 0.0)),
            "openset_auprc": float(metrics.get("openset_auprc", 0.0)),
            "openset_fpr95": float(metrics.get("openset_fpr95", 1.0)),
            "openset_known_false_unknown_rate": float(metrics.get("openset_known_false_unknown_rate", 0.0)),
            "openset_rejected_by_gen": float(metrics.get("openset_rejected_by_gen", 0.0)),
            "openset_rejected_by_energy": float(metrics.get("openset_rejected_by_energy", 0.0)),
            "openset_rejected_by_prototype": float(metrics.get("openset_rejected_by_prototype", 0.0)),
        }
        write_header = not curve_path.exists()
        with curve_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(round_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(round_row)
        logger.info(
            "Prototype-Rank round metrics appended | round=%s | AUROC=%.4f | UnknownRecall=%.4f | KnownFU=%.4f",
            int(server_round),
            float(metrics.get("openset_auroc", 0.0)),
            float(metrics.get("openset_unknown_recall", 0.0)),
            float(metrics.get("openset_known_false_unknown_rate", 0.0)),
        )

    # Standardized B9 Metric Namespaces
    metrics["open_set/auroc"] = float(metrics.get("openset_auroc", 0.0))
    metrics["open_set/auprc"] = float(metrics.get("openset_auprc", 0.0))
    metrics["open_set/fpr_at_95_tpr"] = float(metrics.get("openset_fpr95", 1.0))
    metrics["open_set/unknown_f1"] = float(metrics.get("openset_unknown_f1", 0.0))
    metrics["open_set/unknown_recall"] = float(metrics.get("openset_unknown_recall", 0.0))
    metrics["open_set/known_false_unknown_rate"] = float(metrics.get("openset_known_false_unknown_rate", 0.0))
    metrics["closed_set/accuracy"] = float(metrics.get("openset_known_acc_before", 0.0))

    if tracker:
        tracker.log_metrics(metrics)
    metrics_payload = json.dumps(metrics, indent=2, sort_keys=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "evaluation_metrics.json").write_text(metrics_payload, encoding="utf-8")
    (metrics_dir / "open_set_metrics.json").write_text(metrics_payload, encoding="utf-8")
    logger.info("FedTROS-PR Prototype-Rank evaluation complete: %s", output_dir)
    return metrics
