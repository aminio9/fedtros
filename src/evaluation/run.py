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
from src.tracking.local import LocalRunTracker
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def build_agent(cfg: DictConfig, device: torch.device) -> Agent:
    return Agent(OpenSetQChainModelFactory(cfg.model), cfg.training, device=device)


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
        latent_features = closed_features
        latent_labels = closed_labels
        open_set_data_path = resolve_path(project_root, cfg.evaluation.open_set_data)
        if open_set_data_path.exists():
            open_features, open_labels = load_tensor_dataset(
                open_set_data_path,
                map_location="cpu",
            )
            latent_features = torch.cat([latent_features, open_features], dim=0)
            latent_labels = torch.cat([latent_labels, open_labels], dim=0)

        export_latent_embeddings(
            prior_net=agent.prior_net,
            features=latent_features,
            labels=latent_labels,
            class_names=class_names,
            output_path=resolve_path(project_root, cfg.evaluation.latent_embeddings_output),
            batch_size=int(cfg.evaluation.latent_embeddings_batch_size),
            max_points=int(cfg.evaluation.latent_embeddings_max_points),
        )

    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Evaluation complete. Metrics saved under %s", output_dir)
    return all_metrics
