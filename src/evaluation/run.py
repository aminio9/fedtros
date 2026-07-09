from __future__ import annotations

import csv
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
from src.openset.digos_eval import calibrate_fed_digos, evaluate_fed_digos
from src.tracking.local import LocalRunTracker
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


def build_agent(cfg: DictConfig, device: torch.device) -> Agent:
    return Agent(OpenSetQChainModelFactory(cfg.model), cfg.training, device=device)


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
    output_dir: Path | None = None,
    server_round: int | None = None,
    save_scores: bool | None = None,
    append_round_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate DKD-FedOS with the Fed-DiGOS open-set backend.

    Fed-DiGOS uses the federated student classifier plus its disentangled OSR
    generator branch.  The local RL teacher generator is intentionally not used
    here.  If the branch is missing, fail loudly rather than quietly pretending
    another doomed feature-distance threshold is science.
    """
    evt_cfg = cfg.open_set.evt
    fed_digos_cfg = getattr(cfg.open_set, "fed_digos", None)
    if not bool(getattr(evt_cfg, "enabled", False)):
        logger.info("DKD-FedOS open-set evaluation skipped: EVT disabled.")
        return {}
    backend = str(getattr(evt_cfg, "backend", "fed_digos")).lower()
    backend = {"digos": "fed_digos", "student_digos": "fed_digos", "fed_digos": "fed_digos"}.get(backend, backend)
    if backend != "fed_digos" or not bool(getattr(fed_digos_cfg, "enabled", False)):
        raise ValueError(
            "Fed-DiGOS is now the only DKD-FedOS open-set backend. Set "
            "open_set.evt.backend=fed_digos and open_set.fed_digos.enabled=true."
        )

    base_output_dir = resolve_path(project_root, cfg.evaluation.output_dir)
    output_dir = Path(output_dir) if output_dir is not None else base_output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_scores is None:
        save_scores = True

    agent = build_agent(cfg, device)
    student_ckpt = resolve_path(project_root, Path(str(cfg.checkpointing.dir)) / "dkd_fedos_student_latest.pt")
    if not student_ckpt.exists():
        raise FileNotFoundError(f"DKD-FedOS student checkpoint not found: {student_ckpt}")
    payload = torch.load(student_ckpt, map_location=device, weights_only=False)
    state = payload.get("student_model")
    if state is None:
        raise KeyError(f"Missing 'student_model' in DKD-FedOS checkpoint: {student_ckpt}")
    agent.student_model.load_state_dict(state, strict=True)
    agent.student_model.to(device).eval()
    if not bool(getattr(agent.student_model, "osr_enabled", False)):
        raise RuntimeError(
            "Fed-DiGOS checkpoint has no student OSR branch. Run with "
            "training.dkd_student_osr_enabled=true."
        )

    class_names = load_class_names(resolve_path(project_root, cfg.evaluation.class_names), int(cfg.model.num_actions))
    calibration_features, calibration_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.validation_data), map_location="cpu"
    )
    open_features, open_labels = load_tensor_dataset(
        resolve_path(project_root, cfg.evaluation.open_set_data), map_location="cpu"
    )
    unknown_label_id = int(getattr(evt_cfg, "unknown_label_id", -1))
    num_unknown_train = int((calibration_labels == unknown_label_id).sum().item())
    num_unknown_test = int((open_labels == unknown_label_id).sum().item())
    unknown_label_names = _open_set_unknown_names(cfg, project_root=project_root)
    logger.info(
        "FED-DIGOS OPEN-SET ACTIVE | known_classes=%s | unknown_labels=%s | unknown_label_id=%s | "
        "num_actions=%d | calibration_samples=%d | calibration_unknown=%d | open_test_samples=%d | "
        "open_test_unknown=%d | osr_latent_dim=%d | pseudo_unknown=%s | energy=%s | prototype=%s | server_round=%s",
        [class_names[k] for k in sorted(class_names)],
        unknown_label_names or ["held_out_unknown"],
        unknown_label_id,
        int(cfg.model.num_actions),
        int(calibration_labels.numel()),
        num_unknown_train,
        int(open_labels.numel()),
        num_unknown_test,
        int(getattr(agent.student_model, "osr_latent_dim", 0)),
        bool(getattr(getattr(fed_digos_cfg, "pseudo_unknown", None), "enabled", True)),
        bool(getattr(getattr(fed_digos_cfg, "energy", None), "enabled", True)),
        bool(getattr(getattr(fed_digos_cfg, "prototype", None), "enabled", True)),
        "final" if server_round is None else int(server_round),
    )
    if num_unknown_train != 0:
        raise ValueError("Fed-DiGOS EVT calibration data must contain known classes only; found unknown labels.")

    evt_models, prototype_bank, calibration_df, meta = calibrate_fed_digos(
        calibration_features,
        calibration_labels,
        student_model=agent.student_model,
        batch_size=int(cfg.evaluation.batch_size),
        device=device,
        cfg=fed_digos_cfg,
        logger_=logger,
    )
    evt_output_dir = output_dir / "evt"
    evt_output_dir.mkdir(parents=True, exist_ok=True)
    (evt_output_dir / "fed_digos_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    metrics = evaluate_fed_digos(
        open_features,
        open_labels,
        student_model=agent.student_model,
        batch_size=int(cfg.evaluation.batch_size),
        device=device,
        cfg=fed_digos_cfg,
        class_names=class_names,
        output_dir=output_dir,
        evt_models=evt_models,
        prototype_bank=prototype_bank,
        calibration_df=calibration_df,
        logger_=logger,
        report_to_stdout=bool(cfg.evaluation.report_to_stdout),
    )

    if server_round is not None:
        metrics["server_round"] = int(server_round)
        metrics["open_set/server_round"] = float(server_round)

    if append_round_metrics and server_round is not None:
        curve_path = base_output_dir / "open_set_round_metrics.csv"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        round_row = {
            "round": int(server_round),
            "backend": "fed_digos",
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
            "Fed-DiGOS round metrics appended | round=%s | AUROC=%.4f | UnknownRecall=%.4f | KnownFU=%.4f",
            int(server_round),
            float(metrics.get("openset_auroc", 0.0)),
            float(metrics.get("openset_unknown_recall", 0.0)),
            float(metrics.get("openset_known_false_unknown_rate", 0.0)),
        )

    if tracker:
        tracker.log_metrics(metrics)
    metrics_payload = json.dumps(metrics, indent=2, sort_keys=True)
    (output_dir / "evaluation_metrics.json").write_text(metrics_payload, encoding="utf-8")
    (output_dir / "open_set_metrics.json").write_text(metrics_payload, encoding="utf-8")
    logger.info("Fed-DiGOS open-set evaluation complete: %s", output_dir)
    return metrics
