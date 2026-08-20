#!/usr/bin/env python3
"""Resume an interrupted FedTROS-PR VCT federated run exactly when private state exists."""
from __future__ import annotations

import argparse
import random
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.evaluation import run_evaluation, run_prototype_rank_evaluation
from src.experiment import create_run_services
from src.federated import run_federated_simulation
from src.infrastructure.checkpointing import IncompatibleCheckpointError
from src.infrastructure.logging import configure_logging, get_logger
from src.infrastructure.manifests import RunStatus, update_run_manifest_status
from src.utils.utils import resolve_device_from_config

logger=get_logger("resume")


def _checkpoint(run_dir: Path) -> Path:
    for p in (
        run_dir/"checkpoints"/"latest.pt",
        run_dir/"checkpoints"/"fedtros_pr_student_latest.pt",
        run_dir/"checkpoints"/"global_model_latest.pt",
        run_dir/"latest_checkpoint.pt",  # migration-era location only
    ):
        if p.exists(): return p
    raise FileNotFoundError(f"No resumable global checkpoint found in {run_dir}")


def _config(run_dir: Path) -> Path:
    for p in (run_dir/"config"/"resolved_config.yaml",run_dir/"resolved_config.yaml"):
        if p.exists(): return p
    raise FileNotFoundError(f"Resolved config not found in {run_dir}")


def _summary(run_dir: Path) -> dict:
    out={}
    for p in (run_dir/"metrics"/"final_metrics.json",run_dir/"evaluation_metrics.json",run_dir/"open_set_metrics.json",run_dir/"federated_summary.json"):
        if p.exists():
            try:
                v=json.loads(p.read_text(encoding="utf-8"))
                if isinstance(v,dict): out.update(v)
            except Exception: pass
    return out


def _validate_private_states(run_dir: Path, num_clients: int, saved_round: int, config_hash: str = "") -> None:
    private_dir=run_dir/"checkpoints"/"private"
    missing=[]; stale=[]
    for cid in range(1,num_clients+1):
        p=private_dir/f"client_{cid}_latest.pt"
        if not p.exists(): missing.append(str(cid)); continue
        payload=torch.load(p,map_location="cpu",weights_only=False)
        if payload.get("schema_version")!=2 or payload.get("teacher_type")!="variational_classifier":
            raise IncompatibleCheckpointError(f"Incompatible private VCT checkpoint: {p}")
        if config_hash and payload.get("config_hash") and str(payload.get("config_hash")) != str(config_hash):
            raise IncompatibleCheckpointError(
                f"Private checkpoint config hash mismatch for client {cid}: {p}"
            )
        if int(payload.get("round",-1)) < saved_round:
            stale.append(f"{cid}:{payload.get('round')}")
    if missing or stale:
        raise IncompatibleCheckpointError(
            "Exact VCT resume is not possible because client-private state is incomplete. "
            f"missing_clients={missing}, stale_clients={stale}, required_round={saved_round}. "
            "Start a fresh run rather than presenting a student-only continuation as exact resume."
        )


def resume_run(target: str|Path, *, target_rounds: int|None=None, device: str|None=None) -> Path:
    p=Path(target)
    if p.is_file(): run_dir=p.parent.parent if p.parent.name=="checkpoints" else p.parent; ckpt=p
    else: run_dir=p; ckpt=_checkpoint(run_dir)
    run_dir=run_dir.resolve(); configure_logging(run_dir=run_dir)
    cfg=OmegaConf.load(_config(run_dir))
    if device: OmegaConf.update(cfg,"runtime.device_prefer",device,force_add=True)

    payload=torch.load(ckpt,map_location="cpu",weights_only=False)
    if payload.get("schema_version")!=2 or str(payload.get("teacher_type",""))!="variational_classifier":
        raise IncompatibleCheckpointError(
            f"Checkpoint {ckpt} is not a Schema-v2 FedTROS-PR VCT checkpoint. Legacy DQN-era checkpoints cannot be resumed."
        )
    saved_round=int(payload.get("round",payload.get("epoch",0)))
    resolved_hash = str(OmegaConf.select(cfg, "experiment.config_hash", default="") or "")
    checkpoint_hash = str(payload.get("config_hash", "") or "")
    if resolved_hash and checkpoint_hash and resolved_hash != checkpoint_hash:
        raise IncompatibleCheckpointError(
            f"Checkpoint/config hash mismatch: checkpoint={checkpoint_hash[:12]} resolved={resolved_hash[:12]}"
        )
    intended_total=int(target_rounds if target_rounds is not None else OmegaConf.select(cfg,"federated.num_rounds",default=100))
    if saved_round>=intended_total:
        logger.info("Checkpoint already reached round %d/%d; nothing to resume.",saved_round,intended_total); return run_dir

    num_clients=int(OmegaConf.select(cfg,"federated.num_clients",default=1))
    strategy=str(OmegaConf.select(cfg,"federated.strategy.name",default="")).lower()
    if strategy=="fedtros_pr": _validate_private_states(run_dir,num_clients,saved_round,resolved_hash)

    rng_state = payload.get("rng_state") or {}
    if rng_state:
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("torch_cpu") is not None:
            torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and rng_state.get("torch_cuda"):
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    remaining=intended_total-saved_round
    OmegaConf.update(cfg,"federated.resume_from",str(ckpt),force_add=True)
    OmegaConf.update(cfg,"federated.resume_round_offset",saved_round,force_add=True)
    OmegaConf.update(cfg,"federated.num_rounds",remaining,force_add=True)
    # server.num_rounds is an interpolation in the canonical config; explicit update is safe after loading resolved YAML.
    OmegaConf.update(cfg,"federated.server.num_rounds",remaining,force_add=True)

    run_id=run_dir.name
    study_id=str(OmegaConf.select(cfg,"experiment.id",default="E0-VERIFY"))
    stage=str(OmegaConf.select(cfg,"stage",default="development"))
    services=create_run_services(cfg,run_dir=run_dir,run_id=run_id,human_name=run_id,study_id=study_id,stage=stage,resume=True)
    services.log_resume_config(cfg, resumed_from_round=saved_round)
    update_run_manifest_status(run_dir,RunStatus.RESUMED,tracker_run_id=services.tracker_run_id)
    services.set_status(RunStatus.RESUMED.value)
    logger.info("Resuming %s from round %d through %d (%d Flower rounds)",run_id,saved_round+1,intended_total,remaining)

    try:
        run_federated_simulation(cfg,project_root=_ROOT,tracker=services)
        dev=torch.device(device) if device else resolve_device_from_config(cfg)
        open_enabled=bool(OmegaConf.select(cfg,"open_set.enabled",default=False)) or str(OmegaConf.select(cfg,"evaluation.mode",default="closed_set")).lower()=="open_set"
        if open_enabled:
            run_prototype_rank_evaluation(cfg,project_root=_ROOT,device=dev,tracker=services)
        else:
            run_evaluation(cfg,project_root=_ROOT,device=dev,tracker=services)
        services.set_summary(_summary(run_dir))
        update_run_manifest_status(run_dir,RunStatus.COMPLETED,tracker_run_id=services.tracker_run_id)
        services.finish(status=RunStatus.COMPLETED.value)
        return run_dir
    except KeyboardInterrupt:
        update_run_manifest_status(run_dir,RunStatus.INTERRUPTED,tracker_run_id=services.tracker_run_id)
        services.finish(status=RunStatus.INTERRUPTED.value); raise
    except Exception as exc:
        logger.exception("Resume failed: %s",exc)
        update_run_manifest_status(run_dir,RunStatus.FAILED,error=str(exc),tracker_run_id=services.tracker_run_id)
        services.finish(status=RunStatus.FAILED.value); raise


def main() -> int:
    ap=argparse.ArgumentParser(description="Resume a compatible FedTROS-PR VCT run")
    ap.add_argument("target",help="Run directory or global checkpoint")
    ap.add_argument("--target-rounds",type=int)
    ap.add_argument("--device")
    args=ap.parse_args()
    try: resume_run(args.target,target_rounds=args.target_rounds,device=args.device); return 0
    except Exception as exc: logger.error("Resumption failed: %s",exc); return 1


if __name__=="__main__": raise SystemExit(main())
