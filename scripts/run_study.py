#!/usr/bin/env python3
"""Canonical FedTROS-PR study runner.

Examples:
  python scripts/run_study.py E4-NIID-FOSR --dry-run --stage paper_final
  python scripts/run_study.py E4-NIID-FOSR --stage paper_final --seeds 17 42 73 101 137
  python scripts/run_study.py E6-SCALE --clients 10 50 100 --gpus 0 1 --max-parallel 2
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.infrastructure.logging import configure_logging, get_logger
from src.infrastructure.study import PlannedRun, expand_study_matrix, filter_missing_runs, load_study_config, perform_dry_run
from src.infrastructure.run_id import generate_run_id

logger = get_logger("run_study")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a declarative FedTROS-PR study matrix.")
    p.add_argument("study", help="Study ID/name/YAML (e.g. E4-NIID-FOSR, A1-TEACHER)")
    p.add_argument("--stage", default="development", choices=["smoke","development","tuning","ablation","paper_final","reproduction"])
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--alpha", type=float, nargs="+")
    p.add_argument("--clients", type=int, nargs="+")
    p.add_argument("--method", nargs="+")
    p.add_argument("--dataset", nargs="+")
    p.add_argument("--variant", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume compatible interrupted/failed runs when a checkpoint exists")
    p.add_argument("--force-new", action="store_true", help="Create a fresh run identity even when the same scientific configuration exists")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--gpus", nargs="+", help="GPU indices for independent runs, e.g. --gpus 0 1")
    p.add_argument("--max-parallel", type=int, default=1, help="Maximum simultaneous independent runs")
    p.add_argument("--wandb-mode", choices=["online","offline","disabled"], default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def _matches(value: object, allowed: list[object] | None) -> bool:
    if not allowed:
        return True
    if isinstance(value, float):
        return any(abs(value - float(x)) < 1e-9 for x in allowed)
    return str(value).lower() in {str(x).lower() for x in allowed}


def filter_plans(plans: list[PlannedRun], args: argparse.Namespace) -> list[PlannedRun]:
    return [r for r in plans if (
        _matches(r.alpha, args.alpha)
        and _matches(r.num_clients, args.clients)
        and _matches(r.method, args.method)
        and _matches(r.dataset, args.dataset)
        and _matches(r.variant, args.variant)
    )]


def hydra_args(
    run: PlannedRun,
    wandb_mode: str | None,
    *,
    output_base: Path | None = None,
) -> list[str]:
    args: list[str] = []
    for key, value in run.overrides.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            rendered = "[" + ",".join(repr(x) if isinstance(x, str) else str(x) for x in value) + "]"
        elif value is None:
            rendered = "null"
        else:
            rendered = str(value)
        prefix = "+" if key == "experiment.rerun_token" else ""
        args.append(f"{prefix}{key}={rendered}")
    if wandb_mode:
        args.append(f"tracking.mode={wandb_mode}")
    if output_base is not None:
        args.append(f"output.root_dir={str(output_base.resolve())}")
    return args



def _force_new_identity(run: PlannedRun, token: str) -> PlannedRun:
    overrides = dict(run.overrides)
    overrides["experiment.rerun_token"] = token
    run_id, human_name, config_hash = generate_run_id(overrides, study_id=run.study_id)
    return dataclasses.replace(run, overrides=overrides, run_id=run_id, human_name=human_name, config_hash=config_hash, status="NEW")


def _checkpoint_for(run_dir: Path) -> Path | None:
    for p in (
        run_dir / "checkpoints" / "latest.pt",
        run_dir / "checkpoints" / "fedtros_pr_student_latest.pt",
        run_dir / "latest_checkpoint.pt",
    ):
        if p.exists():
            return p
    return None


def launch_resume(run: PlannedRun, project_root: Path, output_base: Path, gpu: str | None) -> tuple[str, bool, float]:
    run_dir = output_base / "runs" / run.run_id
    checkpoint = _checkpoint_for(run_dir)
    if checkpoint is None:
        logger.warning("No resumable checkpoint for %s; launching from scratch instead.", run.run_id)
        return launch(run, project_root, gpu, None, output_base=output_base)
    cmd = [sys.executable, str(project_root / "scripts" / "resume.py"), str(run_dir)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.perf_counter()
    logger.info("Resuming %s from %s%s", run.run_id, checkpoint.name, f" on GPU {gpu}" if gpu is not None else "")
    proc = subprocess.run(cmd, cwd=project_root, env=env, check=False)
    elapsed = time.perf_counter() - start
    ok = proc.returncode == 0
    (logger.info if ok else logger.error)("%s resume %s in %.1fs", run.run_id, "COMPLETED" if ok else f"FAILED({proc.returncode})", elapsed)
    return run.run_id, ok, elapsed


def launch(
    run: PlannedRun,
    project_root: Path,
    gpu: str | None,
    wandb_mode: str | None,
    *,
    output_base: Path,
) -> tuple[str, bool, float]:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run.py"),
        *hydra_args(run, wandb_mode, output_base=output_base),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.perf_counter()
    logger.info("Launching %s%s", run.run_id, f" on GPU {gpu}" if gpu is not None else "")
    proc = subprocess.run(cmd, cwd=project_root, env=env, check=False)
    elapsed = time.perf_counter() - start
    ok = proc.returncode == 0
    (logger.info if ok else logger.error)("%s %s in %.1fs", run.run_id, "COMPLETED" if ok else f"FAILED({proc.returncode})", elapsed)
    return run.run_id, ok, elapsed


def main() -> int:
    args = parse_args()
    configure_logging()
    root = _ROOT
    try:
        study = load_study_config(args.study, root)
    except FileNotFoundError as exc:
        logger.error("%s", exc); return 2
    plans = expand_study_matrix(study, stage=args.stage, seeds=args.seeds, project_root=root)
    plans = filter_plans(plans, args)
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    summary = perform_dry_run(plans, output_base_dir=out)
    if args.dry_run:
        summary.print_summary(); return 0
    # A completed scientific identity is immutable.  Re-execution requires an
    # explicit fresh identity via --force-new; --only-missing remains a readable
    # way to state the default safe behavior.
    if not args.force_new:
        plans = filter_missing_runs(plans, output_base_dir=out)
    if args.force_new:
        token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        plans = [_force_new_identity(run, token) for run in plans]
    if not plans:
        logger.info("No runs need execution."); return 0

    max_parallel = max(1, int(args.max_parallel))
    gpus = list(args.gpus or [])
    if max_parallel > 1 and not gpus:
        logger.warning("--max-parallel > 1 without --gpus: independent runs may contend for the same device.")
    if args.resume and max_parallel > 1:
        logger.warning("--resume uses sequential execution to protect checkpoint state; forcing --max-parallel=1")
        max_parallel = 1
    failures = 0
    if max_parallel == 1:
        for i, run in enumerate(plans):
            gpu = gpus[i % len(gpus)] if gpus else None
            run_dir = out / "runs" / run.run_id
            status = str(getattr(run, "status", "NEW")).upper()
            if args.resume and run_dir.exists() and _checkpoint_for(run_dir) is not None:
                _, ok, _ = launch_resume(run, root, out, gpu)
            else:
                _, ok, _ = launch(run, root, gpu, args.wandb_mode, output_base=out)
            if not ok:
                failures += 1
                if not args.continue_on_error:
                    break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = []
            for i, run in enumerate(plans):
                gpu = gpus[i % len(gpus)] if gpus else None
                futures.append(
                    pool.submit(
                        launch,
                        run,
                        root,
                        gpu,
                        args.wandb_mode,
                        output_base=out,
                    )
                )
            for fut in concurrent.futures.as_completed(futures):
                _, ok, _ = fut.result()
                if not ok:
                    failures += 1
    logger.info("Study %s finished | requested=%d failures=%d", study.get("study_id", args.study), len(plans), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
