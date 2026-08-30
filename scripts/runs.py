#!/usr/bin/env python3
"""Inspect FedTROS-MC runs and compare them with the declarative study contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.loaders import load_run
from src.analysis.query import query_runs
from src.infrastructure.study import expand_study_matrix, load_study_config, perform_dry_run


def _runs(args: argparse.Namespace, status: str | None = None):
    return query_runs(
        study=getattr(args, "study", None), stage=getattr(args, "stage", None),
        method=getattr(args, "method", None), status=status,
        outputs_dir=Path(args.outputs_dir),
    )


def _print_runs(runs) -> None:
    if not runs:
        print("No matching runs."); return
    print(f"{'Status':<12} {'Study':<20} {'Stage':<12} {'Method':<18} {'Data':<12} {'a':<6} {'C':<5} {'Seed':<6} Run ID")
    print("-" * 132)
    for r in runs:
        print(f"{r.status:<12} {r.study:<20} {r.stage:<12} {r.method:<18} {r.dataset:<12} {r.alpha:<6g} {r.num_clients:<5} {r.seed:<6} {r.run_id}")


def cmd_list(args: argparse.Namespace) -> int:
    _print_runs(_runs(args, status=args.status)); return 0


def cmd_status(args: argparse.Namespace, statuses: set[str]) -> int:
    runs=[r for r in _runs(args, status=None) if r.status.upper() in statuses]
    _print_runs(runs)
    return 0


def _run_path(outputs_dir: Path, target: str) -> Path:
    p=Path(target)
    if p.is_dir(): return p
    for candidate in (outputs_dir/"runs"/target, outputs_dir/target):
        if candidate.is_dir(): return candidate
    raise FileNotFoundError(target)


def cmd_show(args: argparse.Namespace) -> int:
    try: p=_run_path(Path(args.outputs_dir), args.target)
    except FileNotFoundError:
        print(f"Run not found: {args.target}"); return 1
    r=load_run(p)
    print("\nFedTROS-MC run")
    print("="*92)
    for label,value in (
        ("Run ID",r.run_id),("Study",r.study),("Stage",r.stage),("Status",r.status),
        ("Method",r.method),("Dataset",r.dataset),("Alpha",r.alpha),("IID",r.iid),
        ("Clients",r.num_clients),("Seed",r.seed),("Unknown labels",r.unknown_labels),
        ("Variant",r.variant),("Config hash",r.config_hash),("Split hash",r.split_hash),
        ("Git commit",r.git_commit),("Directory",r.run_dir),
    ): print(f"{label:<18}: {value}")
    if r.metrics:
        print("\nFinal metrics")
        for k,v in sorted(r.metrics.items()):
            if isinstance(v,(int,float,str,bool)): print(f"  {k:<45} {v}")
    print("\nArtifacts")
    print(f"  round metrics      : {len(r.history)} rows")
    print(f"  open-set scores    : {len(r.scores)} rows")
    print(f"  client metrics     : {len(r.client_metrics)} rows")
    print(f"  communication      : {len(r.communication)} rows")
    print(f"  runtime            : {len(r.runtime)} rows")
    manifest=p/"metadata"/"run_manifest.json"
    if manifest.exists():
        try:
            data=json.loads(manifest.read_text(encoding="utf-8")); err=data.get("error") or data.get("error_message")
            if err: print(f"\nError: {err}")
        except Exception: pass
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    runs=_runs(args,status=None)
    by: dict[str,dict[str,int]]={}
    for r in runs: by.setdefault(r.study,{})[r.status]=by.setdefault(r.study,{}).get(r.status,0)+1
    print(f"{'Study':<24} {'Total':<7} {'Completed':<11} {'Failed':<8} {'Interrupted':<12} Other")
    print("-"*76)
    for study,counts in sorted(by.items()):
        total=sum(counts.values()); comp=counts.get("COMPLETED",0); fail=counts.get("FAILED",0); intr=counts.get("INTERRUPTED",0)
        print(f"{study:<24} {total:<7} {comp:<11} {fail:<8} {intr:<12} {total-comp-fail-intr}")
    if not runs: print("No runs found.")
    return 0


def cmd_resumable(args: argparse.Namespace) -> int:
    rows=[]
    for r in _runs(args,status=None):
        if r.status.upper() not in {"FAILED","INTERRUPTED","RESUMED"}: continue
        ckpts=[r.run_dir/"checkpoints"/"latest.pt", r.run_dir/"checkpoints"/"fedtros_mc_student_latest.pt"]
        if any(p.exists() for p in ckpts): rows.append(r)
    _print_runs(rows); return 0


def cmd_missing(args: argparse.Namespace) -> int:
    study=load_study_config(args.study,_ROOT)
    plans=expand_study_matrix(study,stage=args.stage,seeds=args.seeds,project_root=_ROOT)
    summary=perform_dry_run(plans,output_base_dir=Path(args.outputs_dir))
    missing=[r for r in summary.planned_runs if r.status!="COMPLETED"]
    print(f"Study {summary.study_id}: expected={summary.total_runs}, completed={summary.completed_runs}, missing={len(missing)}")
    for r in missing: print(f"  {r.status:<12} {r.run_id}")
    return 0


def main() -> int:
    p=argparse.ArgumentParser(description="FedTROS-MC run registry/query CLI")
    p.add_argument("--outputs-dir",default=str(_ROOT/"outputs"))
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("list"); q.add_argument("--study"); q.add_argument("--stage"); q.add_argument("--method"); q.add_argument("--status",default=None)
    for name in ("failed","interrupted","completed","resumable","summary"):
        sp=sub.add_parser(name); sp.add_argument("--study",default=None); sp.add_argument("--stage",default=None); sp.add_argument("--method",default=None)
    show=sub.add_parser("show",aliases=["inspect"]); show.add_argument("target")
    miss=sub.add_parser("missing"); miss.add_argument("study"); miss.add_argument("--stage",default="paper_final"); miss.add_argument("--seeds",type=int,nargs="+")
    args=p.parse_args()
    if args.command=="list": return cmd_list(args)
    if args.command in {"show","inspect"}: return cmd_show(args)
    if args.command=="summary": return cmd_summary(args)
    if args.command=="resumable": return cmd_resumable(args)
    if args.command=="missing": return cmd_missing(args)
    mapping={"failed":{"FAILED","CANCELLED"},"interrupted":{"INTERRUPTED","RESUMED"},"completed":{"COMPLETED"}}
    return cmd_status(args,mapping[args.command])


if __name__=="__main__": raise SystemExit(main())
