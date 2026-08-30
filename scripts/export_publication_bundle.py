#!/usr/bin/env python3
"""Export a versioned, immutable FedTROS-MC -> plots publication bundle.

The two repositories never import each other's Python packages.  This file contract is
the sole integration boundary.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
_ROOT=Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0,str(_ROOT))
import pandas as pd
from src.analysis.aggregation import aggregate_runs, compute_paired_deltas
from src.analysis.export import build_efficiency_curve
from src.analysis.query import query_runs

SCHEMA_NAME="fedtros_mc_publication_bundle"
SCHEMA_VERSION=1
STUDIES=("E1-IID-CS","E2-IID-OSR","E3-NIID-CS","E4-NIID-FOSR","E5-DATASET","E6-SCALE","E7-EFFICIENCY","E8-LOAO","A1-TEACHER","A2-ANCHOR","A3-TRANSFER","A4-PR","A5-FEATURE","S1-SENSITIVITY")

ABLATION_REFERENCES={
    "A1-TEACHER":"vct",
    "A2-ANCHOR":"adaptive_anchor",
    "A3-TRANSFER":"full_transfer",
    "A4-PR":"full_rank",
}
ABLATION_METRICS=(
    "closed_set/macro_f1",
    "open_set/auroc",
    "open_set/unknown_f1",
    "open_set/known_false_unknown_rate",
)

def ablation_delta_rows(runs):
    rows=[]
    for study,reference_variant in ABLATION_REFERENCES.items():
        sr=[r for r in runs if r.study==study]
        if not sr: continue
        # Pair only within identical scientific conditions; variant is the changed factor.
        conditions={}
        for r in sr:
            key=(r.method,r.dataset,r.alpha,r.num_clients,tuple(r.unknown_labels))
            conditions.setdefault(key,[]).append(r)
        for key,group in conditions.items():
            ref=[r for r in group if r.variant==reference_variant]
            if not ref: continue
            variants=sorted({r.variant for r in group if r.variant!=reference_variant})
            for variant in variants:
                candidate=[r for r in group if r.variant==variant]
                deltas=compute_paired_deltas(candidate,ref,ABLATION_METRICS)
                for metric,stat in deltas.items():
                    if stat.n<=0: continue
                    rows.append({
                        "study":study,"method":key[0],"dataset":key[1],"alpha":key[2],
                        "num_clients":key[3],"unknown_labels":"|".join(key[4]),
                        "variant":variant,"reference_variant":reference_variant,"metric":metric,
                        "delta_mean":stat.mean,"delta_sd":stat.std_across_seeds,
                        "delta_ci95_low":stat.ci95_low,"delta_ci95_high":stat.ci95_high,
                        "n_pairs":stat.n,
                    })
    return pd.DataFrame(rows)


def sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def git_commit(root:Path)->str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return "unknown"

def metric_rows(runs):
    rows=[]
    for r in runs:
        base={"run_id":r.run_id,"study":r.study,"stage":r.stage,"method":r.method,"dataset":r.dataset,"alpha":r.alpha,"iid":r.iid,"seed":r.seed,"num_clients":r.num_clients,"unknown_labels":"|".join(r.unknown_labels),"variant":r.variant,"config_hash":r.config_hash,"split_hash":r.split_hash,"git_commit":r.git_commit}
        row=dict(base)
        for k,v in r.metrics.items():
            if isinstance(v,(int,float,str,bool)) or v is None: row[k]=v
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_rows(runs):
    groups={}
    for r in runs: groups.setdefault((r.study,r.method,r.dataset,r.alpha,r.num_clients,r.variant,tuple(r.unknown_labels)),[]).append(r)
    rows=[]
    for key,g in groups.items():
        a=aggregate_runs(g,validate=False)
        base={"study":key[0],"method":key[1],"dataset":key[2],"alpha":key[3],"num_clients":key[4],"variant":key[5],"unknown_labels":"|".join(key[6]),"n_seeds":len(a.seeds),"seeds":"|".join(map(str,a.seeds)),"run_ids":"|".join(a.run_ids)}
        for name,m in a.metrics.items():
            rows.append({**base,"metric":name,"mean":m.mean,"sd":m.std_across_seeds,"ci95_low":m.ci95_low,"ci95_high":m.ci95_high,"n":m.n})
    return pd.DataFrame(rows)

def concat_frame(runs, attr:str)->pd.DataFrame:
    frames=[]
    for r in runs:
        df=getattr(r,attr)
        if df is None or df.empty: continue
        x=df.copy(); x.insert(0,"run_id",r.run_id); x.insert(1,"study",r.study); x.insert(2,"method",r.method); x.insert(3,"dataset",r.dataset); x.insert(4,"alpha",r.alpha); x.insert(5,"seed",r.seed); x.insert(6,"num_clients",r.num_clients); x.insert(7,"variant",r.variant); x.insert(8,"unknown_labels","|".join(r.unknown_labels)); frames.append(x)
    return pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()

def export(outputs:Path,target_root:Path,freeze_id:str|None,include_stages:list[str])->Path:
    all_runs=query_runs(outputs_dir=outputs,status="COMPLETED")
    runs=[r for r in all_runs if r.study in STUDIES and r.stage in set(include_stages)]
    freeze=freeze_id or datetime.now(timezone.utc).strftime("fedtros-pr-vct-%Y%m%dT%H%M%SZ")
    target=target_root/freeze
    if target.exists(): raise FileExistsError(f"Publication bundle already exists: {target}; use a new --freeze-id")
    target.mkdir(parents=True)
    metric_rows(runs).to_csv(target/"runs.csv",index=False)
    aggregate_rows(runs).to_csv(target/"aggregates.csv",index=False)
    delta_frame=ablation_delta_rows(runs)
    if not delta_frame.empty:
        delta_frame.to_csv(target/"paired_deltas.csv",index=False)
    sources={}
    for study in STUDIES:
        sr=[r for r in runs if r.study==study]
        if not sr: continue
        d=target/study; d.mkdir()
        metric_rows(sr).to_csv(d/"summary_runs.csv",index=False)
        aggregate_rows(sr).to_csv(d/"summary.csv",index=False)
        if not delta_frame.empty:
            sd=delta_frame[delta_frame["study"]==study]
            if not sd.empty: sd.to_csv(d/"paired_deltas.csv",index=False)
        for attr,name in (
            ("history","round_curves.csv"),
            ("scores","scores.csv"),
            ("roc_curve","roc.csv"),
            ("pr_curve","pr.csv"),
            ("client_metrics","client_metrics.csv"),
            ("client_distribution","client_distribution.csv"),
            ("client_support","client_support.csv"),
            ("class_metrics","class_metrics.csv"),
            ("communication","communication.csv"),
            ("runtime","runtime.csv"),
        ):
            df=concat_frame(sr,attr)
            if not df.empty: df.to_csv(d/name,index=False)
        if study == "E7-EFFICIENCY":
            efficiency = build_efficiency_curve(sr)
            if not efficiency.empty:
                efficiency.to_csv(d / "efficiency_curve.csv", index=False)
        raw=d/"raw_artifacts"; raw.mkdir()
        for r in sr:
            for label,path in (("confusion_before",r.confusion_before_path),("confusion_after",r.confusion_after_path)):
                if path and path.exists():
                    dest=raw/f"{r.run_id}__{label}{path.suffix}"; shutil.copy2(path,dest)
                    sources[dest.relative_to(target).as_posix()]={"run_id":r.run_id,"source":str(path)}
    prov=target/"provenance"; prov.mkdir()
    (prov/"artifact_sources.json").write_text(json.dumps(sources,indent=2),encoding="utf-8")
    files={}
    for p in sorted(target.rglob("*")):
        if p.is_file() and p.name!="manifest.json": files[p.relative_to(target).as_posix()]={"sha256":sha256(p),"size_bytes":p.stat().st_size}
    manifest={"schema_name":SCHEMA_NAME,"schema_version":SCHEMA_VERSION,"method":"FedTROS-MC","teacher":"VCT","detector":"adaptive_multicenter_lw_split_conformal","experiment_contract_version":"2026-08-19","config_freeze_id":freeze,"created_at":datetime.now(timezone.utc).isoformat(),"code_commit":git_commit(_ROOT),"source_run_ids":[r.run_id for r in runs],"source_config_hashes":sorted({r.config_hash for r in runs if r.config_hash}),"source_split_hashes":sorted({r.split_hash for r in runs if r.split_hash}),"studies_present":sorted({r.study for r in runs}),"include_stages":include_stages,"tabular_format":"csv","files":files}
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    return target

def main():
    p=argparse.ArgumentParser(description="Export versioned publication bundle for the separate plots repository")
    p.add_argument("--outputs-dir",default="outputs"); p.add_argument("--target-root",default="publication_exports"); p.add_argument("--freeze-id",default=None); p.add_argument("--include-stages",nargs="+",default=["paper_final","ablation","reproduction"])
    a=p.parse_args(); target=export(Path(a.outputs_dir),Path(a.target_root),a.freeze_id,a.include_stages); print(target)
if __name__=="__main__": main()
