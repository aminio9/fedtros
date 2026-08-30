#!/usr/bin/env python3
"""Build non-visual Q1 statistics/tables from completed FedTROS-MC runs.

Publication figures are rendered exclusively by the separate plots repository.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
_ROOT=Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0,str(_ROOT))
import pandas as pd
from src.analysis.aggregation import aggregate_runs
from src.analysis.export import generate_provenance_manifest
from src.analysis.query import query_runs
from src.analysis.statistics import compare_paired_significance
from src.analysis.tables import export_all_paper_tables


def build(outputs: Path, target: Path, stage: str|list[str]|tuple[str, ...]|None="main", study: str|None=None) -> dict[str,Any]:
    target.mkdir(parents=True,exist_ok=True)
    tables=target/"tables"; stats=target/"statistics"; agg_dir=target/"aggregates"; prov=target/"provenance"
    for d in (tables,stats,agg_dir,prov): d.mkdir(parents=True,exist_ok=True)
    runs=query_runs(outputs_dir=outputs,stage=stage,study=study,status="COMPLETED")
    export_all_paper_tables(runs,tables)
    rows=[]
    groups={}
    for r in runs: groups.setdefault((r.study,r.method,r.dataset,r.alpha,r.num_clients,r.variant,tuple(r.unknown_labels)),[]).append(r)
    for key, group in groups.items():
        a=aggregate_runs(group,validate=False)
        base={"study":key[0],"method":key[1],"dataset":key[2],"alpha":key[3],"num_clients":key[4],"variant":key[5],"unknown_labels":"|".join(key[6]),"n_seeds":len(a.seeds),"run_ids":"|".join(a.run_ids)}
        for name,m in a.metrics.items():
            rows.append({**base,"metric":name,"mean":m.mean,"sd":m.std_across_seeds,"ci95_low":m.ci95_low,"ci95_high":m.ci95_high,"n":m.n})
    pd.DataFrame(rows).to_csv(agg_dir/"multi_seed_metrics.csv",index=False)

    comparisons=[]
    for st in sorted({r.study for r in runs}):
        cand=[r for r in runs if r.study==st and r.method=="FedTROS-MC"]
        for baseline in ("FedAvg-Student","FedProx-Student"):
            base=[r for r in runs if r.study==st and r.method==baseline]
            if not cand or not base: continue
            for metric in ("closed_set/macro_f1","open_set/auroc","open_set/unknown_f1","open_set/KFR"):
                rep=compare_paired_significance(cand,base,metric)
                if rep: comparisons.append(rep.__dict__|{"study":st,"candidate":"FedTROS-MC","baseline":baseline})
    pd.DataFrame(comparisons).to_csv(stats/"paired_comparisons.csv",index=False)
    generate_provenance_manifest(runs,prov/"provenance_manifest.json",extra_metadata={"analysis":"build_q1_results","stage":stage})
    summary={"status":"SUCCESS" if runs else "EMPTY","runs":len(runs),"conditions":len(groups),"target":str(target)}
    (target/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--outputs-dir",default="outputs"); p.add_argument("--target",default="paper_results"); p.add_argument("--stage",nargs="+",default=["main"],help="One or more run stages, e.g. main ablation reproduction"); p.add_argument("--study",default=None)
    a=p.parse_args(); print(json.dumps(build(Path(a.outputs_dir),Path(a.target),a.stage,a.study),indent=2))
if __name__=="__main__": main()
