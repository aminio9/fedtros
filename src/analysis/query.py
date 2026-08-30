"""Metadata-driven query API for canonical FedTROS-PR run directories."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable, Sequence
from src.analysis.loaders import RunRecord, load_run
logger=logging.getLogger(__name__)

def _norm(v: object) -> str: return str(v).lower().replace("-","").replace("_","").replace(" ","")
def _match(value: object, query: object|Sequence[object]|None) -> bool:
    if query is None: return True
    vals=list(query) if isinstance(query,(list,tuple,set)) else [query]
    nv=_norm(value); return any(nv==_norm(q) or _norm(q) in nv for q in vals)

def _roots(outputs_dir: str|Path|Sequence[str|Path]) -> list[Path]:
    roots=[Path(x) for x in outputs_dir] if isinstance(outputs_dir,(list,tuple)) else [Path(outputs_dir)]
    candidates=[]
    for root in roots:
        if not root.exists(): continue
        search=root/"runs" if (root/"runs").is_dir() else root
        if (search/"metadata"/"run_manifest.json").exists(): candidates.append(search); continue
        for child in sorted(search.iterdir()):
            if child.is_dir() and any((child/x).exists() for x in (Path("metadata/run_manifest.json"),Path("run_manifest.json"),Path("resolved_config.yaml"),Path("metadata.json"))): candidates.append(child)
    return candidates

def query_runs(study=None, stage=None, method=None, dataset=None, alpha=None, seed=None, num_clients=None,
               status: str|None="COMPLETED", outputs_dir: str|Path|Sequence[str|Path]="outputs",
               predicate: Callable[[RunRecord],bool]|None=None,
               include_invalid: bool = False) -> list[RunRecord]:
    out=[]
    for rdir in _roots(outputs_dir):
        try: r=load_run(rdir)
        except Exception as exc: logger.debug("Skip %s: %s",rdir,exc); continue
        if status is not None and r.status.upper()!=status.upper(): continue
        if not include_invalid and r.validity_status != "VALID": continue
        if not _match(r.study,study) or not _match(r.stage,stage) or not _match(r.method,method) or not _match(r.dataset,dataset): continue
        if alpha is not None:
            vals=list(alpha) if isinstance(alpha,(list,tuple,set)) else [alpha]
            if not any(abs(r.alpha-float(v))<1e-9 for v in vals): continue
        if seed is not None:
            vals=list(seed) if isinstance(seed,(list,tuple,set)) else [seed]
            if r.seed not in [int(v) for v in vals]: continue
        if num_clients is not None:
            vals=list(num_clients) if isinstance(num_clients,(list,tuple,set)) else [num_clients]
            if r.num_clients not in [int(v) for v in vals]: continue
        if predicate and not predicate(r): continue
        out.append(r)
    return sorted(out,key=lambda r:(r.study,r.method,r.dataset,r.alpha,r.num_clients,r.seed,r.run_id))
