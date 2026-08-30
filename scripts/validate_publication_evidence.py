#!/usr/bin/env python3
"""Fail-fast gate for publication-ready FedTROS-MC evidence.

This validator intentionally treats smoke/development runs and historical PR
artifacts as non-publication evidence. It reports missing canonical cells instead
of silently aggregating whatever happens to be present.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

SEEDS = {17, 42, 73, 101, 137}
STUDIES = {"E1-IID-CS", "E2-IID-OSR", "E3-NIID-CS", "E4-NIID-FOSR", "E5-DATASET", "E6-SCALE", "E7-EFFICIENCY", "E8-LOAO", "A1-TEACHER", "A2-ANCHOR", "A3-TRANSFER", "A4-PR", "A5-FEATURE", "S1-SENSITIVITY"}

def read_manifest(run: Path) -> dict:
    for path in (run / "metadata" / "run_manifest.json", run / "run_manifest.json"):
        if path.exists():
            try: return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError: return {"status": "CORRUPTED"}
    return {}

def validate_split_provenance(run: Path, study: str) -> list[str]:
    """Validate final-test identifier traceability for open-set evidence."""
    if study not in {"E2-IID-OSR", "E4-NIID-FOSR", "E5-DATASET", "E7-EFFICIENCY", "E8-LOAO",
                     "A1-TEACHER", "A2-ANCHOR", "A3-TRANSFER", "A4-PR", "A5-FEATURE", "S1-SENSITIVITY"}:
        return []
    path = run / "split_manifest.json"
    if not path.exists():
        return [f"{study}/{run.name}: missing split_manifest.json"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [f"{study}/{run.name}: corrupted split_manifest.json"]
    entry = manifest.get("final_test_identifiers") or {}
    rel = entry.get("path")
    expected_hash = str(entry.get("hash") or "")
    if not rel or not expected_hash:
        return [f"{study}/{run.name}: final-test identifier path/hash missing"]
    identifier_path = run / str(rel)
    if not identifier_path.exists():
        return [f"{study}/{run.name}: final-test identifier artifact missing: {rel}"]
    actual_hash = hashlib.sha256(identifier_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        return [f"{study}/{run.name}: final-test identifier checksum mismatch"]
    return []

def validate(runs_root: Path) -> dict:
    groups = defaultdict(list)
    ignored = []
    for run in sorted(runs_root.iterdir() if runs_root.exists() else []):
        if not run.is_dir(): continue
        m = read_manifest(run)
        if str(m.get("status", "")).upper() != "COMPLETED":
            ignored.append(run.name); continue
        study = str(m.get("study_id", "")).upper()
        if study not in STUDIES:
            ignored.append(run.name); continue
        groups[study].append({"run_id": run.name, "seed": m.get("seed"), "rounds": m.get("num_rounds"), "clients": m.get("num_clients"), "method": m.get("method"), "stage": m.get("stage"), "open_set_method": m.get("open_set_method")})
        # Provenance is checked per completed candidate, before aggregation.
        groups[study][-1]["provenance_errors"] = validate_split_provenance(run, study)
    errors = []
    for study in sorted(STUDIES):
        rows = groups.get(study, [])
        if not rows:
            errors.append(f"{study}: no completed canonical runs")
            continue
        seeds = {int(r["seed"]) for r in rows if r["seed"] is not None}
        if study != "E0-VERIFY" and seeds != SEEDS:
            errors.append(f"{study}: seeds={sorted(seeds)}; required={sorted(SEEDS)}")
        if any(int(r["rounds"] or 0) != 100 for r in rows):
            errors.append(f"{study}: at least one run is not 100 rounds")
        if study != "E6-SCALE" and any(int(r["clients"] or 0) != 10 for r in rows):
            errors.append(f"{study}: at least one run is not 10 clients")
        if any("PR" in str(r["method"]).upper() and "MC" not in str(r["method"]).upper() for r in rows):
            errors.append(f"{study}: legacy FedTROS-PR method label present")
        for row in rows:
            errors.extend(row.get("provenance_errors", []))
        open_set_studies = {"E2-IID-OSR", "E4-NIID-FOSR", "E5-DATASET", "E6-SCALE", "E7-EFFICIENCY", "E8-LOAO", "A1-TEACHER", "A2-ANCHOR", "A3-TRANSFER", "A4-PR", "A5-FEATURE", "S1-SENSITIVITY"}
        if study in open_set_studies:
            allowed_detectors = {"multicenter_conformal", "prototype_rank"} if study == "A4-PR" else {"multicenter_conformal"}
            if any(str(r["open_set_method"]).lower() not in allowed_detectors for r in rows):
                errors.append(f"{study}: detector is outside the declared canonical ablation set")
    return {"publication_ready": not errors, "errors": errors, "completed_rows": sum(map(len, groups.values())), "ignored_runs": ignored, "studies_present": sorted(groups), "required_seeds": sorted(SEEDS)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="outputs/runs", type=Path)
    ap.add_argument("--report", default=None, type=Path)
    args = ap.parse_args()
    report = validate(args.runs_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if report["publication_ready"] else 2

if __name__ == "__main__": raise SystemExit(main())
