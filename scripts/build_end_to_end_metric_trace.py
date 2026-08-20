#!/usr/bin/env python3
"""Build a compact local -> W&B -> aggregate -> bundle -> plot-input audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_tiny_validation import read_wandb


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(outputs: Path, prefix: str) -> Path:
    matches = sorted((outputs / "runs").glob(f"{prefix}*"))
    matches = [p for p in matches if (p / "result_manifest.json").exists()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run for {prefix!r}, found {len(matches)}")
    return matches[0]


def _wandb(run: Path) -> dict[str, Any]:
    records = sorted(run.rglob("*.wandb"), key=lambda path: path.stat().st_mtime_ns)
    if not records:
        raise FileNotFoundError(f"No offline W&B record below {run}")
    return read_wandb(records[-1])


def _aggregate(rows: list[dict[str, str]], run: Path, metric: str) -> float:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    keys = {
        "study": str(manifest["study_id"]),
        "method": str(manifest["method_id"]),
        "dataset": str(manifest["dataset"]),
        "alpha": str(manifest["alpha"]),
        "num_clients": str(manifest["num_clients"]),
        "variant": str(manifest.get("variant", "canonical")),
        "unknown_labels": ",".join(manifest.get("unknown_labels", [])),
        "metric": metric,
    }
    aliases = {
        "fedtros_pr": "FedTROS-PR", "fedavg": "FedAvg-Student",
        "fedprox": "FedProx-Student", "FedTROS-PR": "FedTROS-PR",
        "FedAvg": "FedAvg-Student", "FedProx": "FedProx-Student",
    }
    keys["method"] = aliases.get(keys["method"], keys["method"])
    for row in rows:
        if all(str(row.get(key, "")) == value for key, value in keys.items()):
            return float(row["mean"])
    raise KeyError(f"Aggregate not found: {keys}")


def _bundle_summary(bundle: Path, study: str, run: Path, metric: str) -> float:
    rows = _read_csv(bundle / study / "summary.csv")
    return _aggregate(rows, run, metric)


def _close(a: float, b: float, tolerance: float) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    outputs = args.outputs_dir.resolve()
    bundle = args.bundle.resolve()
    aggregate_rows = _read_csv(args.aggregate_csv.resolve())
    rows: list[dict[str, Any]] = []

    scalar_specs = [
        ("E3-NIID-CS", "closed_set/macro_f1", "e3niidcs_bnat_fedtros_pr_a0p5_closed_c2_s42_"),
        ("E4-NIID-FOSR", "open_set/auroc", "e4niidfosr_bnat_fedtros_pr_a0p5_fotunk_c2_s42_"),
        ("E4-NIID-FOSR", "open_set/unknown_f1", "e4niidfosr_bnat_fedtros_pr_a0p5_fotunk_c2_s42_"),
        ("E8-LOAO", "open_set/unknown_f1", "e8loao_bnat_fedtros_pr_a0p5_fotunk_c2_s42_"),
    ]
    for study, metric, prefix in scalar_specs:
        run = _run(outputs, prefix)
        local = float(json.loads((run / "metrics" / "final_metrics.json").read_text(encoding="utf-8"))[metric])
        wandb = float(_wandb(run)["summary"][metric])
        aggregate = _aggregate(aggregate_rows, run, metric)
        bundle_value = _bundle_summary(bundle, study, run, metric)
        passed = _close(local, wandb, args.tolerance) and _close(aggregate, bundle_value, args.tolerance)
        rows.append({"study": study, "metric": metric, "run_id": run.name, "local": local,
                     "wandb": wandb, "aggregate": aggregate, "bundle": bundle_value,
                     "plot_input": bundle_value, "status": "PASS" if passed else "FAIL"})

    # E6 figure 08 renders the median of measured per-round runtime.
    run = _run(outputs, "e6scale_bnat_fedtros_pr_a0p5_fotunk_c2_s42_")
    local_rounds = [float(row["runtime/round_seconds"]) for row in _read_csv(run / "metrics" / "timing_round.csv")]
    wandb_rounds = [float(value) for value in _wandb(run)["history"]["runtime/round_seconds"]]
    bundle_rounds = [float(row["runtime/round_seconds"]) for row in _read_csv(bundle / "E6-SCALE" / "runtime.csv") if row["run_id"] == run.name]
    local = statistics.median(local_rounds)
    wandb = statistics.median(wandb_rounds)
    bundle_value = statistics.median(bundle_rounds)
    passed = _close(local, wandb, args.tolerance) and _close(local, bundle_value, args.tolerance)
    rows.append({"study": "E6-SCALE", "metric": "median(runtime/round_seconds)", "run_id": run.name,
                 "local": local, "wandb": wandb, "aggregate": local, "bundle": bundle_value,
                 "plot_input": bundle_value, "status": "PASS" if passed else "FAIL"})

    # E7 cumulative payload is a scientific join over the structured round series.
    run = _run(outputs, "e7efficiency_bnat_fedtros_pr_a0p5_fotunk_c2_s42_")
    local = sum(float(row["communication/round_bytes"]) for row in _read_csv(run / "metrics" / "communication_round.csv"))
    wandb = sum(float(value) for value in _wandb(run)["history"]["communication/round_bytes"])
    communication = [row for row in _read_csv(bundle / "E7-EFFICIENCY" / "communication.csv") if row["run_id"] == run.name]
    curve = [row for row in _read_csv(bundle / "E7-EFFICIENCY" / "efficiency_curve.csv") if row["run_id"] == run.name]
    bundle_value = max(float(row["communication/cumulative_bytes"]) for row in communication)
    plot_value = max(float(row["communication/cumulative_bytes"]) for row in curve)
    passed = all(_close(local, value, args.tolerance) for value in (wandb, bundle_value, plot_value))
    rows.append({"study": "E7-EFFICIENCY", "metric": "communication/cumulative_bytes", "run_id": run.name,
                 "local": local, "wandb": wandb, "aggregate": local, "bundle": bundle_value,
                 "plot_input": plot_value, "status": "PASS" if passed else "FAIL"})

    args.target.parent.mkdir(parents=True, exist_ok=True)
    with args.target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["study", "metric", "run_id", "local", "wandb", "aggregate", "bundle", "plot_input", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"target": str(args.target.resolve()), "rows": len(rows),
                      "failures": sum(row["status"] != "PASS" for row in rows),
                      "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"}, indent=2))
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
