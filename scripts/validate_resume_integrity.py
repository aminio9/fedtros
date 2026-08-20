#!/usr/bin/env python3
"""Compare an uninterrupted run with an interrupted-and-resumed run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


METRIC_TOLERANCES = {
    "open_set/auroc": 0.03,
    "open_set/auprc": 0.01,
    # FPR95 is quantized on the tiny validation test split, so one threshold
    # crossing moves it by a large fraction despite a small tensor delta.
    "open_set/fpr95": 0.20,
    "open_set/unknown_recall": 0.03,
}


def _checkpoint(run_dir: Path) -> dict:
    return torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)


def _metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics" / "open_set_metrics.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("uninterrupted", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs-tol", type=float, default=5e-3)
    parser.add_argument("--mean-abs-tol", type=float, default=5e-4)
    args = parser.parse_args()

    left = _checkpoint(args.uninterrupted)
    right = _checkpoint(args.resumed)
    left_state = left["student_model"]
    right_state = right["student_model"]
    if left_state.keys() != right_state.keys():
        raise SystemExit("Checkpoint state dictionaries have different keys")
    diffs = torch.cat(
        [(left_state[key].float() - right_state[key].float()).abs().reshape(-1) for key in left_state]
    )
    max_abs = float(diffs.max().item())
    mean_abs = float(diffs.mean().item())
    rmse = float(diffs.square().mean().sqrt().item())

    left_metrics = _metrics(args.uninterrupted)
    right_metrics = _metrics(args.resumed)
    metric_checks = {}
    for name, tolerance in METRIC_TOLERANCES.items():
        a = float(left_metrics[name])
        b = float(right_metrics[name])
        delta = abs(a - b)
        metric_checks[name] = {
            "uninterrupted": a,
            "resumed": b,
            "absolute_delta": delta,
            "tolerance": tolerance,
            "passed": delta <= tolerance,
        }

    passed = (
        max_abs <= args.max_abs_tol
        and mean_abs <= args.mean_abs_tol
        and all(item["passed"] for item in metric_checks.values())
        and int(left.get("round", -1)) == 2
        and int(right.get("round", -1)) == 2
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "comparison": "near_equivalent_with_explicit_tiny_stage_tolerances",
        "uninterrupted_run": str(args.uninterrupted.resolve()),
        "resumed_run": str(args.resumed.resolve()),
        "rounds": {"uninterrupted": int(left.get("round", -1)), "resumed": int(right.get("round", -1))},
        "checkpoint": {
            "num_parameters": int(diffs.numel()),
            "max_absolute_delta": max_abs,
            "mean_absolute_delta": mean_abs,
            "rmse": rmse,
            "max_absolute_tolerance": args.max_abs_tol,
            "mean_absolute_tolerance": args.mean_abs_tol,
        },
        "metrics": metric_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
