#!/usr/bin/env python3
"""Validate a completed FedTROS checkpoint and manually recompute E7 payloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    target = args.target_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)

    manifest = _json(run / "run_manifest.json")
    checkpoint_path = run / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("student_model") or checkpoint.get("student_state_dict")
    if not isinstance(state, dict):
        raise TypeError("Canonical latest checkpoint has no student state dictionary")
    checkpoint_checks = {
        "latest_exists": checkpoint_path.exists(),
        "best_applicability": "NA_NO_BEST_SELECTION_CONTRACT",
        "schema_version_2": checkpoint.get("schema_version") == 2,
        "method": checkpoint.get("method") == "FedTROS-PR",
        "method_id": checkpoint.get("method_id") == "fedtros_pr",
        "teacher_type": checkpoint.get("teacher_type") == "variational_classifier",
        "config_hash_matches_manifest": checkpoint.get("config_hash") == manifest.get("config_hash"),
        "run_id_matches_manifest": checkpoint.get("run_id") == manifest.get("run_id"),
        "round_matches_manifest": int(checkpoint.get("round", -1)) == int(manifest.get("num_rounds", -2)),
        "student_state_present": bool(state),
    }
    checkpoint_report = {
        "run_id": run.name,
        "checkpoint": str(checkpoint_path),
        "fields": sorted(checkpoint),
        "checks": checkpoint_checks,
        "status": "PASS" if all(value is True or str(value).startswith("NA_") for value in checkpoint_checks.values()) else "FAIL",
    }
    (target / "checkpoint_schema_validation.json").write_text(
        json.dumps(checkpoint_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    parameter_count = sum(int(tensor.numel()) for tensor in state.values())
    parameter_bytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in state.values())
    with (run / "metrics" / "communication_round.csv").open(newline="", encoding="utf-8") as handle:
        communication = list(csv.DictReader(handle))
    clients = int(manifest["num_clients"])
    expected_downlink = parameter_bytes * clients
    expected_uplink = parameter_bytes * clients
    expected_round = expected_downlink + expected_uplink
    expected_cumulative = expected_round * len(communication)
    row_checks = []
    for row in communication:
        row_checks.append({
            "round": int(row["round"]),
            "downlink_matches": math.isclose(float(row["communication/downlink_bytes"]), expected_downlink),
            "uplink_matches": math.isclose(float(row["communication/uplink_bytes"]), expected_uplink),
            "round_matches": math.isclose(float(row["communication/round_bytes"]), expected_round),
        })
    logged_cumulative = sum(float(row["communication/round_bytes"]) for row in communication)
    payload_report = {
        "run_id": run.name,
        "student_parameters": parameter_count,
        "model_payload_bytes": parameter_bytes,
        "dtype_assumption": "actual tensor element_size summed from checkpoint",
        "num_clients": clients,
        "rounds": len(communication),
        "expected_downlink_bytes_per_round": expected_downlink,
        "expected_uplink_bytes_per_round": expected_uplink,
        "expected_total_bytes_per_round": expected_round,
        "expected_cumulative_bytes": expected_cumulative,
        "logged_cumulative_bytes": logged_cumulative,
        "round_checks": row_checks,
    }
    payload_report["status"] = "PASS" if (
        all(all(value for key, value in row.items() if key != "round") for row in row_checks)
        and math.isclose(logged_cumulative, expected_cumulative)
    ) else "FAIL"
    (target / "e7_payload_validation.json").write_text(
        json.dumps(payload_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    overall = checkpoint_report["status"] == payload_report["status"] == "PASS"
    print(json.dumps({"checkpoint": checkpoint_report["status"], "payload": payload_report["status"], "status": "PASS" if overall else "FAIL"}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
