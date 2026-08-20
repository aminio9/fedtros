#!/usr/bin/env python3
"""Audit tiny run contracts and W&B offline mirrors without reading metrics from logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any


STUDIES = (
    "E0-VERIFY", "E1-IID-CS", "E2-IID-OSR", "E3-NIID-CS", "E4-NIID-FOSR",
    "E5-DATASET", "E6-SCALE", "E7-EFFICIENCY", "E8-LOAO", "A1-TEACHER",
    "A2-ANCHOR", "A3-TRANSFER", "A4-PR", "A5-FEATURE", "S1-SENSITIVITY",
)
LEGACY_TERMS = ("DKD-FedOS", "Fed-DiGOS", "FedPROTEUS", "CVAE-DQN", "CVQN", "DQN teacher", "RL teacher", "PNPFF")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item_key(item: Any) -> str:
    if getattr(item, "key", ""):
        return str(item.key)
    nested = list(getattr(item, "nested_key", []))
    return ".".join(str(value) for value in nested)


def item_value(item: Any) -> Any:
    raw = str(getattr(item, "value_json", "null"))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def read_wandb(path: Path) -> dict[str, Any]:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(path))
    summary: dict[str, Any] = {}
    history: dict[str, list[Any]] = defaultdict(list)
    identity: dict[str, Any] = {}
    exit_code: int | None = None
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        kind = record.WhichOneof("record_type")
        if kind == "history":
            for item in record.history.item:
                history[item_key(item)].append(item_value(item))
        elif kind == "summary":
            for item in record.summary.update:
                summary[item_key(item)] = item_value(item)
            for item in record.summary.remove:
                summary.pop(item_key(item), None)
        elif kind == "run":
            identity.update({
                "run_id": record.run.run_id,
                "project": record.run.project,
                "display_name": record.run.display_name,
                "group": record.run.run_group,
                "job_type": record.run.job_type,
            })
        elif kind == "exit":
            exit_code = int(record.exit.exit_code)
    return {"summary": summary, "history": dict(history), "identity": identity, "exit_code": exit_code}


def find_checkpoint(run_dir: Path) -> Path | None:
    for relative in ("checkpoints/latest.pt", "checkpoints/fedtros_pr_student_latest.pt", "checkpoints/global_model_latest.pt"):
        candidate = run_dir / relative
        if candidate.exists():
            return candidate
    return None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    outputs = args.outputs_dir.resolve()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    environment = {
        "python": sys.version,
        "os": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": {name: package_version(name) for name in ("torch", "flwr", "hydra-core", "wandb", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy")},
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_value(root, "status", "--porcelain")),
    }
    try:
        import torch
        environment["cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda, "device_count": torch.cuda.device_count(), "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    except Exception as exc:
        environment["cuda"] = {"error": str(exc)}
    (target / "environment_report.json").write_text(json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8")

    completed: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in sorted((outputs / "runs").glob("*")):
        manifest_path = run_dir / "run_manifest.json"
        result_path = run_dir / "result_manifest.json"
        if not manifest_path.exists() or not result_path.exists():
            continue
        manifest = read_json(manifest_path)
        result = read_json(result_path)
        if str(manifest.get("status", "")).upper() == "COMPLETED" and str(result.get("status", "")).upper() == "COMPLETED":
            completed.append((run_dir, manifest))

    artifact_rows: list[dict[str, Any]] = []
    manifest_failures: list[str] = []
    wandb_rows: list[dict[str, Any]] = []
    wandb_audit: dict[str, Any] = {}
    log_lines = ["# Tiny-validation log consistency", "", "Logs were checked only for lifecycle/diagnostic text; scientific values came from structured files.", ""]
    study_runs: dict[str, list[str]] = defaultdict(list)
    figure_violations: list[str] = []
    legacy_log_hits: list[str] = []

    for run_dir, manifest in completed:
        run_id = str(manifest.get("run_id", run_dir.name))
        study = str(manifest.get("study_id", ""))
        study_runs[study].append(run_id)
        is_open = bool(manifest.get("unknown_labels"))
        checks: list[tuple[str, bool | None, str]] = [
            ("resolved_config.yaml", (run_dir / "resolved_config.yaml").exists(), "common"),
            ("run_manifest.json", (run_dir / "run_manifest.json").exists(), "common"),
            ("data_manifest.json", (run_dir / "data_manifest.json").exists(), "common"),
            ("partition_manifest.json", (run_dir / "partition_manifest.json").exists(), "common"),
            ("model_manifest.json", (run_dir / "model_manifest.json").exists(), "common"),
            ("feature_manifest.json", (run_dir / "feature_manifest.json").exists(), "common"),
            ("seed_manifest.json", (run_dir / "seed_manifest.json").exists(), "common"),
            ("metrics/final_metrics.json", (run_dir / "metrics/final_metrics.json").exists(), "common"),
            ("metrics/round_metrics.csv", (run_dir / "metrics/round_metrics.csv").exists(), "common"),
            ("checkpoint", find_checkpoint(run_dir) is not None, "common"),
            ("logs/run.log", (run_dir / "logs/run.log").exists(), "common"),
            ("result_manifest.json", (run_dir / "result_manifest.json").exists(), "common"),
            ("predictions/open_set_scores.csv", (run_dir / "predictions/open_set_scores.csv").exists() if is_open else None, "open_set"),
            ("artifacts/prototype_bank", bool(list((run_dir / "artifacts").glob("*prototype_bank*"))) if is_open else None, "open_set"),
            ("artifacts/rank_calibration", bool(list((run_dir / "artifacts").glob("*rank_calibration*"))) if is_open else None, "open_set"),
            ("artifacts/roc_data", bool(list((run_dir / "artifacts").glob("*roc*"))) if is_open else None, "open_set"),
            ("artifacts/pr_data", bool(list((run_dir / "artifacts").glob("*pr_*"))) if is_open else None, "open_set"),
            ("artifacts/confusion_open", bool(list((run_dir / "artifacts").glob("*confusion_open*"))) if is_open else None, "open_set"),
        ]
        for artifact, passed, applicability in checks:
            artifact_rows.append({"study": study, "run_id": run_id, "artifact": artifact, "applicability": applicability, "status": "NA" if passed is None else ("PASS" if passed else "FAIL")})

        required_manifest = {
            "run_id": run_id, "study_id": study, "stage": "smoke", "status": "COMPLETED",
            "config_hash": manifest.get("config_hash"), "split_hash": manifest.get("split_hash"),
            "git_commit": manifest.get("git_commit"), "method_id": manifest.get("method_id"),
            "teacher_type": manifest.get("teacher_type"), "dataset": manifest.get("dataset"),
            "seed": manifest.get("seed"), "num_clients": manifest.get("num_clients"),
            "known_labels": manifest.get("known_labels"),
        }
        for key, value in required_manifest.items():
            if value in (None, "", [], {}) or (key == "stage" and str(manifest.get(key)) != "smoke") or (key == "status" and str(manifest.get(key)) != "COMPLETED"):
                manifest_failures.append(f"{run_id}:{key}")

        inventory = manifest.get("artifact_inventory", {})
        for relative, expected in inventory.items():
            if relative.startswith("logs/") or Path(relative).name in {"run_manifest.json", "result_manifest.json"}:
                continue
            path = run_dir / relative
            if not path.exists() or path.stat().st_size != int(expected.get("size_bytes", -1)) or sha256(path) != expected.get("sha256"):
                manifest_failures.append(f"{run_id}:inventory:{relative}")

        for extension in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf"):
            figure_violations.extend(str(path) for path in run_dir.rglob(extension))

        log_path = run_dir / "logs" / "run.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        events = {term: (term.lower() in log_text.lower()) for term in ("run starting", "Preprocessing complete", "ROUND 1", "checkpoint", "evaluation", "Experiment completed")}
        log_lines.append(f"- `{run_id}`: " + ", ".join(f"{key}={'PASS' if value else 'MISS'}" for key, value in events.items()))
        for term in LEGACY_TERMS:
            if term.lower() in log_text.lower():
                legacy_log_hits.append(f"{run_id}:{term}")

        local_metrics_path = run_dir / "metrics" / "final_metrics.json"
        local_metrics = read_json(local_metrics_path) if local_metrics_path.exists() else {}
        wandb_files = sorted(run_dir.rglob("*.wandb"), key=lambda path: path.stat().st_mtime)
        tracker = str(read_json(run_dir / "result_manifest.json").get("tracker", ""))
        if tracker == "wandb" and not wandb_files:
            manifest_failures.append(f"{run_id}:missing_wandb")
            continue
        if not wandb_files:
            wandb_audit[run_id] = {"tracker": tracker, "status": "NA_DISABLED"}
            continue
        parsed = read_wandb(wandb_files[-1])
        wandb_audit[run_id] = {
            "file": str(wandb_files[-1]), "identity": parsed["identity"], "exit_code": parsed["exit_code"],
            "history_keys": sorted(parsed["history"]), "summary_keys": sorted(parsed["summary"]),
        }
        if parsed["identity"].get("run_id") != run_id or parsed["identity"].get("project") != "FedTROS-PR" or parsed["exit_code"] != 0:
            manifest_failures.append(f"{run_id}:wandb_identity_or_exit")
        shared = 0
        for metric, local_value in sorted(local_metrics.items()):
            if isinstance(local_value, bool) or not isinstance(local_value, (int, float)) or metric not in parsed["summary"]:
                continue
            wandb_value = parsed["summary"][metric]
            if isinstance(wandb_value, bool) or not isinstance(wandb_value, (int, float)):
                continue
            difference = abs(float(local_value) - float(wandb_value))
            passed = math.isclose(float(local_value), float(wandb_value), rel_tol=args.tolerance, abs_tol=args.tolerance)
            wandb_rows.append({"run_id": run_id, "metric": metric, "local_value": local_value, "wandb_value": wandb_value, "difference": difference, "status": "PASS" if passed else "FAIL"})
            shared += 1
        if shared == 0:
            manifest_failures.append(f"{run_id}:no_shared_wandb_metrics")

    write_csv(target / "artifact_matrix.csv", artifact_rows, ["study", "run_id", "artifact", "applicability", "status"])
    write_csv(target / "wandb_consistency.csv", wandb_rows, ["run_id", "metric", "local_value", "wandb_value", "difference", "status"])
    (target / "wandb_audit.json").write_text(json.dumps(wandb_audit, indent=2, sort_keys=True), encoding="utf-8")
    log_lines.extend(["", f"Legacy active-name hits: {len(legacy_log_hits)}", f"Scientific values parsed from logs: 0"])
    (target / "log_consistency_report.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    failed_artifacts = sum(row["status"] == "FAIL" for row in artifact_rows)
    failed_wandb = sum(row["status"] == "FAIL" for row in wandb_rows)
    summary = {
        "completed_run_count": len(completed),
        "studies": {study: sorted(study_runs.get(study, [])) for study in STUDIES},
        "missing_studies": [study for study in STUDIES if not study_runs.get(study)],
        "artifact_failures": failed_artifacts,
        "manifest_failures": sorted(set(manifest_failures)),
        "wandb_comparisons": len(wandb_rows),
        "wandb_consistency_failures": failed_wandb,
        "forbidden_run_figures": figure_violations,
        "legacy_log_hits": legacy_log_hits,
    }
    summary["status"] = "PASS" if not (summary["missing_studies"] or failed_artifacts or manifest_failures or failed_wandb or figure_violations or legacy_log_hits) else "FAIL"
    (target / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
