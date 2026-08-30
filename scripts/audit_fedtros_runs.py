#!/usr/bin/env python3
"""Mark historical runs that used an incorrect/unsupported strategy.

The audit is deliberately non-destructive: it only writes metadata/run_validity.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _manifest(run_dir: Path) -> dict:
    for path in (run_dir / "metadata" / "run_manifest.json", run_dir / "run_manifest.json"):
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def _log_text(run_dir: Path) -> str:
    chunks: list[str] = []
    candidates = list((run_dir / "logs").glob("**/*")) if (run_dir / "logs").exists() else []
    candidates.extend(path for path in run_dir.glob("*") if path.is_file())
    for path in sorted(set(candidates)):
        if path.is_file() and path.suffix.lower() in {".log", ".txt", ".out"}:
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(chunks).lower()


def detect_reasons(run_dir: Path) -> list[str]:
    manifest = _manifest(run_dir)
    method = str(manifest.get("method", "")).lower()
    method_id = str(manifest.get("method_id", "")).lower()
    reasons: list[str] = []
    log_text = _log_text(run_dir)
    if "fedgpa" in method or "fedgpa" in method_id:
        reasons.append("unsupported_fedgpa_strategy")
    if ("fedtros-mc" in method or method_id in {"fedtros", "fedtros_mc"}) and (
        "fedavg with model saving" in log_text
        or "fedavg-student" in log_text
        or "standard baseline local training" in log_text
    ):
        reasons.append("fedtros_mc_dispatched_to_fedavg")
    return sorted(set(reasons))


def audit_run(run_dir: Path) -> list[str]:
    reasons = detect_reasons(run_dir)
    if not reasons:
        return []
    payload = {
        "schema_version": 1,
        "status": "INVALID",
        "reasons": sorted(set(reasons)),
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    target = run_dir / "metadata" / "run_validity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload["reasons"]


def iter_runs(root: Path):
    search = root / "runs" if (root / "runs").is_dir() else root
    if (search / "metadata" / "run_manifest.json").exists() or (search / "run_manifest.json").exists():
        yield search
        return
    if not search.exists():
        return
    for child in sorted(search.iterdir()):
        if child.is_dir() and (_manifest(child)):
            yield child


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="outputs directory or directory containing runs/")
    parser.add_argument("--dry-run", action="store_true", help="report invalid runs without writing metadata")
    args = parser.parse_args()
    count = 0
    for run_dir in iter_runs(args.root.resolve()):
        reasons = detect_reasons(run_dir)
        if not args.dry_run and reasons:
            audit_run(run_dir)
        if reasons:
            count += 1
            print(f"{run_dir}: {', '.join(reasons)}")
    print(f"Audited {count} invalid run(s)." if count else "No invalid runs found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
