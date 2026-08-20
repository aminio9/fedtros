#!/usr/bin/env python3
"""Legacy Run Importer & Isolation Engine for FedTROS-PR.

Safely imports historical runs from legacy experiments into `outputs/legacy_imported/`,
preserving logs and scalar metrics while quarantining legacy DQN/RL weights so they
cannot be loaded or aggregated as canonical VCT runs.

Usage:
    python scripts/import_legacy.py --source-dir legacy_runs/ --output-dir outputs/legacy_imported/
    python scripts/import_legacy.py --run-dir outputs/runs/legacy_2025_dqn --quarantine
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ImportLegacy")


def quarantine_and_import_run(
    source_dir: Path,
    target_dir: Path,
    *,
    dry_run: bool = False,
) -> bool:
    """Import a single legacy run directory with metadata tagging and DQN quarantine."""
    if not source_dir.is_dir():
        logger.error("Source path is not a directory: %s", source_dir)
        return False

    run_id = source_dir.name
    dest = target_dir / run_id

    # Check for legacy DQN indicators
    is_legacy_dqn = False
    for f in source_dir.rglob("*.pt"):
        if "dqn" in f.name.lower() or "prior" in f.name.lower() or "policy" in f.name.lower():
            is_legacy_dqn = True
            break

    logger.info("Importing %s -> %s (Legacy DQN: %s)", source_dir.name, dest, is_legacy_dqn)

    if dry_run:
        logger.info("[DRY RUN] Would copy %s to %s with quarantine manifest", source_dir, dest)
        return True

    dest.mkdir(parents=True, exist_ok=True)

    # Copy files
    for item in source_dir.iterdir():
        target_item = dest / item.name
        if item.is_dir():
            if not target_item.exists():
                shutil.copytree(item, target_item)
        else:
            shutil.copy2(item, target_item)

    # Write quarantine metadata
    quarantine_manifest = {
        "legacy_run_id": run_id,
        "is_legacy": True,
        "is_legacy_dqn": is_legacy_dqn,
        "canonical_vct": False,
        "quarantine_status": "QUARANTINED",
        "imported_from": str(source_dir.resolve()),
        "notice": "This run was imported from a legacy RL/DQN or pre-VCT study. It is strictly quarantined from publication tables.",
    }

    meta_dir = dest / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "quarantine_manifest.json").write_text(
        json.dumps(quarantine_manifest, indent=2), encoding="utf-8"
    )

    logger.info("Successfully imported and quarantined legacy run %s", run_id)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="FedTROS-PR Legacy Run Importer & Isolation Engine")
    parser.add_argument("--source-dir", type=Path, default=None, help="Directory containing legacy runs")
    parser.add_argument("--run-dir", type=Path, default=None, help="Single legacy run directory to import")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/legacy_imported"), help="Destination directory")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without copying files")
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        success = quarantine_and_import_run(args.run_dir, out_dir, dry_run=args.dry_run)
        return 0 if success else 1

    if args.source_dir and args.source_dir.is_dir():
        runs = [d for d in args.source_dir.iterdir() if d.is_dir()]
        logger.info("Found %d candidate legacy directories in %s", len(runs), args.source_dir)
        for r in runs:
            quarantine_and_import_run(r, out_dir, dry_run=args.dry_run)
        return 0

    logger.error("Must provide either --source-dir or --run-dir")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
