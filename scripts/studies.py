#!/usr/bin/env python3
"""Inspect the declarative FedTROS-MC experiment contract."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.infrastructure.study import CANONICAL_SEEDS, expand_study_matrix, load_study_config


def _study_files() -> list[Path]:
    return sorted((_ROOT / "src" / "configs" / "study").glob("*.yaml"))


def cmd_list() -> int:
    print("FedTROS-MC canonical studies")
    print("=" * 92)
    print(f"{'Study':<20} {'Datasets':<28} {'Methods':<28} Description")
    print("-" * 92)
    for path in _study_files():
        cfg = load_study_config(path, _ROOT)
        print(
            f"{str(cfg.get('study_id', path.stem)):<20} "
            f"{','.join(map(str, cfg.get('datasets', []))):<28} "
            f"{','.join(map(str, cfg.get('methods', []))):<28} "
            f"{cfg.get('description', '')}"
        )
    print("=" * 92)
    print("Headline seed policy:", ", ".join(map(str, CANONICAL_SEEDS)))
    return 0


def cmd_show(study: str, stage: str) -> int:
    cfg = load_study_config(study, _ROOT)
    plans = expand_study_matrix(cfg, stage=stage, project_root=_ROOT)
    print(f"Study       : {cfg.get('study_id', study)}")
    print(f"Name        : {cfg.get('name', '')}")
    print(f"Description : {cfg.get('description', '')}")
    print(f"Stage       : {stage}")
    print(f"Datasets    : {cfg.get('datasets', [])}")
    print(f"Methods     : {cfg.get('methods', [])}")
    print(f"Alphas      : {cfg.get('alphas', [])}")
    print(f"IID flags   : {cfg.get('iids', [])}")
    print(f"Seeds       : {cfg.get('seeds', list(CANONICAL_SEEDS))}")
    print(f"Clients     : {cfg.get('num_clients_values', [cfg.get('num_clients', 10)])}")
    print(f"Variants    : {[v.get('name', v) if isinstance(v, dict) else v for v in cfg.get('variants', ['canonical'])]}")
    print(f"Expanded runs: {len(plans)}")
    print("\nFirst planned runs:")
    for run in plans[:12]:
        print(f"  {run.run_id}")
    if len(plans) > 12:
        print(f"  ... {len(plans)-12} more")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect FedTROS-MC study definitions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List canonical studies")
    show = sub.add_parser("show", help="Show one study and its planned matrix")
    show.add_argument("study")
    show.add_argument("--stage", default="paper_final")
    args = parser.parse_args()
    return cmd_list() if args.command == "list" else cmd_show(args.study, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
