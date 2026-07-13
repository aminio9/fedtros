from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = ("btat", "toniot", "cicids2017")
RAW_FILES = {
    "btat": Path("data/raw/BTAT.csv"),
    "toniot": Path("data/raw/ToN-IoT.csv"),
    "cicids2017": Path("data/raw/CIC-IDS2017.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen DKD-FedOS E5 protocol.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=(*DATASETS, "all"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", choices=("capped", "full"), default="capped")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; may be passed multiple times.",
    )
    return parser.parse_args()


def selected_datasets(values: list[str]) -> list[str]:
    if "all" in values:
        return list(DATASETS)
    return list(dict.fromkeys(values))


def build_command(dataset: str, args: argparse.Namespace) -> list[str]:
    smoke_prefix = "smoke_" if args.smoke else ""
    run_id = f"{smoke_prefix}e5_{dataset}_noniid_alpha05_open_dkd_fedos_seed{args.seed}"
    command = [
        sys.executable,
        "run.py",
        "experiment=exp5",
        f"dataset={dataset}",
        "+method=dkd_fedos",
        f"seed={args.seed}",
        f"tracking.run_id={run_id}",
    ]
    if args.smoke:
        command.extend(
            [
                "federated.num_rounds=1",
                "training.local_episodes_per_round=1",
            ]
        )
    if args.profile == "full":
        command.append("dataset.preprocessing.max_samples_per_class=null")
    command.extend(args.override)
    return command


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise SystemExit("E5 is frozen to seed 42.")
    datasets = selected_datasets(args.datasets)
    missing = [str(RAW_FILES[name]) for name in datasets if not RAW_FILES[name].exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise SystemExit(
            "Canonical datasets are missing. Run the preparation command first:\n"
            f"  - {formatted}\n"
            "poetry run python scripts/prepare_external_datasets.py --dataset all"
        )
    for dataset in datasets:
        command = build_command(dataset, args)
        print(f"Running E5 dataset={dataset} profile={args.profile} smoke={args.smoke}")
        subprocess.run(command, cwd=Path.cwd(), check=True)


if __name__ == "__main__":
    main()
