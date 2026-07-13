from __future__ import annotations

import argparse
from pathlib import Path

from src.data.external_datasets import prepare_external_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, validate, and convert BTAT, ToN-IoT, and CIC-IDS2017."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("btat", "toniot", "cicids2017", "all"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = prepare_external_dataset(args.dataset, raw_root=args.raw_root.resolve())
    for output in outputs:
        print(f"Prepared {output}")


if __name__ == "__main__":
    main()
