from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path("BTAT")
OUTPUT_CSV = Path("BTAT_dataset_final.csv")
TARGET_FOLDERS = (
    "Normal",
    "DoS",
    "Delegatecall",
    "FoT",
    "OaU",
    "Reentrancy",
    "FDV",
)


def iter_json_records(file_path: Path) -> Iterator[dict[str, Any]]:
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            handle.seek(0)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    yield item
        else:
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    yield item


def should_skip_file(path: Path) -> bool:
    return path.name.startswith(".") or path.suffix.lower() == ".md"


def load_btat_optimized(root_dir: str | Path) -> pd.DataFrame:
    root_path = Path(root_dir)
    all_data: list[dict[str, Any]] = []
    total_files = 0
    start_time = time.time()

    print(f"Scanning directory: {root_path.resolve()}")

    for target in TARGET_FOLDERS:
        start_path = root_path / target
        if not start_path.exists():
            print(f"Skipping {target} (not found)")
            continue

        for current_root, _, files in os.walk(start_path):
            current_path = Path(current_root)
            label = "_".join(current_path.relative_to(root_path).parts)
            print(f"--> Processing folder: {label} | Found {len(files)} files...")

            for index, filename in enumerate(files):
                if index > 0 and index % 100 == 0:
                    print(f"    Processed {index} files in {label}...", end="\r")

                file_path = current_path / filename
                if should_skip_file(file_path):
                    continue

                try:
                    for item in iter_json_records(file_path):
                        item["label"] = label
                        all_data.append(item)
                except OSError as exc:
                    print(f"Error reading {filename}: {exc}")
                    continue

                total_files += 1

            print(f"    Finished folder: {label}                    ")

    elapsed = time.time() - start_time
    print("=" * 40)
    print(f"Done. Processed {total_files} files in {elapsed:.2f}s")
    return pd.DataFrame(all_data)


def main() -> None:
    df = load_btat_optimized(REPO_ROOT)
    if df.empty:
        print("No data extracted. Please check the path.")
        return

    print(f"Saving {len(df)} rows to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Success.")


if __name__ == "__main__":
    main()
