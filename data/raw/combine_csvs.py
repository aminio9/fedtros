from pathlib import Path

import pandas as pd

COLUMN_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "src_bytes",
    "dst_bytes",
    "flag",
    "count",
    "srv_count",
    "serror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_serror_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_serror_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_srv_serror_rate",
    "label",
]

INPUT_FILES = [
    Path("data/raw/w1.csv"),
    Path("data/raw/w2.csv"),
    Path("data/raw/w3.csv"),
]
OUTPUT_FILE = Path("data/raw/BNaT.csv")


def read_no_header_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        dtype=str,
        sep=",",
        engine="python",
        on_bad_lines="warn",
    )

    if df.shape[1] > len(COLUMN_NAMES):
        df = df.iloc[:, : len(COLUMN_NAMES)]

    return df.apply(lambda column: column.astype(str).str.strip())


def combine_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = read_no_header_csv(path)
        if df.shape[1] != len(COLUMN_NAMES):
            print(
                f"[WARN] {path} has {df.shape[1]} columns; expected {len(COLUMN_NAMES)}. "
                "Extra columns, if present, were dropped."
            )
        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True)


def main() -> None:
    combined = combine_csvs(INPUT_FILES)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE.name} with {len(combined)} rows and {len(combined.columns)} columns")


if __name__ == "__main__":
    main()
