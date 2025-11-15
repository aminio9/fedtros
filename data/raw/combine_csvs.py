# combine_csvs.py
import pandas as pd
from pathlib import Path

COL_NAMES = [
    "duration","protocol_type","service","src_bytes","dst_bytes","flag","count","srv_count",
    "serror_rate","same_srv_rate","diff_srv_rate","srv_serror_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_serror_rate","dst_host_srv_diff_host_rate",
    "dst_host_srv_serror_rate","label",
]

files = [
    "data/raw/w1.csv",
    "data/raw/w2.csv",
    "data/raw/w3.csv",
]

def read_no_header_csv(path: str) -> pd.DataFrame:
    # If some rows have extra columns, we keep only the first len(COL_NAMES)
    df = pd.read_csv(
        path,
        header=None,                 # <-- no header in source files
        names=COL_NAMES,             # <-- assign your headers
        dtype=str,                   # keep everything as string (safe)
        sep=",",                     # change to sep=r"\s+" if they’re whitespace-separated
        engine="python",             # more tolerant for ragged rows
        on_bad_lines="warn",         # skip/flag bad rows (pandas >= 1.3)
    )

    # If file had more columns than expected, trim them
    if df.shape[1] > len(COL_NAMES):
        df = df.iloc[:, :len(COL_NAMES)]

    # Basic cleanup: strip surrounding whitespace
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    return df

frames = []
for f in files:
    df = read_no_header_csv(f)
    # Quick sanity check
    if df.shape[1] != len(COL_NAMES):
        print(f"[WARN] {f} has {df.shape[1]} columns; expected {len(COL_NAMES)}. "
              "Extra columns (if any) were dropped.")
    frames.append(df)

combined = pd.concat(frames, axis=0, ignore_index=True)

out_path = Path("data/raw/Bnat.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out_path, index=False)

print(f"Wrote {out_path.name} with {len(combined)} rows and {len(combined.columns)} columns")
