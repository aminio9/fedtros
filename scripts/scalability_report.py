from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts.communication import build_communication_metrics  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_cfg(run_dir: Path) -> Any:
    path = run_dir / "resolved_config.yaml"
    if not path.exists():
        return None
    try:
        return OmegaConf.load(path)
    except Exception:
        return None


def _cfg_select(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        value = OmegaConf.select(cfg, key, default=default)
    except Exception:
        value = default
    return default if value in (None, "???") else value


def _metric(metrics: dict[str, Any], candidates: Iterable[str]) -> float | None:
    for key in candidates:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except Exception:
        return {}


def _load_final_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "evaluation_metrics.json")
    if metrics:
        return metrics
    metrics = _load_json(run_dir / "open_set_metrics.json")
    if metrics:
        return metrics
    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        try:
            frame = pd.read_csv(metrics_csv)
            if not frame.empty:
                return frame.iloc[-1].dropna().to_dict()
        except Exception:
            pass
    return _last_jsonl(run_dir / "metrics.jsonl")


def _load_round_metrics(run_dir: Path) -> pd.DataFrame:
    open_path = run_dir / "open_set_round_metrics.csv"
    if open_path.exists():
        frame = pd.read_csv(open_path)
        if not frame.empty:
            return frame

    scalability_path = run_dir / "scalability_round_metrics.csv"
    if scalability_path.exists():
        frame = pd.read_csv(scalability_path)
        if not frame.empty:
            rename = {
                "round_openset_f1_macro": "openset_f1_macro",
                "round_openset_overall_acc": "openset_overall_acc",
                "round_openset_known_acc": "openset_known_acc",
                "round_openset_auroc": "openset_auroc",
                "round_openset_fpr95": "openset_fpr95",
                "round_openset_unknown_recall": "openset_unknown_recall",
            }
            return frame.rename(columns={old: new for old, new in rename.items() if old in frame.columns})

    history_path = run_dir / "federated_history.csv"
    if not history_path.exists():
        return pd.DataFrame()
    history = pd.read_csv(history_path)
    if not {"round", "metric_name", "metric_value"}.issubset(history.columns):
        return pd.DataFrame()
    keep = history[history["metric_name"].astype(str).str.startswith("round_openset_")].copy()
    if keep.empty:
        return pd.DataFrame()
    keep["metric_name"] = keep["metric_name"].astype(str).str.replace("^round_", "", regex=True)
    return keep.pivot_table(index="round", columns="metric_name", values="metric_value", aggfunc="last").reset_index()


def _load_communication(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "communication_metrics.csv"
    if path.exists():
        try:
            frame = pd.read_csv(path)
            if not frame.empty:
                return frame
        except Exception:
            pass
    frame = build_communication_metrics(run_dir=run_dir, project_root=PROJECT_ROOT)
    if not frame.empty:
        frame.to_csv(path, index=False)
    return frame


def _run_summary(run_dir: Path) -> dict[str, Any]:
    cfg = _load_cfg(run_dir)
    metadata = _load_json(run_dir / "metadata.json")
    final_metrics = _load_final_metrics(run_dir)
    federated_summary = _load_json(run_dir / "federated_summary.json")
    round_metrics = _load_round_metrics(run_dir)
    communication = _load_communication(run_dir)

    num_clients = int(_cfg_select(cfg, "federated.num_clients", metadata.get("num_clients", 0)) or 0)
    seed = int(_cfg_select(cfg, "seed", metadata.get("seed", 0)) or 0)
    alpha = float(_cfg_select(cfg, "dataset.preprocessing.alpha", metadata.get("alpha", 0.0)) or 0.0)
    run_id = str(metadata.get("run_id", run_dir.name))

    final_round = round_metrics.sort_values("round").tail(1).iloc[0].to_dict() if not round_metrics.empty else {}
    last_scalability = {}
    scalability_path = run_dir / "scalability_round_metrics.csv"
    if scalability_path.exists():
        try:
            sf = pd.read_csv(scalability_path)
            if not sf.empty:
                last_scalability = sf.sort_values("round").tail(1).iloc[0].to_dict()
        except Exception:
            pass

    total_comm = None
    comm_per_round = None
    if not communication.empty:
        last_comm = communication.sort_values("round").tail(1).iloc[0]
        total_comm = float(last_comm.get("total_communication_mb", last_comm.get("cumulative_mb", float("nan"))))
        comm_per_round = float(last_comm.get("communication_per_round_mb", float("nan")))

    avg_round_time = _metric(federated_summary, ("federated/avg_round_time_sec",))
    total_time = _metric(federated_summary, ("federated/total_training_time_sec",))
    if avg_round_time is None and scalability_path.exists():
        try:
            sf = pd.read_csv(scalability_path)
            if "round_time_sec" in sf.columns:
                avg_round_time = float(pd.to_numeric(sf["round_time_sec"], errors="coerce").dropna().mean())
        except Exception:
            pass

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "num_clients": num_clients,
        "seed": seed,
        "alpha": alpha,
        "accuracy": _metric(final_metrics, ("openset_overall_acc", "accuracy", "test/accuracy")),
        "macro_f1": _metric(final_metrics, ("openset_f1_macro", "macro_f1", "f1_macro", "test/macro_f1")),
        "known_accuracy": _metric(final_metrics, ("openset_known_acc", "known_accuracy")),
        "auroc": _metric(final_metrics, ("openset_auroc", "open_set/auroc", "auroc")),
        "fpr95": _metric(final_metrics, ("openset_fpr95", "open_set/fpr95", "fpr95")),
        "unknown_recall": _metric(final_metrics, ("openset_unknown_recall", "unknown_recall")),
        "final_round_macro_f1": _metric(final_round, ("openset_f1_macro",)),
        "final_round_auroc": _metric(final_round, ("openset_auroc",)),
        "avg_round_time_sec": avg_round_time,
        "total_training_time_sec": total_time,
        "communication_per_round_mb": comm_per_round,
        "total_communication_mb": total_comm,
        "mean_client_macro_f1": _metric(last_scalability, ("mean_client_macro_f1", "client_macro_f1_mean")),
        "std_client_macro_f1": _metric(last_scalability, ("std_client_macro_f1", "client_macro_f1_std")),
        "worst_client_macro_f1": _metric(last_scalability, ("worst_client_macro_f1", "client_macro_f1_worst")),
    }


def _retention(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty or not {50, 100}.issubset(set(summary["num_clients"].astype(int))):
        return {}
    row50 = summary.loc[summary["num_clients"].astype(int) == 50].iloc[0]
    row100 = summary.loc[summary["num_clients"].astype(int) == 100].iloc[0]
    ratios: dict[str, Any] = {}
    for metric in ("macro_f1", "auroc", "accuracy", "known_accuracy", "worst_client_macro_f1"):
        a = row50.get(metric)
        b = row100.get(metric)
        if pd.notna(a) and pd.notna(b) and float(a) != 0.0:
            ratios[f"{metric}_retention_100_over_50"] = float(b) / float(a)
    for metric in ("avg_round_time_sec", "total_training_time_sec", "total_communication_mb"):
        a = row50.get(metric)
        b = row100.get(metric)
        if pd.notna(a) and pd.notna(b) and float(a) != 0.0:
            ratios[f"{metric}_growth_100_over_50"] = float(b) / float(a)
    return ratios


def build_report(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_run_summary(run_dir) for run_dir in run_dirs]
    summary_df = pd.DataFrame(summaries).sort_values("num_clients").reset_index(drop=True)
    summary_path = output_dir / "scalability_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    round_frames: list[pd.DataFrame] = []
    for summary, run_dir in zip(summaries, run_dirs, strict=True):
        frame = _load_round_metrics(run_dir)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["num_clients"] = int(summary["num_clients"])
        frame["run_id"] = str(summary["run_id"])
        round_frames.append(frame)
    round_df = pd.concat(round_frames, ignore_index=True) if round_frames else pd.DataFrame()
    round_path = output_dir / "scalability_round_curves.csv"
    if not round_df.empty:
        round_df.to_csv(round_path, index=False)

    retention = _retention(summary_df)
    retention_path = output_dir / "scalability_retention.json"
    retention_path.write_text(json.dumps(retention, indent=2, sort_keys=True), encoding="utf-8")

    generated: list[str] = [str(summary_path), str(retention_path)]
    if not round_df.empty:
        generated.append(str(round_path))

    manifest = {"runs": [str(run_dir) for run_dir in run_dirs], "generated_files": generated}
    manifest_path = output_dir / "scalability_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data-only 50/100-client scalability summaries.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to compare.")
    parser.add_argument("--output", required=True, help="Output report directory.")
    args = parser.parse_args()

    run_dirs = [_resolve(path) for path in args.runs]
    missing = [str(path) for path in run_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run directories not found: {missing}")
    manifest = build_report(run_dirs, _resolve(args.output))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
