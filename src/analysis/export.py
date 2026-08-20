"""Standardized data contract exporters and provenance tracking (Items C14–C17, C20).

Serializes canonical data artifacts matching strict scientific schemas:
  - C14: OSR Sample-Level Result Contract
  - C15: Client-Level Result Contract
  - C16: Communication Data Contract
  - C17: Runtime / Timing Data Contract
  - C20: Provenance Manifest Generator
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.analysis.loaders import RunRecord

logger = logging.getLogger(__name__)


def export_osr_sample_contract(
    scores_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Export sample-level open-set predictions matching the C14 schema.

    Schema:
      sample_id, true_label, closed_pred, open_pred, unknown_flag, raw_score, rank_score, is_rejected
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = scores_df.copy()

    # Column mapping & normalization
    sample_id = df["sample_id"] if "sample_id" in df.columns else np.arange(len(df))
    true_label = (
        df["y_true"]
        if "y_true" in df.columns
        else (df["y_raw"] if "y_raw" in df.columns else df.get("true_label", 0))
    )
    closed_pred = (
        df["raw_pred"]
        if "raw_pred" in df.columns
        else (
            df["pred_before_osr"] if "pred_before_osr" in df.columns else df.get("closed_pred", 0)
        )
    )
    open_pred = (
        df["y_pred"]
        if "y_pred" in df.columns
        else (df["final_pred"] if "final_pred" in df.columns else df.get("open_pred", 0))
    )

    # Unknown flag
    if "known_or_unknown" in df.columns:
        unknown_flag = np.where(df["known_or_unknown"].astype(str).str.lower() == "unknown", 1, 0)
    elif "is_unknown" in df.columns:
        unknown_flag = df["is_unknown"].astype(int)
    else:
        unknown_flag = np.where(true_label == 99, 1, 0)

    raw_score = (
        df["raw_score"]
        if "raw_score" in df.columns
        else (df["recon_error"] if "recon_error" in df.columns else df.get("proser_score", 0.0))
    )
    rank_score = (
        df["prototype_rank_score"]
        if "prototype_rank_score" in df.columns
        else (df["unknown_score"] if "unknown_score" in df.columns else df.get("rank_score", 0.0))
    )
    is_rejected = (
        df["final_reject"] if "final_reject" in df.columns else np.where(open_pred == 99, 1, 0)
    )

    contract_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "true_label": true_label,
            "closed_pred": closed_pred,
            "open_pred": open_pred,
            "unknown_flag": unknown_flag,
            "raw_score": pd.to_numeric(raw_score, errors="coerce").fillna(0.0),
            "rank_score": pd.to_numeric(rank_score, errors="coerce").fillna(0.0),
            "is_rejected": pd.to_numeric(is_rejected, errors="coerce").fillna(0).astype(int),
        }
    )
    contract_df.to_csv(output_path, index=False)
    logger.info(
        "Saved C14 OSR sample-level contract to %s (%d rows)", output_path, len(contract_df)
    )
    return contract_df


def export_client_level_contract(
    client_metrics_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Export client-level performance traces matching the C15 schema.

    Schema:
      round, client_id, sample_count, class_count, class_coverage, accuracy, macro_f1
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = client_metrics_df.copy()

    round_num = df["round"] if "round" in df.columns else 1
    client_id = df["client_id"] if "client_id" in df.columns else 0
    sample_count = df["num_examples"] if "num_examples" in df.columns else df.get("sample_count", 0)
    class_count = df["class_count"] if "class_count" in df.columns else 0
    class_coverage = df["class_coverage"] if "class_coverage" in df.columns else 1.0
    accuracy = df["accuracy"] if "accuracy" in df.columns else 0.0
    macro_f1 = df["macro_f1"] if "macro_f1" in df.columns else 0.0

    contract_df = pd.DataFrame(
        {
            "round": round_num,
            "client_id": client_id,
            "sample_count": sample_count,
            "class_count": class_count,
            "class_coverage": class_coverage,
            "accuracy": pd.to_numeric(accuracy, errors="coerce"),
            "macro_f1": pd.to_numeric(macro_f1, errors="coerce"),
        }
    )
    contract_df.to_csv(output_path, index=False)
    logger.info("Saved C15 client-level contract to %s (%d rows)", output_path, len(contract_df))
    return contract_df


def export_communication_contract(
    comm_df: pd.DataFrame,
    method_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """Export communication efficiency metrics matching the C16 schema.

    Schema:
      round, method, downlink_bytes, uplink_bytes, total_bytes, cumulative_bytes, validation_accuracy, validation_macro_f1
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = comm_df.copy()

    round_num = df["round"] if "round" in df.columns else np.arange(1, len(df) + 1)
    downlink = (
        df["communication/downlink_bytes"]
        if "communication/downlink_bytes" in df.columns
        else (df["downlink_bytes"] if "downlink_bytes" in df.columns else 0)
    )
    uplink = (
        df["communication/uplink_bytes"]
        if "communication/uplink_bytes" in df.columns
        else (
            df["uplink_bytes"]
            if "uplink_bytes" in df.columns
            else (df["total_bytes"] if "total_bytes" in df.columns else 0)
        )
    )
    total = (
        df["communication/round_bytes"]
        if "communication/round_bytes" in df.columns
        else (df["total_bytes"] if "total_bytes" in df.columns else (downlink + uplink))
    )

    if "communication/cumulative_bytes" in df.columns:
        cumulative = df["communication/cumulative_bytes"]
    elif "cumulative_bytes" in df.columns:
        cumulative = df["cumulative_bytes"]
    elif "cumulative_mb" in df.columns:
        cumulative = df["cumulative_mb"] * (1024 * 1024)
    else:
        cumulative = total.cumsum()

    val_acc = (
        df["validation_accuracy"]
        if "validation_accuracy" in df.columns
        else df.get("accuracy", np.nan)
    )
    val_f1 = (
        df["validation_macro_f1"]
        if "validation_macro_f1" in df.columns
        else df.get("macro_f1", np.nan)
    )

    contract_df = pd.DataFrame(
        {
            "round": round_num,
            "method": method_name,
            "downlink_bytes": downlink,
            "uplink_bytes": uplink,
            "total_bytes": total,
            "cumulative_bytes": cumulative,
            "validation_accuracy": pd.to_numeric(val_acc, errors="coerce"),
            "validation_macro_f1": pd.to_numeric(val_f1, errors="coerce"),
        }
    )
    contract_df.to_csv(output_path, index=False)
    logger.info("Saved C16 communication contract to %s (%d rows)", output_path, len(contract_df))
    return contract_df


def export_runtime_contract(
    timing_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Export per-round runtime decomposition matching the C17 schema.

    Schema:
      round, client_fit_seconds, teacher_seconds, student_seconds, aggregation_seconds,
      open_set_eval_seconds, orchestration_seconds, round_seconds, cumulative_seconds
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = timing_df.copy()

    round_num = df["round"] if "round" in df.columns else np.arange(1, len(df) + 1)
    fit_s = pd.to_numeric(
        df.get("runtime/client_fit_seconds", df.get("client_fit_wall_time_sec", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    teacher_s = pd.to_numeric(
        df.get("runtime/teacher_seconds", df.get("teacher_train_sec", 0.0)), errors="coerce"
    ).fillna(0.0)
    student_s = pd.to_numeric(
        df.get("runtime/student_seconds", df.get("student_train_sec", fit_s)), errors="coerce"
    ).fillna(0.0)
    agg_s = pd.to_numeric(
        df.get("runtime/aggregation_seconds", df.get("server_aggregation_time_sec", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    eval_s = pd.to_numeric(
        df.get("runtime/open_set_eval_seconds", df.get("open_set_round_eval_time_sec", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    round_s = pd.to_numeric(
        df.get("runtime/round_seconds", df.get("round_time_sec", fit_s + agg_s + eval_s)),
        errors="coerce",
    ).fillna(0.0)
    orch_s = pd.to_numeric(
        df.get(
            "runtime/orchestration_seconds", np.maximum(round_s - (fit_s + agg_s + eval_s), 0.0)
        ),
        errors="coerce",
    ).fillna(0.0)
    cum_s = pd.to_numeric(
        df.get("runtime/cumulative_seconds", round_s.cumsum()), errors="coerce"
    ).fillna(0.0)

    contract_df = pd.DataFrame(
        {
            "round": round_num,
            "client_fit_seconds": fit_s,
            "teacher_seconds": teacher_s,
            "student_seconds": student_s,
            "aggregation_seconds": agg_s,
            "open_set_eval_seconds": eval_s,
            "orchestration_seconds": orch_s,
            "round_seconds": round_s,
            "cumulative_seconds": cum_s,
        }
    )
    contract_df.to_csv(output_path, index=False)
    logger.info("Saved C17 runtime contract to %s (%d rows)", output_path, len(contract_df))
    return contract_df


# Metric priority for the Q1 E7 performance-vs-communication curve.  The first
# available metric is used consistently per run; validation metrics are preferred
# over client-evaluation summaries and training/fit diagnostics.
_EFFICIENCY_PERFORMANCE_CANDIDATES: tuple[str, ...] = (
    "val/macro_f1",
    "federated/global_macro_f1",
    "eval_client_macro_f1_mean",
    "closed_set/macro_f1",
    "validation_macro_f1",
    "student_f1_macro",
    "f1_macro",
    "macro_f1",
    "val/accuracy",
    "federated/global_accuracy",
    "eval_client_accuracy_mean",
    "closed_set/accuracy",
    "validation_accuracy",
    "student_accuracy",
    "accuracy",
)


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _performance_series_from_history(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    """Resolve one recorded validation/performance series from canonical round history.

    This function performs *selection*, not scientific recomputation.  It prefers a
    central-validation event when several events exist for the same federated round,
    followed by distributed client evaluation and then fit summaries.
    """
    if history is None or history.empty:
        return None, None
    round_col = _first_column(history, ("federated/round", "round", "server_round"))
    if round_col is None:
        return None, None
    phase_col = _first_column(history, ("federated/phase", "phase"))
    phase_priority = {"central_validation": 0, "client_evaluate": 1, "fit": 2}

    for metric in _EFFICIENCY_PERFORMANCE_CANDIDATES:
        if metric not in history.columns:
            continue
        x = history[[round_col, metric] + ([phase_col] if phase_col else [])].copy()
        x["_round"] = pd.to_numeric(x[round_col], errors="coerce")
        x["_value"] = pd.to_numeric(x[metric], errors="coerce")
        x = x.dropna(subset=["_round", "_value"])
        if x.empty:
            continue
        if phase_col:
            x["_phase_order"] = x[phase_col].astype(str).map(phase_priority).fillna(9)
            x = x.sort_values(["_round", "_phase_order"], kind="stable")
        else:
            x = x.sort_values("_round", kind="stable")
        x = x.drop_duplicates("_round", keep="first")
        return x[["_round", "_value"]].rename(
            columns={"_round": "round", "_value": "performance_value"}
        ), metric
    return None, None


def build_efficiency_curve(runs: Sequence[RunRecord]) -> pd.DataFrame:
    """Build the canonical E7 performance-vs-communication publication contract.

    Communication is taken from the actually recorded payload accounting.  Performance
    is *not* recomputed here: it is selected from the communication trace when already
    present, otherwise from the canonical round-metric history and joined by round.

    Output columns are deliberately stable for the separate plotting repository:
    ``round``, ``communication/cumulative_bytes``, ``performance_metric``, and
    ``performance_value`` plus immutable run metadata.
    """
    frames: list[pd.DataFrame] = []
    cumulative_candidates = (
        "communication/cumulative_bytes",
        "cumulative_bytes",
        "cumulative_bidirectional_bytes",
        "cumulative_model_bytes",
    )
    round_bytes_candidates = ("communication/round_bytes", "round_bytes", "total_bytes")

    for run in runs:
        comm = run.communication
        if comm is None or comm.empty:
            continue
        c = comm.copy()
        round_col = _first_column(c, ("round", "federated/round", "server_round"))
        if round_col is None:
            c["round"] = np.arange(1, len(c) + 1, dtype=int)
        else:
            c["round"] = pd.to_numeric(c[round_col], errors="coerce")
        c = c.dropna(subset=["round"]).copy()
        c["round"] = c["round"].astype(int)

        cumulative_col = _first_column(c, cumulative_candidates)
        if cumulative_col is not None:
            c["communication/cumulative_bytes"] = pd.to_numeric(c[cumulative_col], errors="coerce")
        else:
            round_bytes_col = _first_column(c, round_bytes_candidates)
            if round_bytes_col is None:
                continue
            c["communication/cumulative_bytes"] = (
                pd.to_numeric(c[round_bytes_col], errors="coerce").fillna(0.0).cumsum()
            )

        # Prefer performance already colocated with the communication trace (legacy
        # imports and some instrumented runs do this), otherwise join the canonical
        # ResultStore round history.
        perf_metric = next(
            (
                m
                for m in _EFFICIENCY_PERFORMANCE_CANDIDATES
                if m in c.columns and pd.to_numeric(c[m], errors="coerce").notna().any()
            ),
            None,
        )
        if perf_metric is not None:
            c["performance_value"] = pd.to_numeric(c[perf_metric], errors="coerce")
            joined = c[["round", "communication/cumulative_bytes", "performance_value"]].copy()
        else:
            perf, perf_metric = _performance_series_from_history(run.history)
            if perf is None or perf_metric is None:
                continue
            joined = c[["round", "communication/cumulative_bytes"]].merge(
                perf, on="round", how="inner"
            )

        joined = joined.dropna(subset=["communication/cumulative_bytes", "performance_value"])
        if joined.empty:
            continue
        joined.insert(0, "run_id", run.run_id)
        joined.insert(1, "study", run.study)
        joined.insert(2, "method", run.method)
        joined.insert(3, "dataset", run.dataset)
        joined.insert(4, "alpha", run.alpha)
        joined.insert(5, "seed", run.seed)
        joined.insert(6, "num_clients", run.num_clients)
        joined.insert(7, "variant", run.variant)
        joined.insert(8, "unknown_labels", "|".join(run.unknown_labels))
        joined["performance_metric"] = str(perf_metric)
        frames.append(joined)

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["method", "seed", "round"], kind="stable")
        .reset_index(drop=True)
    )


def generate_provenance_manifest(
    runs: Sequence[RunRecord],
    output_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate C20 provenance manifest linking all artifacts to source runs, commits, and configs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "analysis_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_source_runs": len(runs),
        "source_run_ids": [r.run_id for r in runs],
        "studies": sorted({r.study for r in runs}),
        "stages": sorted({r.stage for r in runs}),
        "methods": sorted({r.method for r in runs}),
        "datasets": sorted({r.dataset for r in runs}),
        "seeds": sorted({r.seed for r in runs}),
        "git_commits": sorted({r.git_commit for r in runs if r.git_commit}),
        "config_hashes": sorted({r.config_hash for r in runs if r.config_hash}),
        "runs_summary": [
            {
                "run_id": r.run_id,
                "study": r.study,
                "stage": r.stage,
                "method": r.method,
                "dataset": r.dataset,
                "alpha": r.alpha,
                "seed": r.seed,
                "num_clients": r.num_clients,
                "status": r.status,
                "config_hash": r.config_hash,
                "git_commit": r.git_commit,
                "timestamp_utc": r.timestamp_utc,
            }
            for r in runs
        ],
        **(extra_metadata or {}),
    }

    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Saved C20 provenance manifest to %s", output_path)
    return manifest
