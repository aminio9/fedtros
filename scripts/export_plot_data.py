#!/usr/bin/env python3
"""Good-Plot-Project Adapter Script (Item C10).

Transforms canonical FedTROS experiment outputs into the exact file and schema
contract expected by the external publication plotting repository (plots/).

Usage:
    python scripts/export_plot_data.py --study E4-NIID-FOSR --output-dir paper_artifacts/plot_data/
    python scripts/export_plot_data.py --output-dir ../plots/data/processed/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.analysis.export import (
    export_client_level_contract,
    export_communication_contract,
    export_osr_sample_contract,
    export_runtime_contract,
    generate_provenance_manifest,
)
from src.analysis.loaders import RunRecord, load_run
from src.analysis.query import query_runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ExportPlotData")


def export_scalability_series(runs: list[RunRecord], output_dir: Path) -> dict[str, Path]:
    """Export scalability_{10,50,100}_clients.csv files from matching runs."""
    exported: dict[str, Path] = {}

    for client_count in (10, 50, 100):
        # Look for run with specific num_clients or in scalability round metrics
        matching = [r for r in runs if r.num_clients == client_count]
        if not matching:
            # Check if any run has scalability table containing this client count
            for r in runs:
                sc = r.scalability
                if (
                    not sc.empty
                    and "num_clients" in sc.columns
                    and client_count in sc["num_clients"].values
                ):
                    matching.append(r)
                    break

        if matching:
            r = matching[0]
            sc_df = r.scalability
            if not sc_df.empty:
                target_p = output_dir / f"scalability_{client_count}_clients.csv"
                sc_df.to_csv(target_p, index=False)
                exported[f"scalability_{client_count}"] = target_p
                logger.info("Exported scalability log for %d clients -> %s", client_count, target_p)

    return exported


def export_convergence_series(runs: list[RunRecord], output_dir: Path) -> dict[str, Path]:
    """Export convergence_alpha1.csv and convergence_alpha01.csv for Figures 03 and 04."""
    exported: dict[str, Path] = {}

    for alpha_val, stem in [(1.0, "convergence_alpha1"), (0.1, "convergence_alpha01")]:
        matching = [r for r in runs if abs(r.alpha - alpha_val) < 1e-4]
        rows: list[dict[str, Any]] = []

        for r in matching:
            hist = r.history
            if hist.empty:
                continue
            if "metric_name" in hist.columns and "metric_value" in hist.columns:
                m_rows = hist[
                    hist["metric_name"].isin(
                        ["round_openset_f1_macro", "local_student_f1_macro", "macro_f1"]
                    )
                ]
                for _, row in m_rows.iterrows():
                    rows.append(
                        {
                            "round": int(row.get("round", 1)),
                            "method": r.method,
                            "macro_f1_percent": float(row.get("metric_value", 0.0)) * 100.0,
                            "seed": r.seed,
                        }
                    )
            elif "macro_f1" in hist.columns or "round_openset_f1_macro" in hist.columns:
                col = (
                    "round_openset_f1_macro"
                    if "round_openset_f1_macro" in hist.columns
                    else "macro_f1"
                )
                for _, row in hist.iterrows():
                    rows.append(
                        {
                            "round": int(row.get("round", 1)),
                            "method": r.method,
                            "macro_f1_percent": float(row.get(col, 0.0)) * 100.0,
                            "seed": r.seed,
                        }
                    )

        if rows:
            df = pd.DataFrame(rows)
            # Compute mean and standard error band per round and method
            grouped = (
                df.groupby(["round", "method"])["macro_f1_percent"]
                .agg(["mean", "std"])
                .reset_index()
            )
            grouped.columns = ["round", "method", "macro_f1_percent", "band"]
            grouped["band"] = grouped["band"].fillna(0.0)
            target_p = output_dir / f"{stem}.csv"
            grouped.to_csv(target_p, index=False)
            exported[stem] = target_p
            logger.info("Exported convergence trajectory -> %s (%d rows)", target_p, len(grouped))

    return exported


def export_osr_curves_and_scores(runs: list[RunRecord], output_dir: Path) -> dict[str, Path]:
    """Export exp2_scores.csv, exp2_roc_curve.csv, exp2_pr_curve.csv, and confusion matrices."""
    exported: dict[str, Path] = {}

    # Select best or canonical FedTROS-PR run with open set scores
    osr_runs = [r for r in runs if not r.scores.empty]
    if not osr_runs:
        logger.warning("No runs with open-set score files found.")
        return exported

    target_run = osr_runs[0]
    scores_df = target_run.scores

    # 1. exp2_scores.csv (Sample level contract)
    scores_target = output_dir / "exp2_scores.csv"
    export_osr_sample_contract(scores_df, scores_target)
    exported["exp2_scores"] = scores_target

    # 2. ROC and PR curves
    y_true = (
        np.where(scores_df["known_or_unknown"].astype(str).str.lower() == "unknown", 1, 0)
        if "known_or_unknown" in scores_df.columns
        else scores_df.get("is_unknown", np.zeros(len(scores_df)))
    )
    score_vals = pd.to_numeric(
        scores_df.get("prototype_rank_score", scores_df.get("unknown_score", 0.0)), errors="coerce"
    ).fillna(0.0)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, score_vals)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "method": target_run.method})
        roc_target = output_dir / "exp2_roc_curve.csv"
        roc_df.to_csv(roc_target, index=False)
        exported["exp2_roc_curve"] = roc_target

        prec, rec, _ = precision_recall_curve(y_true, score_vals)
        pr_df = pd.DataFrame({"recall": rec, "precision": prec, "method": target_run.method})
        pr_target = output_dir / "exp2_pr_curve.csv"
        pr_df.to_csv(pr_target, index=False)
        exported["exp2_pr_curve"] = pr_target

    # 3. Confusion matrices before and after OSR
    cm_b = target_run.confusion_before
    if not cm_b.empty:
        p_b = output_dir / "exp2_confusion_before.csv"
        cm_b.to_csv(p_b)
        exported["exp2_confusion_before"] = p_b

    cm_a = target_run.confusion_after
    if not cm_a.empty:
        p_a = output_dir / "exp2_confusion_after.csv"
        cm_a.to_csv(p_a)
        exported["exp2_confusion_after"] = p_a

    return exported


def export_communication_series(runs: list[RunRecord], output_dir: Path) -> dict[str, Path]:
    """Export communication_alpha1.csv."""
    exported: dict[str, Path] = {}
    matching = [r for r in runs if not r.communication.empty]
    if matching:
        comm_df = matching[0].communication
        target_p = output_dir / "communication_alpha1.csv"
        export_communication_contract(comm_df, matching[0].method, target_p)
        exported["communication_alpha1"] = target_p
    return exported


def export_client_distribution(runs: list[RunRecord], output_dir: Path) -> dict[str, Path]:
    """Export client_distribution_alpha01.csv."""
    exported: dict[str, Path] = {}
    for r in runs:
        dist_p = r.run_dir / "processed" / "client_class_distribution.csv"
        if not dist_p.exists():
            dist_p = r.run_dir / "client_class_distribution.csv"
        if dist_p.exists():
            df = pd.read_csv(dist_p)
            target_p = output_dir / "client_distribution_alpha01.csv"
            df.to_csv(target_p, index=False)
            exported["client_distribution_alpha01"] = target_p
            break
    return exported


def build_adapted_result_json(runs: list[RunRecord], output_path: Path) -> dict[str, Any]:
    """Assemble result.json metadata file for the plots project."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fedtros_runs = [r for r in runs if "fedtros" in r.method.lower()]
    base_run = fedtros_runs[0] if fedtros_runs else (runs[0] if runs else None)

    result_data: dict[str, Any] = {
        "schema_version": "2.0",
        "units": {
            "classification_metrics": "percent",
            "communication": "MB",
        },
        "metadata": {
            "method_display_name": "FedTROS-PR",
            "dataset": base_run.dataset if base_run else "B-NAT",
            "seed": base_run.seed if base_run else 42,
            "evidence_policy": "Authoritative measured outputs from FedTROS canonical runs.",
            "source_runs": [r.run_id for r in runs],
        },
        "plots": {},
    }

    # Plot 01: Scalability summary
    scalability_runs = [r for r in runs if r.num_clients in (10, 50, 100)]
    if scalability_runs:
        s_rows = []
        for r in scalability_runs:
            s_rows.append(
                {
                    "clients": r.num_clients,
                    "overall_accuracy": r.get_metric(
                        ["round_openset_overall_acc", "overall_accuracy", "accuracy"], 0.0
                    )
                    * 100.0,
                    "known_accuracy": r.get_metric(["round_openset_known_acc", "known_acc"], 0.0)
                    * 100.0,
                    "unknown_f1": r.get_metric(["openset_unknown_f1", "unknown_f1"], 0.0) * 100.0,
                    "auroc": r.get_metric(["round_openset_auroc", "openset_auroc", "auroc"], 0.0)
                    * 100.0,
                }
            )
        result_data["plots"]["plot_01_scalability"] = {
            "source_status": "measured",
            "rows": sorted(s_rows, key=lambda x: x["clients"]),
        }

    output_path.write_text(json.dumps(result_data, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Saved adapted result.json to %s", output_path)
    return result_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export canonical FedTROS outputs to good-plot-project schemas."
    )
    parser.add_argument(
        "--study", type=str, default=None, help="Filter by study ID (e.g. E4-NIID-FOSR)"
    )
    parser.add_argument(
        "--stage", type=str, default=None, help="Filter by stage (e.g. paper_final, smoke)"
    )
    parser.add_argument(
        "--all-paper-studies", action="store_true", help="Export across all paper final studies"
    )
    parser.add_argument(
        "--runs-dir", type=str, default="outputs", help="Directory containing run outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_artifacts/plot_data",
        help="Target output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_filter = args.stage
    study_filter = args.study
    if args.all_paper_studies and not stage_filter:
        stage_filter = None  # query across all completed runs for paper

    logger.info(
        "Querying runs from %s (study=%s, stage=%s, all_paper=%s)...",
        args.runs_dir,
        study_filter,
        stage_filter,
        args.all_paper_studies,
    )
    runs = query_runs(
        study=study_filter, stage=stage_filter, outputs_dir=args.runs_dir, status=None
    )
    logger.info("Found %d matching runs.", len(runs))

    if not runs:
        logger.warning("No runs found matching query. Writing empty placeholder metadata.")
        build_adapted_result_json([], out_dir / "result.json")
        generate_provenance_manifest([], out_dir / "provenance_manifest.json")
        return

    # Export all schemas
    export_scalability_series(runs, out_dir)
    export_convergence_series(runs, out_dir)
    export_osr_curves_and_scores(runs, out_dir)
    export_communication_series(runs, out_dir)
    export_client_distribution(runs, out_dir)
    build_adapted_result_json(runs, out_dir / "result.json")
    generate_provenance_manifest(runs, out_dir / "provenance_manifest.json")

    logger.info("Successfully exported plot project artifacts to %s", out_dir)


if __name__ == "__main__":
    main()
