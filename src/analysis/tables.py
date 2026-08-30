"""Paper table generation for FedTROS Q1 publication (Item C9).

Generates machine-readable CSVs and publication-ready LaTeX tables for:
  - E1: IID Utility Benchmark
  - E3: Non-IID Closed-Set Robustness
  - E4: Non-IID Open-Set Detection
  - E5: Cross-Dataset Generalization
  - E8: Leave-One-Attack-Out Robustness
  - Ablation Study Matrix
  - Scalability Summary (10, 50, 100 Clients)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.analysis.aggregation import AggregatedGroup, aggregate_runs, compute_paired_deltas
from src.analysis.loaders import RunRecord

logger = logging.getLogger(__name__)


def _metric(agg: AggregatedGroup, *names: str):
    for name in names:
        value = agg.metrics.get(name)
        if value is not None:
            return value
    return None


def _study_runs(runs: Sequence[RunRecord], study: str) -> list[RunRecord]:
    return [r for r in runs if r.study.upper() == study.upper()]


def _format_cell(mean: float, std: float, percent: bool = True, bold: bool = False) -> str:
    """Format numerical cell for LaTeX tables."""
    scale = 100.0 if percent else 1.0
    m = mean * scale
    s = std * scale
    text = f"{m:.2f} \\pm {s:.2f}"
    return f"\\textbf{{{text}}}" if bold else text


def build_e1_iid_table(runs: Sequence[RunRecord]) -> pd.DataFrame:
    """Generate E1 IID Utility Benchmark table."""
    runs = [r for r in _study_runs(runs, "E1-IID-CS") if r.dataset.upper() == "B-NAT"]
    methods = ["Centralized", "FedAvg-Student", "FedProx-Student", "FedTROS-MC"]
    rows = []

    for m in methods:
        m_runs = [
            r
            for r in runs
            if r.method.lower() == m.lower()
            or (m == "Centralized" and "central" in r.method.lower())
        ]
        if not m_runs:
            continue
        agg = aggregate_runs(m_runs, validate=False)
        acc = _metric(agg, "closed_set/accuracy", "overall_accuracy", "test_acc", "accuracy")
        f1 = _metric(agg, "closed_set/macro_f1", "macro_f1", "local_student_f1_macro")

        rows.append(
            {
                "Method": m,
                "Seeds": len(agg.seeds),
                "Accuracy (%)": acc.format_mean_std(percent=acc.mean <= 1.0) if acc else "N/A",
                "Macro-F1 (%)": f1.format_mean_std(percent=f1.mean <= 1.0) if f1 else "N/A",
                "raw_acc_mean": acc.mean if acc else np.nan,
                "raw_acc_std": acc.std_across_seeds if acc else np.nan,
                "raw_f1_mean": f1.mean if f1 else np.nan,
                "raw_f1_std": f1.std_across_seeds if f1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_e3_non_iid_table(runs: Sequence[RunRecord]) -> pd.DataFrame:
    """Generate E3 Non-IID Closed-Set Robustness table across alpha in {0.1, 0.5, 1.0}."""
    alphas = [1.0, 0.5, 0.1]
    runs = [r for r in _study_runs(runs, "E3-NIID-CS") if r.dataset.upper() == "B-NAT"]
    methods = ["FedAvg-Student", "FedProx-Student", "FedTROS-MC"]
    rows = []

    for alpha in alphas:
        for m in methods:
            m_runs = [
                r for r in runs if abs(r.alpha - alpha) < 1e-4 and r.method.lower() == m.lower()
            ]
            if not m_runs:
                continue
            agg = aggregate_runs(m_runs, validate=False)
            acc = _metric(
                agg, "closed_set/accuracy", "local_student_accuracy", "overall_accuracy", "accuracy"
            )
            f1 = _metric(agg, "closed_set/macro_f1", "local_student_f1_macro", "macro_f1")
            worst_f1 = _metric(agg, "worst_client_macro_f1")

            rows.append(
                {
                    "Alpha": alpha,
                    "Method": m,
                    "Seeds": len(agg.seeds),
                    "Mean Acc (%)": acc.format_mean_std(percent=acc.mean <= 1.0) if acc else "N/A",
                    "Macro-F1 (%)": f1.format_mean_std(percent=f1.mean <= 1.0) if f1 else "N/A",
                    "Worst-Client F1 (%)": worst_f1.format_mean_std(percent=worst_f1.mean <= 1.0)
                    if worst_f1
                    else "N/A",
                    "raw_f1_mean": f1.mean if f1 else np.nan,
                    "raw_f1_std": f1.std_across_seeds if f1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_e4_open_set_table(runs: Sequence[RunRecord]) -> pd.DataFrame:
    """Generate E4 Non-IID Open-Set Detection table."""
    runs = [r for r in _study_runs(runs, "E4-NIID-FOSR") if r.dataset.upper() == "B-NAT"]
    methods = ["FedAvg-Student", "FedProx-Student", "FedTROS-MC"]
    rows = []

    for m in methods:
        m_runs = [r for r in runs if r.method.lower() == m.lower()]
        if not m_runs:
            continue
        agg = aggregate_runs(m_runs, validate=False)
        f1 = _metric(
            agg, "open_set/macro_f1", "openset_f1_macro", "round_openset_f1_macro", "macro_f1"
        )
        auroc = _metric(agg, "open_set/auroc", "openset_auroc", "round_openset_auroc")
        auprc = _metric(agg, "open_set/auprc", "openset_auprc")
        fpr95 = _metric(agg, "open_set/fpr95", "openset_fpr95", "round_openset_fpr95")
        unkn_rec = _metric(
            agg, "open_set/unknown_recall", "openset_unknown_recall", "round_openset_unknown_recall"
        )
        k_fur = _metric(
            agg, "open_set/KFR", "open_set/known_false_unknown_rate", "openset_known_false_unknown_rate", "openset_KFR"
        )

        rows.append(
            {
                "Method": m,
                "Seeds": len(agg.seeds),
                "Open-Set Macro-F1 (%)": f1.format_mean_std(percent=f1.mean <= 1.0)
                if f1
                else "N/A",
                "AUROC (%)": auroc.format_mean_std(percent=auroc.mean <= 1.0) if auroc else "N/A",
                "AUPRC (%)": auprc.format_mean_std(percent=auprc.mean <= 1.0) if auprc else "N/A",
                "FPR@95 (%)": fpr95.format_mean_std(percent=fpr95.mean <= 1.0) if fpr95 else "N/A",
                "Unknown Recall (%)": unkn_rec.format_mean_std(percent=unkn_rec.mean <= 1.0)
                if unkn_rec
                else "N/A",
                "K-FUR (%)": k_fur.format_mean_std(percent=k_fur.mean <= 1.0) if k_fur else "N/A",
            }
        )
    return pd.DataFrame(rows)


def build_e5_multidataset_table(runs: Sequence[RunRecord]) -> pd.DataFrame:
    """Generate E5 Multi-Dataset Generalization table (B-NAT, ToN-IoT, CIC-IDS2017)."""
    runs = _study_runs(runs, "E5-DATASET")
    datasets = ["B-NAT", "B-TAT", "ToN-IoT", "CIC-IDS2017"]
    methods = ["FedAvg-Student", "FedProx-Student", "FedTROS-MC"]
    rows = []

    for d in datasets:
        for m in methods:
            m_runs = [
                r for r in runs if r.dataset.lower() == d.lower() and r.method.lower() == m.lower()
            ]
            if not m_runs:
                continue
            agg = aggregate_runs(m_runs, validate=False)
            f1 = _metric(agg, "open_set/macro_f1", "openset_f1_macro", "macro_f1")
            auroc = _metric(agg, "open_set/auroc", "openset_auroc", "auroc")

            rows.append(
                {
                    "Dataset": d,
                    "Method": m,
                    "Seeds": len(agg.seeds),
                    "Open-Set Macro-F1 (%)": f1.format_mean_std(percent=f1.mean <= 1.0)
                    if f1
                    else "N/A",
                    "AUROC (%)": auroc.format_mean_std(percent=auroc.mean <= 1.0)
                    if auroc
                    else "N/A",
                }
            )
    return pd.DataFrame(rows)


def build_ablation_table(runs: Sequence[RunRecord]) -> pd.DataFrame:
    """Generate Ablation Study Matrix table."""
    ablation_configs = [
        ("Base Student Only (FedAvg)", "base_fedavg"),
        ("w/ Variational Teacher Distillation", "teacher_only"),
        ("w/ Disagreement Gating", "gating_only"),
        ("w/ Coverage Anchoring", "anchor_only"),
        ("w/ Prototype-Rank Module", "prototype_only"),
        ("FedTROS-MC (Full Method)", "fedtros_full"),
    ]
    rows = []
    for label, key in ablation_configs:
        m_runs = [r for r in runs if r.study.lower() == key.lower() or key in r.run_id.lower()]
        if not m_runs:
            continue
        agg = aggregate_runs(m_runs, validate=False)
        f1 = agg.metrics.get("openset_f1_macro") or agg.metrics.get("macro_f1")
        auroc = agg.metrics.get("openset_auroc") or agg.metrics.get("auroc")
        unkn_rec = agg.metrics.get("openset_unknown_recall")

        rows.append(
            {
                "Configuration": label,
                "Macro-F1 (%)": f1.format_mean_std(percent=f1.mean <= 1.0) if f1 else "N/A",
                "AUROC (%)": auroc.format_mean_std(percent=auroc.mean <= 1.0) if auroc else "N/A",
                "Unknown Recall (%)": unkn_rec.format_mean_std(percent=unkn_rec.mean <= 1.0)
                if unkn_rec
                else "N/A",
            }
        )
    return pd.DataFrame(rows)


def export_all_paper_tables(runs: Sequence[RunRecord], output_dir: Path) -> dict[str, Path]:
    """Build and save all publication tables as machine-readable CSVs and LaTeX code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    table_builders = {
        "table_e1_iid_utility": build_e1_iid_table,
        "table_e3_non_iid_closed": build_e3_non_iid_table,
        "table_e4_open_set": build_e4_open_set_table,
        "table_e5_multidataset": build_e5_multidataset_table,
        "table_ablation_matrix": build_ablation_table,
    }

    for stem, builder in table_builders.items():
        df = builder(runs)
        if not df.empty:
            csv_path = output_dir / f"{stem}.csv"
            df.to_csv(csv_path, index=False)
            generated[f"{stem}_csv"] = csv_path

            # Export LaTeX fragment
            tex_path = output_dir / f"{stem}.tex"
            tex_code = df.to_latex(index=False, escape=False)
            tex_path.write_text(tex_code, encoding="utf-8")
            generated[f"{stem}_tex"] = tex_path

    logger.info("Generated %d table artifacts in %s", len(generated), output_dir)
    return generated
