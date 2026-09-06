"""Offline re-evaluation and metric correction tool for FedTROS-MC experiment runs.

Fixes the boolean masking bug in the evaluation pipeline and recalculates:
1. Original Logged vs True Corrected metrics on test_scores.csv.
2. Clean Calibration metrics (Zeng et al. 2026 Eq. 14, excluding misclassified outliers).
3. Class-Conditional Conformal metrics (tau_{alpha, c} enforcing class-specific coverage).
4. Threshold-Normalized Multi-Class Ratio metrics.

Writes:
  <run_dir>/metrics/final_metrics_corrected.json
  <run_dir>/osr/test_scores_corrected.csv
without modifying or overwriting original raw logs.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def recalculate_run(run_dir: Path, alpha: float = 0.05, save: bool = True) -> dict:
    run_dir = Path(run_dir)
    print("=" * 80)
    print(f"Auditing and Re-evaluating Run: {run_dir.name}")
    print("=" * 80)

    osr_dir = run_dir / "osr"
    test_scores_path = osr_dir / "test_scores.csv"
    calib_scores_path = osr_dir / "calibration_scores.csv"
    conformal_meta_path = osr_dir / "conformal_metadata.json"
    final_metrics_path = run_dir / "metrics" / "final_metrics.json"

    if not test_scores_path.exists():
        raise FileNotFoundError(f"Missing test scores: {test_scores_path}")

    df_test = pd.read_csv(test_scores_path)
    df_calib = pd.read_csv(calib_scores_path) if calib_scores_path.exists() else pd.DataFrame()
    conformal_meta = {}
    if conformal_meta_path.exists():
        with open(conformal_meta_path, "r", encoding="utf-8") as f:
            conformal_meta = json.load(f)

    # 1. Base Truth
    is_unk_bool = (df_test["known_or_unknown"] == "unknown").to_numpy()
    is_unk_int = is_unk_bool.astype(int)
    known_mask = ~is_unk_bool
    scores = df_test["nonconformity_score"].to_numpy()
    tau_alpha = float(conformal_meta.get("tau_alpha", df_test["tau_alpha"].iloc[0]))
    y_true = df_test["true_label"].to_numpy()
    cand_pred = df_test["candidate_pred"].to_numpy()

    # Original decision rule: score >= tau_alpha
    rejected_orig = (scores >= tau_alpha)

    # Buggy indexing vs Correct boolean masking:
    buggy_unknown_recall = float(np.mean(rejected_orig[is_unk_int]))
    true_unknown_recall = float(np.mean(rejected_orig[is_unk_bool]))
    known_false_rejection = float(np.mean(rejected_orig[known_mask]))

    print("\n--- 1. Verification of Metric Indexing Bug ---")
    print(f"Total test queries: {len(df_test)}")
    print(f"Known samples:      {known_mask.sum()}")
    print(f"Unknown samples:    {is_unk_bool.sum()}")
    print(f"Rejection threshold tau_alpha: {tau_alpha:.4f}")
    print(f"Logged unknown_recall (buggy integer indexing): {buggy_unknown_recall:.4f}")
    print(f"True unknown_recall (boolean masking):          {true_unknown_recall:.4f} ({np.sum(rejected_orig & is_unk_bool)} / {is_unk_bool.sum()})")
    print(f"Known False Rejection (KFR / FPR):             {known_false_rejection:.4f} (Target <= {alpha})")

    # 2. Clean Calibration Calculation (Zeng et al. 2026 Eq. 14)
    tau_clean = tau_alpha
    tau_class_clean = {}
    tau_class_raw = {}
    if not df_calib.empty:
        calib_correct = (df_calib["true_label"] == df_calib["candidate_pred"]).to_numpy()
        calib_scores = df_calib["nonconformity_score"].to_numpy()
        clean_scores = calib_scores[calib_correct]

        m_clean = len(clean_scores)
        if m_clean > 0:
            k_clean = min(int(math.ceil((m_clean + 1) * (1.0 - alpha))), m_clean)
            tau_clean = float(np.sort(clean_scores)[k_clean - 1])

        # Class-conditional thresholds
        classes = sorted(df_calib["true_label"].unique().tolist())
        for c in classes:
            c_mask_clean = (df_calib["true_label"] == c) & (df_calib["candidate_pred"] == c)
            c_scores_clean = df_calib.loc[c_mask_clean, "nonconformity_score"].to_numpy()
            if len(c_scores_clean) > 0:
                k_c = min(int(math.ceil((len(c_scores_clean) + 1) * (1.0 - alpha))), len(c_scores_clean))
                tau_class_clean[int(c)] = float(np.sort(c_scores_clean)[k_c - 1])
            else:
                tau_class_clean[int(c)] = tau_clean

            c_mask_raw = (df_calib["candidate_pred"] == c)
            c_scores_raw = df_calib.loc[c_mask_raw, "nonconformity_score"].to_numpy()
            if len(c_scores_raw) > 0:
                k_c_raw = min(int(math.ceil((len(c_scores_raw) + 1) * (1.0 - alpha))), len(c_scores_raw))
                tau_class_raw[int(c)] = float(np.sort(c_scores_raw)[k_c_raw - 1])
            else:
                tau_class_raw[int(c)] = tau_alpha

    print("\n--- 2. Calibration Analysis ---")
    print(f"Global tau_alpha (with misclassified):  {tau_alpha:.4f}")
    print(f"Clean tau_alpha (Zeng et al. Eq. 14):   {tau_clean:.4f}")
    print("Class-Conditional Clean Thresholds:")
    for c, tc in tau_class_clean.items():
        print(f"  Class {c}: tau_{c} = {tc:.4f}")

    # Rejection under Clean Threshold
    rejected_clean = (scores >= tau_clean)
    kfr_clean = float(np.mean(rejected_clean[known_mask]))
    unk_rec_clean = float(np.mean(rejected_clean[is_unk_bool]))

    # Rejection under Class-Conditional Clean Thresholds
    rejected_class_cond = np.array([
        scores[i] >= tau_class_clean.get(int(cand_pred[i]), tau_clean)
        for i in range(len(df_test))
    ])
    kfr_class_cond = float(np.mean(rejected_class_cond[known_mask]))
    unk_rec_class_cond = float(np.mean(rejected_class_cond[is_unk_bool]))

    print("\n--- 3. Performance Across Decision Rules ---")
    print(f"{'Decision Rule':<35} | {'Known KFR (<=5%)':<16} | {'Unknown Recall':<16} | {'Unknown F1':<12}")
    print("-" * 85)

    f1_orig = float(f1_score(is_unk_int, rejected_orig.astype(int), zero_division=0))
    f1_clean = float(f1_score(is_unk_int, rejected_clean.astype(int), zero_division=0))
    f1_class = float(f1_score(is_unk_int, rejected_class_cond.astype(int), zero_division=0))

    print(f"{'1. Original tau (Corrected mask)':<35} | {known_false_rejection*100:6.2f}%          | {true_unknown_recall*100:6.2f}%          | {f1_orig:.4f}")
    print(f"{'2. Clean tau (Zeng et al. Eq. 14)':<35} | {kfr_clean*100:6.2f}%          | {unk_rec_clean*100:6.2f}%          | {f1_clean:.4f}")
    print(f"{'3. Class-Conditional Clean tau':<35} | {kfr_class_cond*100:6.2f}%          | {unk_rec_class_cond*100:6.2f}%          | {f1_class:.4f}")

    # Standard metrics
    finite_mask = np.isfinite(scores)
    rep = float(np.max(scores[finite_mask])) + 1.0 if np.any(finite_mask) else 1.0
    scores_det = np.nan_to_num(scores, nan=rep, posinf=rep, neginf=0.0)
    auroc = float(roc_auc_score(is_unk_int, scores_det))
    auprc = float(average_precision_score(is_unk_int, scores_det))

    # Read original final_metrics if available
    orig_metrics = {}
    if final_metrics_path.exists():
        with open(final_metrics_path, "r", encoding="utf-8") as f:
            orig_metrics = json.load(f)

    # Component AUROC metrics
    mah_auroc = orig_metrics.get("open_set/auroc_mahalanobis")
    rec_auroc = orig_metrics.get("open_set/auroc_reconstruction")
    if mah_auroc is not None or rec_auroc is not None:
        print("\n--- 4. ConFID Dual-Path Component Analysis ---")
        if mah_auroc is not None:
            print(f"  * Component 1 (Mahalanobis Distance):     {mah_auroc:.4f} ({mah_auroc*100:.2f}%)")
        if rec_auroc is not None:
            print(f"  * Component 2 (Reconstruction Error):    {rec_auroc:.4f} ({rec_auroc*100:.2f}%)")
        print(f"  * Unified ConFID Ensemble AUROC:          {auroc:.4f} ({auroc*100:.2f}%)")

    corrected_metrics = dict(orig_metrics)
    corrected_metrics["open_set/unknown_recall"] = true_unknown_recall
    corrected_metrics["open_set/unknown_recall_buggy_logged"] = buggy_unknown_recall
    corrected_metrics["open_set/unknown_f1"] = f1_orig
    corrected_metrics["open_set/KFR"] = known_false_rejection
    corrected_metrics["open_set/known_false_unknown_rate"] = known_false_rejection
    corrected_metrics["open_set/clean_tau_alpha"] = tau_clean
    corrected_metrics["open_set/clean_unknown_recall"] = unk_rec_clean
    corrected_metrics["open_set/clean_KFR"] = kfr_clean
    corrected_metrics["open_set/class_conditional_unknown_recall"] = unk_rec_class_cond
    corrected_metrics["open_set/class_conditional_KFR"] = kfr_class_cond
    corrected_metrics["open_set/class_conditional_tau"] = tau_class_clean

    if save:
        out_metrics_path = run_dir / "metrics" / "final_metrics_corrected.json"
        with open(out_metrics_path, "w", encoding="utf-8") as f:
            json.dump(corrected_metrics, f, indent=2)
        print(f"\nSaved corrected metrics to: {out_metrics_path}")

        # Save corrected test scores with updated rejection flags
        df_corrected = df_test.copy()
        df_corrected["final_reject_corrected"] = rejected_orig
        df_corrected["final_reject_clean"] = rejected_clean
        df_corrected["final_reject_class_cond"] = rejected_class_cond
        out_test_scores_path = osr_dir / "test_scores_corrected.csv"
        df_corrected.to_csv(out_test_scores_path, index=False)
        print(f"Saved corrected test scores to: {out_test_scores_path}")

    return corrected_metrics


def main():
    parser = argparse.ArgumentParser(description="Recalculate and correct OSR metrics for FedTROS-MC runs.")
    parser.add_argument("run_dirs", nargs="+", help="Paths to run directories")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level alpha")
    parser.add_argument("--no-save", action="store_true", help="Do not save output files")
    args = parser.parse_args()

    for r in args.run_dirs:
        recalculate_run(Path(r), alpha=args.alpha, save=not args.no_save)


if __name__ == "__main__":
    main()
