import logging
import hashlib
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from src.openset.conformal import fit_multicenter_conformal, score_multicenter_conformal
from src.openset.prototype_rank_pipeline import _collect_student_scores, _nested, UNKNOWN_LABEL_ID
import json
import hashlib

logger = logging.getLogger("MulticenterConformal")

def calibrate_multicenter_conformal(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    cfg: Any,
    output_dir: Path | None = None,
    logger_: logging.Logger | None = None,
) -> tuple[dict, pd.DataFrame, dict[str, Any]]:
    log = logger_ or logger
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    unknown_label_id = int(_nested(cfg, "unknown_label_id", UNKNOWN_LABEL_ID))
    if np.any(labels_np == unknown_label_id):
        raise ValueError("Multicenter Conformal calibration data must contain known classes only.")

    df = _collect_student_scores(
        features,
        labels,
        student_model=student_model,
        batch_size=batch_size,
        device=device,
        cfg=cfg,
        class_condition="true",
    )
    num_classes = int(student_model.num_classes)

    proto_fit_fraction = float(_nested(cfg, "calibration.prototype_fit_fraction", 0.70))
    threshold_calib_fraction = float(_nested(cfg, "calibration.threshold_calibration_fraction", 0.30))
    split_seed = int(_nested(cfg, "prototype.seed", 42))

    if not np.isclose(proto_fit_fraction + threshold_calib_fraction, 1.0, atol=1e-6):
        raise ValueError(
            "calibration prototype_fit_fraction + threshold_calibration_fraction must equal 1.0."
        )

    if 0.0 < proto_fit_fraction < 1.0 and len(df) >= max(2 * num_classes, 20):
        try:
            proto_indices, calib_indices = train_test_split(
                np.arange(len(df)),
                test_size=threshold_calib_fraction,
                stratify=df["y_raw"].to_numpy(),
                random_state=split_seed,
            )
        except Exception as exc:
            log.warning("Stratified calibration split failed (%s); falling back to seeded random split.", exc)
            proto_indices, calib_indices = train_test_split(
                np.arange(len(df)),
                test_size=threshold_calib_fraction,
                random_state=split_seed,
            )
        if set(proto_indices.tolist()).intersection(calib_indices.tolist()):
            raise AssertionError("Prototype-fit and threshold-calibration indices overlap.")
        df_proto = df.iloc[proto_indices].copy()
        df_calib = df.iloc[calib_indices].copy()
        
        split_provenance = {
            "disjoint_split": True,
            "prototype_fit_samples": int(len(df_proto)),
            "threshold_calibration_samples": int(len(df_calib)),
            "prototype_fit_fraction": float(proto_fit_fraction),
            "threshold_calibration_fraction": float(threshold_calib_fraction),
            "split_seed": split_seed,
            "proto_indices_hash": hashlib.sha256(str(sorted(proto_indices.tolist())).encode()).hexdigest(),
            "calib_indices_hash": hashlib.sha256(str(sorted(calib_indices.tolist())).encode()).hexdigest(),
        }
    else:
        if bool(_nested(cfg, "calibration.strict_disjoint", True)):
            raise ValueError(
                "Known-only calibration pool is too small for disjoint split. Set calibration.strict_disjoint=false for smoke."
            )
        df_proto = df.copy()
        df_calib = df.copy()
        split_provenance = {
            "disjoint_split": False,
            "total_samples": int(len(df)),
            "note": "Smoke fallback only: fitting and calibration reuse the same pool.",
        }

    alpha = float(_nested(cfg, "alpha", 0.05))
    conformal_meta = fit_multicenter_conformal(df_proto, df_calib, num_classes, alpha=alpha, seed=split_seed, output_dir=output_dir)
    
    # Clean up pca object from conformal_meta before serialization
    conformal_meta_serializable = {k: v for k, v in conformal_meta.items() if k != 'pca'}
    
    meta = {
        "split": split_provenance,
        "conformal": conformal_meta_serializable,
    }
    
    if output_dir:
        osr_dir = output_dir / "osr"
        osr_dir.mkdir(parents=True, exist_ok=True)
        try:
            from omegaconf import OmegaConf
            cfg_hash = hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()
        except Exception:
            cfg_hash = ""
        conf_meta = {
            "method": "global_split_conformal",
            "alpha": conformal_meta["alpha"],
            "calibration_size": conformal_meta["m"],
            "k_alpha": conformal_meta["k_alpha"],
            "tau_alpha": conformal_meta["tau_alpha"],
            "score": "candidate_class_squared_mahalanobis",
            "feature_source": "l2_normalized_student_hidden",
            "calibration_scope": "global_known",
            "split_hash": split_provenance.get("proto_indices_hash", ""),
            "config_hash": cfg_hash,
            "unknown_data_used_for_fitting": False,
            "misclassified_known_calibration_included": True,
        }
        (osr_dir / "conformal_metadata.json").write_text(json.dumps(conf_meta, indent=2), encoding="utf-8")
        
        
        # EXPORT CANONICAL FILES
        
        # Prototype centers, precision matrices, summaries, assignments, and
        # calibration scores are emitted by fit_multicenter_conformal.  Do not
        # rewrite them here: doing so would erase the populated audit artifacts.

        # Save split manifest
        split_manifest = {
            "schema_version": 2,
            "train_known": {"count": None, "hash": "", "status": "not supplied to calibration API"},
            "D_proto": {"count": len(df_proto), "hash": split_provenance.get("proto_indices_hash", "")},
            "D_cal": {"count": len(df_calib), "hash": split_provenance.get("calib_indices_hash", "")},
            "final_known_test": {"count": None, "hash": "", "status": "updated after final evaluation"},
            "final_unknown_test": {"count": None, "hash": "", "status": "updated after final evaluation"},
            "pairwise_disjoint": split_provenance["disjoint_split"],
            "split_seed": split_seed,
            "unknown_labels": None,
            "unknown_data_used_for_fitting": False,
            "proof": "Prototype selection/covariance use D_proto known only; calibration uses D_cal known only.",
        }
        (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
        
    return conformal_meta, df_calib, meta

def evaluate_multicenter_conformal(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    student_model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    cfg: Any,
    class_names: dict[int, str],
    output_dir: Path | None = None,
    conformal_meta: dict,
    logger_: logging.Logger | None = None,
    report_to_stdout: bool = False,
    meta: dict[str, Any] | None = None,
) -> dict[str, float]:
    log = logger_ or logger
    df_eval = _collect_student_scores(
        features,
        labels,
        student_model=student_model,
        batch_size=batch_size,
        device=device,
        cfg=cfg,
        class_condition="pred",
    )
    unknown_label_id = int(_nested(cfg, "unknown_label_id", UNKNOWN_LABEL_ID))
    
    df_eval = score_multicenter_conformal(df_eval, conformal_meta)
    
    y_true = df_eval["y_raw"].to_numpy()
    is_unknown = (y_true == unknown_label_id).astype(int)
    scores = df_eval["conformal_score"].to_numpy()
    
    # Missing candidate-class models are maximally nonconforming, not missing
    # observations.  Preserve them in binary detection metrics instead of
    # silently dropping those samples.
    finite = scores[np.isfinite(scores)]
    replacement = (float(np.max(finite)) + 1.0) if finite.size else 1.0
    scores_for_detection = np.nan_to_num(scores, nan=replacement, posinf=replacement, neginf=0.0)
    valid_scores_mask = np.ones_like(is_unknown, dtype=bool)
    
    metrics = {}
    if np.sum(is_unknown) > 0 and np.sum(1 - is_unknown) > 0:
        metrics["open_set/auroc"] = float(roc_auc_score(is_unknown, scores_for_detection))
        metrics["open_set/auprc"] = float(average_precision_score(is_unknown, scores_for_detection))
        fpr, tpr, thresholds = roc_curve(is_unknown, scores_for_detection)
        
        # roc.csv
        if output_dir:
            osr_dir = output_dir / "osr"
            osr_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(osr_dir / "roc.csv", index=False)
            
            p, r, t = precision_recall_curve(is_unknown[valid_scores_mask], scores[valid_scores_mask])
            t = np.append(t, np.nan)
            pd.DataFrame({"precision": p, "recall": r, "threshold": t}).to_csv(osr_dir / "pr.csv", index=False)
            
        idx = np.where(tpr >= 0.95)[0][0]
        metrics["open_set/fpr95"] = float(fpr[idx])
    else:
        metrics["open_set/auroc"] = 0.5
        metrics["open_set/auprc"] = 0.0
        metrics["open_set/fpr95"] = 1.0

    rejected = df_eval["rejected"].to_numpy().astype(bool)
    
    y_pred_before = df_eval["pred_before_osr"].to_numpy()
    y_pred_after = y_pred_before.copy()
    y_pred_after[rejected] = unknown_label_id
    
    known_mask = (y_true != unknown_label_id)
    if np.any(known_mask):
        metrics["open_set/known_accuracy_before"] = float(accuracy_score(y_true[known_mask], y_pred_before[known_mask]))
        metrics["open_set/known_accuracy_after"] = float(accuracy_score(y_true[known_mask], y_pred_after[known_mask]))
        metrics["open_set/KFR"] = float(np.mean(rejected[known_mask]))
    else:
        metrics["open_set/known_accuracy_before"] = 0.0
        metrics["open_set/known_accuracy_after"] = 0.0
        metrics["open_set/KFR"] = 0.0
        
    if np.any(is_unknown):
        metrics["open_set/unknown_recall"] = float(np.mean(rejected[is_unknown]))
    else:
        metrics["open_set/unknown_recall"] = 0.0
        
    metrics["open_set/unknown_f1"] = float(f1_score(
        is_unknown, rejected.astype(int), labels=[1], average="binary", zero_division=0
    )) if np.any(is_unknown) else 0.0
    metrics["open_set/macro_f1"] = float(f1_score(
        y_true, y_pred_after, average="macro", zero_division=0
    )) if len(y_true) else 0.0
    metrics["open_set/requested_alpha"] = float(conformal_meta.get("alpha", 0.05))

    if output_dir is not None:
        osr_dir = output_dir / "osr"
        osr_dir.mkdir(parents=True, exist_ok=True)
        
        # Test scores
        test_scores_df = pd.DataFrame({
            "sample_id": df_eval.get("sample_id", df_eval.index).to_numpy(),
            "true_label": y_true,
            "candidate_pred": y_pred_before,
            "final_pred": y_pred_after,
            "known_or_unknown": np.where(is_unknown, "unknown", "known"),
            "nonconformity_score": scores,
            "tau_alpha": conformal_meta.get("tau_alpha", float('inf')),
            "final_reject": rejected,
            "nearest_prototype_id": df_eval["nearest_prototype_id"],
            "nearest_prototype_class": df_eval["nearest_prototype_class"],
            "candidate_correct": (y_pred_before == y_true) & ~is_unknown
        })
        test_scores_df.to_csv(osr_dir / "test_scores.csv", index=False)

        # Complete the split manifest with final evaluation sample counts and
        # deterministic sample-id hashes once the frozen test set is available.
        split_manifest_path = output_dir / "split_manifest.json"
        if split_manifest_path.exists():
            try:
                split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
                known_ids = test_scores_df.loc[~is_unknown, "sample_id"].astype(str).tolist()
                unknown_ids = test_scores_df.loc[is_unknown, "sample_id"].astype(str).tolist()
                split_manifest["final_known_test"] = {
                    "count": len(known_ids),
                    "hash": hashlib.sha256("|".join(sorted(known_ids)).encode()).hexdigest(),
                }
                split_manifest["final_unknown_test"] = {
                    "count": len(unknown_ids),
                    "hash": hashlib.sha256("|".join(sorted(unknown_ids)).encode()).hexdigest(),
                }
                split_manifest["unknown_labels"] = [int(unknown_label_id)] if unknown_ids else []
                split_manifest["final_test_sample_ids_recorded"] = True
                split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
            except Exception as exc:
                log.warning("Could not update final split manifest: %s", exc)
        
        # Latent projection
        pca = conformal_meta.get("pca")
        if pca is not None:
            from src.openset.conformal import normalize_features
            z_eval = normalize_features(np.stack(df_eval["feature"].to_numpy()))
            proj = pca.transform(z_eval)
            
            proj_df = pd.DataFrame({
                "x": proj[:, 0],
                "y": proj[:, 1],
                "point_type": "sample",
                "split": "test",
                "true_label": y_true,
                "candidate_pred": y_pred_before,
                "is_unknown": is_unknown,
                "final_reject": rejected,
                "sample_id": test_scores_df["sample_id"],
                "prototype_class": -1,
                "prototype_id": -1,
                "selected_k": -1
            })
            
            # Also append prototypes
            proto_recs = []
            for c, m_data in conformal_meta["models"].items():
                centers = np.array(m_data["centers"])
                k_val = m_data["k"]
                p_proj = pca.transform(centers)
                for pid, p in enumerate(p_proj):
                    proto_recs.append({
                        "x": p[0],
                        "y": p[1],
                        "point_type": "prototype",
                        "split": "D_proto",
                        "true_label": c,
                        "candidate_pred": c,
                        "is_unknown": 0,
                        "final_reject": False,
                        "sample_id": -1,
                        "prototype_class": c,
                        "prototype_id": pid,
                        "selected_k": k_val
                    })
            if proto_recs:
                proj_df = pd.concat([proj_df, pd.DataFrame(proto_recs)], ignore_index=True)
                
            proj_df.to_csv(osr_dir / "latent_projection.csv", index=False)
            
            proj_meta = {
                "feature_source": "l2_normalized_student_hidden",
                "projection": "pca",
                "projection_fit_split": "D_proto_known_only",
                "prototype_method": "adaptive_classwise_kmeans",
                "boundary_prototypes": 0
            }
            (osr_dir / "latent_projection_metadata.json").write_text(json.dumps(proj_meta, indent=2), encoding="utf-8")
        
    return metrics
