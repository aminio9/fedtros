import os
import json
import numpy as np
import pandas as pd
import logging
import hashlib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

logger = logging.getLogger("Conformal")
EPS = 1e-8

def normalize_features(h: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    return h / np.maximum(norms, EPS)

def fit_multicenter_conformal(
    df_proto: pd.DataFrame, 
    df_calib: pd.DataFrame, 
    num_classes: int, 
    alpha: float = 0.05,
    seed: int = 42,
    output_dir: Path = None,
    clean_calibration: bool = False,
    score_mode: str = "candidate",
    ensemble_recon_weight: float = 0.5,
) -> dict:
    models = {}
    
    proto_sel_records = []
    proto_summary_records = []
    proto_assign_records = []
    proto_internal_records = []
    
    z_all_proto = []
    
    for c in range(num_classes):
        cls_proto = df_proto[df_proto["y_raw"] == c]
        if cls_proto.empty:
            continue
            
        z_c = normalize_features(np.stack(cls_proto["feature"].to_numpy()))
        z_all_proto.append(z_c)
        sample_ids = cls_proto.get("sample_id", cls_proto.index).to_numpy()
        
        best_k = 1
        best_score = -np.inf
        best_centers = z_c.mean(axis=0, keepdims=True)
        best_labels = np.zeros(len(z_c), dtype=int)
        
        # Select K on a temporary, deterministic validation split inside D_proto,
        # then refit the selected model on all D_proto samples.  This prevents the
        # silhouette score from being evaluated on the same samples used to fit KMeans.
        model_selection_fraction = 0.80
        rng = np.random.default_rng(seed + int(c))
        perm = rng.permutation(len(z_c))
        n_fit = len(z_c) if len(z_c) < 4 else max(2, min(len(z_c) - 1, int(round(model_selection_fraction * len(z_c)))))
        fit_idx, val_idx = perm[:n_fit], perm[n_fit:]
        z_fit = z_c[fit_idx]
        z_val = z_c[val_idx]
        for idx in fit_idx:
            proto_internal_records.append({
                "sample_id": sample_ids[idx], "class_id": c, "subset": "proto-fit"
            })
        for idx in val_idx:
            proto_internal_records.append({
                "sample_id": sample_ids[idx], "class_id": c, "subset": "proto-val"
            })
        max_k = min(10, len(z_fit) - 1)
        proto_sel_records.append({"class_id": c, "class_name": f"Class {c}",
            "candidate_k": 1, "fold_id": 0, "surrogate_metric": np.nan,
            "metric_direction": "maximize", "selected": True, "selected_k": 1,
            "seed": seed, "n_fit": len(z_fit), "n_validation": len(z_val)})
        if max_k > 1 and len(z_val) >= 2:
            for k in range(2, max_k + 1):
                try:
                    km = KMeans(n_clusters=k, n_init=5, random_state=seed)
                    fit_labels = km.fit_predict(z_fit)
                    val_dist = ((z_val[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
                    val_labels = val_dist.argmin(axis=1)
                    if len(np.unique(val_labels)) > 1 and len(z_val) > len(np.unique(val_labels)):
                        score = silhouette_score(z_val, val_labels)
                        proto_sel_records.append({
                            "class_id": c, "class_name": f"Class {c}",
                            "candidate_k": k, "fold_id": 0,
                            "surrogate_metric": score, "metric_direction": "maximize",
                            "selected": False, "selected_k": -1, "seed": seed,
                            "n_fit": len(z_fit), "n_validation": len(z_val)
                        })
                        if score > best_score:
                            best_score = score
                            best_k = k
                except Exception:
                    continue

        # Final prototype fit uses every sample in D_proto with the selected K.
        if best_k > 1:
            final_km = KMeans(n_clusters=best_k, n_init=5, random_state=seed)
            best_labels = final_km.fit_predict(z_c)
            best_centers = final_km.cluster_centers_
        else:
            best_labels = np.zeros(len(z_c), dtype=int)
            best_centers = z_c.mean(axis=0, keepdims=True)

        for r in proto_sel_records:
            if r["class_id"] == c:
                r["selected_k"] = best_k
                r["selected"] = (r["candidate_k"] == best_k)

        residuals = z_c - best_centers[best_labels]
        models[c] = {
            "centers": best_centers.tolist(),
            "residuals": residuals,
            "precision": None,
            "k": best_k,
            "n_proto_samples": len(z_c),
        }
        for idx, s_id in enumerate(sample_ids):
            proto_assign_records.append({
                "sample_id": s_id,
                "true_class": c,
                "nearest_same_class_prototype_id": int(best_labels[idx]),
                "residual_norm": float(np.linalg.norm(residuals[idx]))
            })
        
    # Fit class-centred covariance estimates after all prototype residuals are
    # available.  Small classes use a pooled residual covariance, as required by
    # the canonical detector contract; identity is the last numerical fallback.
    pooled_residuals = [m["residuals"] for m in models.values() if len(m["residuals"]) > 0]
    pooled = np.concatenate(pooled_residuals, axis=0) if pooled_residuals else np.empty((0, 1))
    min_cov_samples = 8
    for c, model in models.items():
        residuals = model.pop("residuals")
        fallback = False
        covariance_scope = "class_center_shared"
        fit_residuals = residuals
        if len(residuals) < min_cov_samples and len(pooled) >= min_cov_samples:
            fit_residuals = pooled
            covariance_scope = "pooled_residual_fallback"
            fallback = True
        try:
            lw = LedoitWolf()
            lw.fit(fit_residuals)
            precision = lw.precision_
            cond = float(np.linalg.cond(lw.covariance_))
            shrinkage = float(lw.shrinkage_)
        except Exception as exc:
            logger.warning("LedoitWolf failed for class %s (%s); using identity precision.", c, exc)
            precision = np.eye(residuals.shape[1] if residuals.ndim == 2 else pooled.shape[1])
            cond = 1.0
            shrinkage = 0.0
            covariance_scope = "identity_fallback"
            fallback = True
        model["precision"] = precision.tolist()
        proto_summary_records.append({
            "class_id": c, "class_name": f"Class {c}",
            "n_proto_samples": model["n_proto_samples"], "selected_k": model["k"],
            "prototype_seed": seed, "covariance_estimator": "ledoit_wolf",
            "covariance_scope": covariance_scope, "shrinkage": shrinkage,
            "covariance_condition_number": cond,
            "covariance_fallback_used": fallback,
        })

    all_scores_mah = []
    calib_records = []
    
    if not df_calib.empty:
        calib_z = normalize_features(np.stack(df_calib["feature"].to_numpy()))
        calib_pred = df_calib.get("pred_before_osr", df_calib["y_raw"]).to_numpy()
        calib_true = df_calib["y_raw"].to_numpy()
        calib_ids = df_calib.get("sample_id", df_calib.index).to_numpy()
        
        for i, z in enumerate(calib_z):
            c_hat = calib_pred[i]
            if c_hat not in models:
                score = float('inf')
            else:
                centers = np.array(models[c_hat]["centers"])
                prec = np.array(models[c_hat]["precision"])
                diff = z[None, :] - centers
                dists = np.sum(np.dot(diff, prec) * diff, axis=1)
                score = float(np.min(dists))
            all_scores_mah.append(score)
            
            calib_records.append({
                "sample_id": calib_ids[i],
                "true_label": calib_true[i],
                "candidate_pred": c_hat,
                "score_mahalanobis": score,
            })

    # ConFID Dual-Score Ensemble Statistics (Farsimadan et al. 2026 Eq. 18-19)
    scores_mah_arr = np.array(all_scores_mah, dtype=np.float64)
    has_recon = ("recon_error" in df_calib.columns) and not df_calib["recon_error"].isna().all()
    if has_recon:
        scores_rec_arr = df_calib["recon_error"].to_numpy().astype(np.float64)
        finite_rec = np.isfinite(scores_rec_arr)
        if finite_rec.sum() > 0:
            mu_rec = float(np.mean(scores_rec_arr[finite_rec]))
            sigma_rec = float(np.std(scores_rec_arr[finite_rec]))
        else:
            has_recon = False
            mu_rec, sigma_rec = 0.0, 1.0
    else:
        scores_rec_arr = np.full(len(all_scores_mah), np.nan, dtype=np.float64)
        mu_rec, sigma_rec = 0.0, 1.0

    finite_mah = np.isfinite(scores_mah_arr)
    if finite_mah.sum() > 0:
        mu_mah = float(np.mean(scores_mah_arr[finite_mah]))
        sigma_mah = float(np.std(scores_mah_arr[finite_mah]))
    else:
        mu_mah, sigma_mah = 0.0, 1.0

    ensemble_stats = {
        "mu_mah": mu_mah,
        "sigma_mah": max(sigma_mah, 1e-8),
        "mu_rec": mu_rec,
        "sigma_rec": max(sigma_rec, 1e-8),
        "has_recon": bool(has_recon),
    }

    use_ensemble = str(score_mode).lower() in {"confid_ensemble", "ensemble"} and has_recon
    effective_scores = []
    for i in range(len(all_scores_mah)):
        s_m = scores_mah_arr[i]
        s_r = scores_rec_arr[i]
        if use_ensemble and np.isfinite(s_r) and np.isfinite(s_m):
            norm_m = (s_m - ensemble_stats["mu_mah"]) / ensemble_stats["sigma_mah"]
            norm_r = (s_r - ensemble_stats["mu_rec"]) / ensemble_stats["sigma_rec"]
            s_eff = float((1.0 - ensemble_recon_weight) * norm_m + ensemble_recon_weight * norm_r)
        elif use_ensemble and np.isfinite(s_m):
            norm_m = (s_m - ensemble_stats["mu_mah"]) / ensemble_stats["sigma_mah"]
            s_eff = float(norm_m)
        else:
            s_eff = s_m
        effective_scores.append(s_eff)
        calib_records[i]["nonconformity_score"] = s_eff
        calib_records[i]["score_reconstruction"] = float(s_r) if np.isfinite(s_r) else np.nan

    m = len(effective_scores)
    sorted_scores = np.sort(effective_scores)
    
    k_alpha = int(np.ceil((m + 1) * (1.0 - alpha)))
    if 1 <= k_alpha <= m:
        tau_alpha_global = float(sorted_scores[k_alpha - 1])
    else:
        tau_alpha_global = float('inf')

    # Clean calibration quantile (Zeng et al. 2026 Eq. 14): filter misclassified known outliers
    clean_scores = [
        r["nonconformity_score"] for r in calib_records
        if r["true_label"] == r["candidate_pred"]
    ]
    if clean_scores:
        m_clean = len(clean_scores)
        k_clean = min(int(np.ceil((m_clean + 1) * (1.0 - alpha))), m_clean)
        tau_alpha_clean = float(np.sort(clean_scores)[k_clean - 1])
    else:
        tau_alpha_clean = tau_alpha_global

    tau_alpha = tau_alpha_clean if clean_calibration else tau_alpha_global
        
    logger.info(
        "=== Multicenter Conformal Calibration (Algorithm 2 & Eq. 196) ===\n"
        "  * Prototype Banks: %s\n"
        "  * Calibration Samples: |D_cal| = %d\n"
        "  * Scoring Mode: %s (Ensemble Active: %s, Recon Weight: %.2f)\n"
        "  * Significance Level alpha = %.4f (Confidence = %.2f%%)\n"
        "  * Quantile Rank k_alpha = ceil((%d + 1) * %.4f) = %d\n"
        "  * Rejection Threshold tau_alpha = %.4f",
        ", ".join(f"Class {c}: K*={m_d['k']}, N={m_d['n_proto_samples']}" for c, m_d in models.items()),
        m, score_mode, use_ensemble, ensemble_recon_weight,
        alpha, (1.0 - alpha) * 100.0,
        m, 1.0 - alpha, k_alpha, tau_alpha
    )
    
    for r in calib_records:
        r["alpha"] = alpha
        r["calibration_size"] = m
        r["k_alpha"] = k_alpha
        r["tau_alpha"] = tau_alpha
        
    # Fit PCA for latent_projection
    pca = None
    if len(z_all_proto) > 0:
        z_all_proto_cat = np.concatenate(z_all_proto, axis=0)
        pca = PCA(n_components=2, random_state=seed)
        pca.fit(z_all_proto_cat)
        
    if output_dir:
        osr_dir = output_dir / "osr"
        osr_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(proto_sel_records).to_csv(osr_dir / "prototype_selection.csv", index=False)
        internal_frame = pd.DataFrame(proto_internal_records)
        internal_frame.to_csv(osr_dir / "prototype_internal_split.csv", index=False)
        pd.DataFrame(proto_summary_records).to_csv(osr_dir / "prototype_summary.csv", index=False)
        pd.DataFrame(proto_assign_records).to_csv(osr_dir / "prototype_assignments.csv", index=False)
        pd.DataFrame(calib_records).to_csv(osr_dir / "calibration_scores.csv", index=False)
        
        # Lossless scientific artifacts: centers and one precision matrix per class.
        c_dict = {"class_ids": np.asarray(sorted(models), dtype=np.int64)}
        for c, m_data in models.items():
            c_dict[f"class_{c}_centers"] = np.array(m_data["centers"])
            c_dict[f"class_{c}_precision"] = np.array(m_data["precision"])
        np.savez(osr_dir / "prototype_centers.npz", **c_dict)
        np.savez(osr_dir / "covariance_precision.npz", **{
            f"class_{c}": np.asarray(m_data["precision"], dtype=np.float64)
            for c, m_data in models.items()})
        # Canonical contract name; keep the explicit precision alias for readers.
        np.savez(osr_dir / "covariance.npz", **{
            f"class_{c}": np.asarray(m_data["precision"], dtype=np.float64)
            for c, m_data in models.items()})

    internal_payload = json.dumps(
        sorted(
            ({"sample_id": str(row["sample_id"]), "class_id": int(row["class_id"]), "subset": str(row["subset"])}
             for row in proto_internal_records),
            key=lambda row: (row["class_id"], row["subset"], row["sample_id"]),
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "models": models,
        "tau_alpha": tau_alpha,
        "tau_alpha_global": tau_alpha_global,
        "tau_alpha_clean": tau_alpha_clean,
        "clean_calibration": clean_calibration,
        "score_mode": score_mode,
        "ensemble_recon_weight": ensemble_recon_weight,
        "ensemble_stats": ensemble_stats,
        "alpha": alpha,
        "k_alpha": k_alpha,
        "m": m,
        "pca": pca,
        "prototype_internal_split": {
            "fit_count": sum(1 for row in proto_internal_records if row["subset"] == "proto-fit"),
            "val_count": sum(1 for row in proto_internal_records if row["subset"] == "proto-val"),
            "identifier_hash": hashlib.sha256(internal_payload).hexdigest(),
            "artifact": "osr/prototype_internal_split.csv" if output_dir else None,
        }
    }

def score_multicenter_conformal(
    df: pd.DataFrame, 
    conformal_meta: dict
) -> pd.DataFrame:
    z = normalize_features(np.stack(df["feature"].to_numpy()))
    preds = df.get("pred_before_osr", df["y_raw"]).to_numpy()
    
    models = conformal_meta["models"]
    tau_alpha = conformal_meta["tau_alpha"]
    score_mode = str(conformal_meta.get("score_mode", "candidate")).lower()
    ensemble_stats = conformal_meta.get("ensemble_stats", {})
    recon_weight = float(conformal_meta.get("ensemble_recon_weight", 0.5))
    has_recon = bool(ensemble_stats.get("has_recon", False)) and ("recon_error" in df.columns)
    
    nonconformity_scores = np.zeros(len(df))
    scores_mah = np.zeros(len(df))
    scores_rec = np.full(len(df), np.nan)
    rejected = np.zeros(len(df), dtype=bool)
    nearest_p = np.zeros(len(df), dtype=int)
    nearest_c = np.zeros(len(df), dtype=int)
    
    for i in range(len(df)):
        c = preds[i]
        nearest_c[i] = c
        if c not in models:
            nonconformity_scores[i] = float('inf')
            scores_mah[i] = float('inf')
            rejected[i] = True
            nearest_p[i] = -1
            continue
            
        centers = np.array(models[c]["centers"])
        prec = np.array(models[c]["precision"])
        
        diff = z[i:i+1] - centers
        dists = np.sum(np.dot(diff, prec) * diff, axis=1)
        s_mah = float(np.min(dists))
        n_id = int(np.argmin(dists))
        scores_mah[i] = s_mah
        nearest_p[i] = n_id

        if score_mode in {"confid_ensemble", "ensemble"} and has_recon:
            rec_val = float(df["recon_error"].iloc[i])
            scores_rec[i] = rec_val
            if np.isfinite(rec_val) and np.isfinite(s_mah):
                norm_m = (s_mah - ensemble_stats["mu_mah"]) / ensemble_stats["sigma_mah"]
                norm_r = (rec_val - ensemble_stats["mu_rec"]) / ensemble_stats["sigma_rec"]
                score = float((1.0 - recon_weight) * norm_m + recon_weight * norm_r)
            elif np.isfinite(s_mah):
                score = float((s_mah - ensemble_stats["mu_mah"]) / ensemble_stats["sigma_mah"])
            else:
                score = s_mah
        else:
            score = s_mah

        nonconformity_scores[i] = score
        rejected[i] = bool(score >= tau_alpha)
        
    df_out = df.copy()
    df_out["conformal_score"] = nonconformity_scores
    df_out["score_mah"] = scores_mah
    df_out["score_rec"] = scores_rec
    df_out["rejected"] = rejected
    df_out["nearest_prototype_id"] = nearest_p
    df_out["nearest_prototype_class"] = nearest_c

    total_q = len(df_out)
    n_rej = int(np.sum(rejected))
    n_acc = total_q - n_rej
    logger.info(
        "=== Conformal Decision Rule Evaluation (Algorithm 2) ===\n"
        "  * Total Test Queries: %d\n"
        "  * Admitted Known (s(x) < tau_alpha): %d (%.2f%%)\n"
        "  * Rejected Unknown Anomalies (s(x) >= tau_alpha): %d (%.2f%%)",
        total_q, n_acc, 100.0 * n_acc / max(total_q, 1),
        n_rej, 100.0 * n_rej / max(total_q, 1)
    )
    return df_out
