import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.covariance import LedoitWolf

logger = logging.getLogger("Conformal")

EPS = 1e-8

def normalize_features(h: np.ndarray) -> np.ndarray:
    """Phase 6: Canonical frozen feature z(x) = h_S(x) / (||h_S(x)||_2 + eps)"""
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    return h / np.maximum(norms, EPS)

def fit_multicenter_conformal(
    df_proto: pd.DataFrame, 
    df_calib: pd.DataFrame, 
    num_classes: int, 
    alpha: float = 0.05
) -> dict:
    """
    Phases 7, 8, 11: Fit centers, Ledoit-Wolf covariance, and conformal thresholds.
    """
    models = {}
    
    # Phase 7 & 8: Fit per class
    for c in range(num_classes):
        cls_proto = df_proto[df_proto["y_raw"] == c]
        if cls_proto.empty:
            continue
            
        z_c = normalize_features(np.stack(cls_proto["feature"].to_numpy()))
        
        # Adaptive multi-center fitting K_c in {1..10}
        best_k = 1
        best_score = -1.0
        best_centers = z_c.mean(axis=0, keepdims=True)
        best_labels = np.zeros(len(z_c), dtype=int)
        
        max_k = min(10, len(z_c) - 1)
        if max_k > 1:
            for k in range(2, max_k + 1):
                try:
                    km = KMeans(n_clusters=k, n_init=5, random_state=42)
                    labels = km.fit_predict(z_c)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(z_c, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                            best_centers = km.cluster_centers_
                            best_labels = labels
                except Exception as e:
                    pass
                    
        # Compute residuals
        residuals = z_c - best_centers[best_labels]
        
        # Phase 8: Fit Ledoit-Wolf covariance
        try:
            lw = LedoitWolf()
            lw.fit(residuals)
            precision = lw.precision_
        except Exception as e:
            logger.warning(f"LedoitWolf failed for class {c}: {e}. Falling back to identity precision.")
            precision = np.eye(z_c.shape[1])
            
        models[c] = {
            "centers": best_centers.tolist(),
            "precision": precision.tolist(),
            "k": best_k
        }
        logger.info(f"Class {c}: fitted K={best_k} centers, precision_shape={precision.shape}")
        
    # Phase 11: Candidate-conditioned conformal calibration
    thresholds = {}
    if not df_calib.empty:
        calib_z = normalize_features(np.stack(df_calib["feature"].to_numpy()))
        # Phase 11: Candidate-conditioned conformal calibration
        # Group by the predicted candidate, NOT the true label
        if "pred_before_osr" in df_calib.columns:
            calib_pred = df_calib["pred_before_osr"].to_numpy()
        else:
            calib_pred = df_calib["y_raw"].to_numpy()

        
        for c in range(num_classes):
            mask = calib_pred == c
            if not np.any(mask) or c not in models:
                thresholds[c] = float('inf')
                continue
                
            z_calib_c = calib_z[mask]
            
            # Phase 9 score: min_k (z - mu_k)^T Sigma^-1 (z - mu_k)
            centers = np.array(models[c]["centers"])
            prec = np.array(models[c]["precision"])
            
            scores = []
            for z in z_calib_c:
                diff = z[None, :] - centers
                # dists: [K]
                dists = np.sum(np.dot(diff, prec) * diff, axis=1)
                scores.append(np.min(dists))
                
            scores = np.array(scores)
            scores.sort()
            
            m = len(scores)
            # Exact finite-sample conformal order statistic
            k = int(np.ceil((m + 1) * (1.0 - alpha)))
            if k <= m:
                tau = scores[k - 1]
            else:
                tau = float('inf')
                
            thresholds[c] = float(tau)
            logger.info(f"Class {c}: calibration n={m}, k={k}, tau={tau:.4f}")
            
    return {
        "models": models,
        "thresholds": thresholds,
        "alpha": alpha
    }

def score_multicenter_conformal(
    df: pd.DataFrame, 
    conformal_meta: dict
) -> pd.DataFrame:
    """
    Phase 9 & 10: Compute raw nonconformity scores and threshold using canonical calibration.
    """
    z = normalize_features(np.stack(df["feature"].to_numpy()))
    preds = df["pred_before_osr"].to_numpy()
    
    models = conformal_meta["models"]
    thresholds = conformal_meta["thresholds"]
    
    nonconformity_scores = np.zeros(len(df))
    rejected = np.zeros(len(df), dtype=bool)
    
    for i in range(len(df)):
        c = preds[i]
        if c not in models:
            nonconformity_scores[i] = float('inf')
            rejected[i] = True
            continue
            
        centers = np.array(models[c]["centers"])
        prec = np.array(models[c]["precision"])
        
        diff = z[i:i+1] - centers
        dists = np.sum(np.dot(diff, prec) * diff, axis=1)
        score = np.min(dists)
        
        nonconformity_scores[i] = score
        rejected[i] = score > thresholds.get(c, float('inf'))
        
    df_out = df.copy()
    df_out["conformal_score"] = nonconformity_scores
    df_out["rejected"] = rejected
    return df_out
