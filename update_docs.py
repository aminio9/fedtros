import os

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements
    replacements = {
        "Prototype-Rank Rejection (PR)": "Multicenter Conformal Rejection (MC)",
        "Prototype-Rank Rejection": "Multicenter Conformal Rejection",
        "Prototype-Rank": "Multicenter Conformal",
        "FedTROS-PR": "FedTROS-MC",
        "q_i": "kappa_i",
        "q_i anchoring": "kappa_i entropy-effective anchoring",
        "theta_O": "theta_O (ablation only)",
        "class-conditioned OSR branch": "class-conditioned OSR branch (ablation only)",
        "reconstruction OSR loss": "reconstruction OSR loss (ablation only)",
        "boundary interpolation": "boundary interpolation (ablation only)",
        "5000 boundary features": "5000 boundary features (ablation only)",
        "32 boundary prototypes": "32 boundary prototypes (ablation only)",
        "lambda_b = 0.35": "lambda_b = 0.35 (ablation only)",
        "K_c = min(16, floor(N_c/25))": "Adaptive K_c in {1..10} via Silhouette score",
        "r_c = Q_0.95": "Ledoit-Wolf shrinkage covariance",
        "empirical rank": "split conformal calibration",
        "tau_rank": "tau_alpha",
        "rho = 0.05": "alpha = 0.05",
        "n_min": "n_min (removed from canonical)",
        "n_max": "n_max (removed from canonical)",
        "a_min": "a_min (removed from canonical)",
        "clipped sqrt": "gamma=0.5",
    }
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        print(f"Updated {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

for root, _, files in os.walk('d:/Research/Code/fedtros'):
    for f in files:
        if f.endswith('.md'):
            process_file(os.path.join(root, f))
