from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.plotting import generate_plots


def _write_fixture_tables(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "num_clients": [3, 10, 20, 50, 100],
            "final_accuracy": [0.967, 0.9577, 0.925, 0.918, 0.905],
        }
    ).to_csv(run_dir / "scalability.csv", index=False)

    pd.DataFrame(
        {
            "client_id": [1, 2, 3],
            "Normal": [12, 2, 1],
            "BP": [1, 8, 2],
            "DoS": [0, 1, 9],
            "MitM": [2, 0, 1],
            "FoT": [0, 1, 0],
        }
    ).to_csv(run_dir / "client_class_distribution.csv", index=False)

    comparison = []
    for method, base in {"FMRL_LA": 0.94, "FedProx": 0.90, "FedAvg": 0.88}.items():
        for round_idx in range(1, 6):
            comparison.append(
                {
                    "round": round_idx,
                    "method": method,
                    "alpha": 0.1,
                    "accuracy": base - (5 - round_idx) * 0.005,
                }
            )
    pd.DataFrame(comparison).to_csv(run_dir / "comparison_metrics.csv", index=False)

    pd.DataFrame(
        {
            "unknown_score": [0.08, 0.12, 0.18, 0.72, 0.82, 0.91],
            "is_unknown": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(run_dir / "open_set_scores.csv", index=False)
    (run_dir / "open_set_metrics.json").write_text(
        json.dumps({"openset_global_delta": 0.5}, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "fpr": np.linspace(0, 1, 6),
            "tpr": np.array([0.0, 0.45, 0.68, 0.82, 0.93, 1.0]),
            "method": ["FMRL_LA"] * 6,
        }
    ).to_csv(run_dir / "open_set_roc_curve.csv", index=False)

    cross_dataset = pd.DataFrame(
        {
            "dataset": ["B-TAT", "B-TAT", "ToN-IoT", "ToN-IoT", "CIC-IDS2017", "CIC-IDS2017"],
            "metric": ["f1", "auroc", "f1", "auroc", "f1", "auroc"],
            "metric_value": [0.96, 0.95, 0.94, 0.92, 0.91, 0.89],
        }
    )
    cross_dataset.to_csv(run_dir / "cross_dataset_metrics.csv", index=False)

    labels = ["Normal", "BP", "DoS", "MitM", "Unknown"]
    before = pd.DataFrame(
        [
            [24, 2, 1, 0, 0],
            [1, 22, 2, 0, 0],
            [1, 2, 21, 1, 0],
            [0, 1, 2, 23, 0],
            [7, 4, 2, 3, 0],
        ],
        index=labels,
        columns=labels,
    )
    after = pd.DataFrame(
        [
            [24, 1, 1, 0, 1],
            [1, 22, 1, 0, 1],
            [0, 1, 22, 0, 2],
            [0, 1, 1, 24, 0],
            [0, 0, 0, 1, 29],
        ],
        index=labels,
        columns=labels,
    )
    before.to_csv(run_dir / "before_osr_confusion_matrix.csv")
    after.to_csv(run_dir / "after_osr_confusion_matrix.csv")

    seed_rows = []
    for heterogeneity, center in [("0.1", 88.5), ("0.5", 90.1), ("1.0", 91.4), ("10", 92.2), ("IID", 94.6)]:
        for offset in (-0.6, 0.0, 0.5, 0.8):
            seed_rows.append({"heterogeneity": heterogeneity, "accuracy": (center + offset) / 100.0})
    pd.DataFrame(seed_rows).to_csv(run_dir / "seed_robustness.csv", index=False)

    latent_rows = []
    for label, center in [("Normal", (-2.0, 0.5)), ("DoS", (0.5, 1.5)), ("MitM", (1.8, -1.1)), ("Unknown", (3.2, 2.8))]:
        for idx in range(12):
            latent_rows.append(
                {
                    "x": center[0] + np.cos(idx) * 0.2,
                    "y": center[1] + np.sin(idx) * 0.2,
                    "label": label,
                }
            )
    pd.DataFrame(latent_rows).to_csv(run_dir / "latent_embeddings.csv", index=False)

    pd.DataFrame(
        {
            "method": ["FMRL_LA"] * 5 + ["FedAvg"] * 5,
            "cumulative_mb": [1.2, 2.4, 3.6, 4.8, 6.0, 2.5, 5.0, 7.5, 10.0, 12.5],
            "accuracy": [0.89, 0.91, 0.925, 0.932, 0.94, 0.86, 0.875, 0.885, 0.89, 0.895],
        }
    ).to_csv(run_dir / "communication_metrics.csv", index=False)

    pd.DataFrame(
        {
            "configuration": [
                "Base Model (Centralized, No OSR)",
                "Base + FL (FedAvg)",
                "Base + OSR (Centralized)",
                "Proposed (FL + OSR)",
            ],
            "macro_f1": [0.72, 0.70, 0.91, 0.94],
        }
    ).to_csv(run_dir / "ablation_metrics.csv", index=False)


def test_generate_plots_writes_all_required_figures(tmp_path):
    run_dir = tmp_path / "outputs" / "run_001"
    _write_fixture_tables(run_dir)

    cfg = OmegaConf.create(
        {
            "tracking": {"run_dir": "outputs/run_001"},
            "plotting": {
                "output_dir": "${tracking.run_dir}/plots",
                "formats": ["png", "pdf"],
                "plot_dpi": 120,
            },
        }
    )

    generated = generate_plots(cfg, project_root=tmp_path, run_dir="outputs/run_001")

    plots_dir = run_dir / "plots"
    manifest = json.loads((plots_dir / "plot_manifest.json").read_text(encoding="utf-8"))

    assert len(generated) == 28
    assert len(manifest["files"]) == 28
    assert manifest["output_dir"] == str(plots_dir)

    pngs = sorted(plots_dir.glob("*.png"))
    pdfs = sorted(plots_dir.glob("*.pdf"))
    assert len(pngs) == 14
    assert len(pdfs) == 14
    for path in pngs + pdfs:
        assert path.stat().st_size > 0

    assert (plots_dir / "05_known_unknown_score_distribution.png").exists()
    assert (plots_dir / "10_confusion_matrix_after_osr.pdf").exists()
