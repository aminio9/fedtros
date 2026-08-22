"""Comprehensive test suite for the FedTROS canonical analysis package (Item C21).

Tests:
  1. test_result_loader: Verifies structured parsing of run metadata, configs, scalar JSON, and JSONL traces.
  2. test_incompatible_run_rejection: Verifies IncompatibleRunsError on mismatched alpha, held-outs, client counts.
  3. test_seed_aggregation: Verifies mathematical correctness of multi-seed mean, across-seed SD, and 95% CI.
  4. test_paired_delta: Verifies paired differences against baselines per seed.
  5. test_ci_calculation: Verifies Student's t-distribution confidence intervals.
  6. test_temporal_vs_seed_variability: Verifies strict separation between temporal variance and seed variance.
  7. test_open_set_score_export: Verifies C14 OSR sample-level schema compliance.
  8. test_client_metric_export: Verifies C15 client-level performance schema compliance.
  9. test_plot_adapter_schema: Verifies export_plot_data matches schemas expected by plots project.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.aggregation import (
    aggregate_runs,
    compute_metric_stats,
    compute_paired_deltas,
    compute_temporal_last_rounds_stats,
)
from src.analysis.export import (
    export_client_level_contract,
    export_communication_contract,
    export_osr_sample_contract,
    export_runtime_contract,
    build_efficiency_curve,
    generate_provenance_manifest,
)
from src.analysis.loaders import RunRecord, is_run_completed, load_run
from src.analysis.query import query_runs
from src.analysis.statistics import compare_paired_significance, compute_cohens_d, format_p_value
from src.analysis.tables import (
    build_ablation_table,
    build_e1_iid_table,
    build_e3_non_iid_table,
    build_e4_open_set_table,
    export_all_paper_tables,
)
from src.analysis.validation import IncompatibleRunsError, validate_compatibility


def _create_mock_run(
    base_dir: Path,
    run_id: str,
    *,
    study: str = "E4-NIID-FOSR",
    stage: str = "paper_final",
    method: str = "FedTROS-PR",
    dataset: str = "B-NAT",
    alpha: float = 0.1,
    seed: int = 42,
    num_clients: int = 10,
    unknown_labels: list[str] | None = None,
    macro_f1: float = 0.945,
    auroc: float = 0.895,
    accuracy: float = 0.952,
    history_rounds: int = 10,
) -> Path:
    """Create a fully-populated mock experiment run directory for testing."""
    rdir = base_dir / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    unkn = unknown_labels or ["FoT"]

    # 1. metadata.json
    metadata = {
        "run_id": run_id,
        "study": study,
        "stage": stage,
        "method": method,
        "dataset": dataset,
        "alpha": alpha,
        "seed": seed,
        "num_clients": num_clients,
        "unknown_labels": unkn,
        "git_commit": "82d4fdf6e07c7b306671b12e516e839b79a4719f",
        "config_hash": f"cfg_hash_{seed}",
        "timestamp_utc": "2026-08-19T20:00:00Z",
    }
    (rdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run_manifest = {
        **metadata,
        "study_id": study,
        "status": "COMPLETED",
        "split_hash": f"split_hash_{seed}",
    }
    (rdir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    # 2. evaluation_metrics.json
    eval_metrics = {
        "openset_f1_macro": macro_f1,
        "openset_auroc": auroc,
        "openset_auprc": auroc * 0.95,
        "openset_fpr95": 0.05,
        "openset_unknown_recall": 0.88,
        "openset_known_false_unknown_rate": 0.03,
        "local_student_accuracy": accuracy,
        "local_student_f1_macro": macro_f1,
        "overall_accuracy": accuracy,
        "worst_client_macro_f1": macro_f1 - 0.04,
    }
    (rdir / "evaluation_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2), encoding="utf-8"
    )

    # 3. federated_history.csv
    hist_rows = []
    for r in range(1, history_rounds + 1):
        progress = 1.0 - math.exp(-r / 3.0)
        hist_rows.append(
            {
                "round": r,
                "local_student_accuracy": accuracy * progress,
                "round_openset_f1_macro": macro_f1 * progress,
                "round_openset_auroc": auroc * progress,
            }
        )
    pd.DataFrame(hist_rows).to_csv(rdir / "federated_history.csv", index=False)

    # 4. open_set_scores.csv
    scores_df = pd.DataFrame(
        {
            "sample_id": np.arange(20),
            "y_true": [0] * 15 + [99] * 5,
            "raw_pred": [0] * 20,
            "y_pred": [0] * 14 + [99] * 6,
            "known_or_unknown": ["known"] * 15 + ["unknown"] * 5,
            "recon_error": np.linspace(0.01, 0.99, 20),
            "prototype_rank_score": np.linspace(0.05, 0.95, 20),
            "final_reject": [0] * 14 + [1] * 6,
        }
    )
    scores_df.to_csv(rdir / "open_set_scores.csv", index=False)

    # 5. before and after confusion matrices
    labels = ["Normal", "BP", "DoS", "MitM", "Unknown"]
    cm_b = pd.DataFrame(np.eye(5, dtype=int) * 20, index=labels, columns=labels)
    cm_a = pd.DataFrame(np.eye(5, dtype=int) * 20, index=labels, columns=labels)
    cm_b.to_csv(rdir / "before_osr_confusion_matrix.csv")
    cm_a.to_csv(rdir / "after_osr_confusion_matrix.csv")

    # 6. client-level performance metrics
    fair_rows = []
    for c in range(num_clients):
        fair_rows.append(
            {
                "round": history_rounds,
                "client_id": c,
                "num_examples": 100 + c * 10,
                "class_count": 4,
                "accuracy": accuracy - (c * 0.005),
                "macro_f1": macro_f1 - (c * 0.005),
            }
        )
    (rdir / "metrics").mkdir(exist_ok=True)
    pd.DataFrame(fair_rows).to_csv(rdir / "metrics" / "client_metrics.csv", index=False)

    # 7. communication metrics
    comm_df = pd.DataFrame(
        {
            "round": list(range(1, history_rounds + 1)),
            "total_bytes": [1024 * 1024] * history_rounds,
            "cumulative_bytes": [1024 * 1024 * r for r in range(1, history_rounds + 1)],
            "validation_accuracy": [
                accuracy * (1.0 - math.exp(-r / 3.0)) for r in range(1, history_rounds + 1)
            ],
            "validation_macro_f1": [
                macro_f1 * (1.0 - math.exp(-r / 3.0)) for r in range(1, history_rounds + 1)
            ],
        }
    )
    comm_df.to_csv(rdir / "communication_metrics.csv", index=False)

    # 8. scalability round metrics
    scalability_df = pd.DataFrame(
        {
            "round": list(range(1, history_rounds + 1)),
            "num_clients": [num_clients] * history_rounds,
            "mean_client_macro_f1": [macro_f1] * history_rounds,
            "std_client_macro_f1": [0.02] * history_rounds,
            "worst_client_macro_f1": [macro_f1 - 0.05] * history_rounds,
            "client_fit_wall_time_sec": [12.5] * history_rounds,
            "round_time_sec": [15.0] * history_rounds,
            "server_aggregation_time_sec": [1.5] * history_rounds,
            "open_set_round_eval_time_sec": [1.0] * history_rounds,
            "round_openset_f1_macro": [macro_f1] * history_rounds,
            "round_openset_overall_acc": [accuracy] * history_rounds,
            "round_openset_known_acc": [accuracy] * history_rounds,
            "round_openset_auroc": [auroc] * history_rounds,
            "round_openset_fpr95": [0.05] * history_rounds,
            "round_openset_unknown_recall": [0.88] * history_rounds,
        }
    )
    scalability_df.to_csv(rdir / "scalability_round_metrics.csv", index=False)

    return rdir


def test_result_loader(tmp_path: Path):
    """Test loading RunRecord and safe ingestion."""
    rdir = _create_mock_run(tmp_path, "run_001", seed=42, macro_f1=0.935)

    assert is_run_completed(rdir) is True
    record = load_run(rdir)

    assert record.run_id == "run_001"
    assert record.study == "E4-NIID-FOSR"
    assert record.method == "FedTROS-PR"
    assert record.seed == 42
    assert record.num_clients == 10
    assert record.alpha == 0.1
    assert record.status == "COMPLETED"

    f1 = record.get_metric(["openset_f1_macro"])
    assert f1 is not None
    assert abs(f1 - 0.935) < 1e-5

    # Lazy loaded frames
    assert not record.history.empty
    assert len(record.history) == 10
    assert not record.scores.empty
    assert not record.confusion_before.empty
    assert not record.client_metrics.empty
    assert not record.communication.empty
    assert not record.scalability.empty


def test_incompatible_run_rejection(tmp_path: Path):
    """Test that mismatched parameters raise IncompatibleRunsError."""
    r1 = load_run(_create_mock_run(tmp_path, "run_s42_a01", seed=42, alpha=0.1))
    r2 = load_run(
        _create_mock_run(tmp_path, "run_s73_a05", seed=73, alpha=0.5)
    )  # Mismatched alpha!

    with pytest.raises(IncompatibleRunsError, match="Mismatched Dirichlet alpha"):
        validate_compatibility([r1, r2], require_same_alpha=True)

    r3 = load_run(
        _create_mock_run(tmp_path, "run_s73_diff_unkn", seed=73, alpha=0.1, unknown_labels=["DoS"])
    )
    with pytest.raises(IncompatibleRunsError, match="Mismatched held-out unknown"):
        validate_compatibility([r1, r3], require_same_held_out=True)

    r4 = load_run(_create_mock_run(tmp_path, "run_s42_dup", seed=42, alpha=0.1))
    with pytest.raises(IncompatibleRunsError, match="Duplicate seed execution"):
        validate_compatibility([r1, r4], require_compatible_method=True)


def test_seed_aggregation(tmp_path: Path):
    """Test multi-seed statistical aggregation: mean, SD, CI."""
    f1_values = [0.920, 0.940, 0.960]
    runs = []
    for idx, (seed, val) in enumerate(zip([17, 42, 73], f1_values)):
        rdir = _create_mock_run(tmp_path, f"run_seed_{seed}", seed=seed, macro_f1=val)
        runs.append(load_run(rdir))

    agg = aggregate_runs(runs)
    assert agg.completed_count == 3
    assert agg.seeds == [17, 42, 73]

    f1_stat = agg.metrics["openset_f1_macro"]
    assert abs(f1_stat.mean - 0.940) < 1e-5
    assert abs(f1_stat.std_across_seeds - 0.020) < 1e-5
    assert f1_stat.n == 3

    # Format verification
    fmt = f1_stat.format_mean_std(percent=True)
    assert "94.00 ± 2.00" in fmt


def test_ci_calculation():
    """Verify Student's t-distribution confidence interval calculation."""
    vals = [10.0, 12.0, 14.0]  # mean = 12.0, sample std = 2.0, n = 3
    # df = 2, t_crit(0.975) = 4.30265
    # sem = 2.0 / sqrt(3) = 1.1547
    # ci_margin = 4.30265 * 1.1547 = 4.968
    stats_res = compute_metric_stats(vals)
    assert abs(stats_res.mean - 12.0) < 1e-4
    assert abs(stats_res.std_across_seeds - 2.0) < 1e-4
    assert abs(stats_res.ci95_margin - 4.968) < 1e-2
    assert abs(stats_res.ci95_low - (12.0 - 4.968)) < 1e-2
    assert abs(stats_res.ci95_high - (12.0 + 4.968)) < 1e-2


def test_paired_delta(tmp_path: Path):
    """Test paired per-seed difference against baseline: Delta = Candidate - Baseline."""
    cand_runs = [
        load_run(
            _create_mock_run(tmp_path, "cand_17", method="FedTROS-PR", seed=17, macro_f1=0.94)
        ),
        load_run(
            _create_mock_run(tmp_path, "cand_42", method="FedTROS-PR", seed=42, macro_f1=0.96)
        ),
        load_run(
            _create_mock_run(tmp_path, "cand_73", method="FedTROS-PR", seed=73, macro_f1=0.95)
        ),
    ]
    base_runs = [
        load_run(_create_mock_run(tmp_path, "base_17", method="FedAvg", seed=17, macro_f1=0.88)),
        load_run(_create_mock_run(tmp_path, "base_42", method="FedAvg", seed=42, macro_f1=0.89)),
        load_run(_create_mock_run(tmp_path, "base_73", method="FedAvg", seed=73, macro_f1=0.87)),
    ]
    # Deltas: (0.94 - 0.88)=0.06, (0.96 - 0.89)=0.07, (0.95 - 0.87)=0.08
    # Mean delta = 0.07, std = 0.01

    deltas = compute_paired_deltas(cand_runs, base_runs, ["openset_f1_macro"])
    assert "openset_f1_macro" in deltas
    d_stat = deltas["openset_f1_macro"]
    assert abs(d_stat.mean - 0.07) < 1e-5
    assert abs(d_stat.std_across_seeds - 0.01) < 1e-5


def test_temporal_vs_seed_variability(tmp_path: Path):
    """Verify strict separation between temporal last-round variance and seed variance."""
    runs = [
        load_run(_create_mock_run(tmp_path, "run_t1", seed=17, macro_f1=0.92, history_rounds=10)),
        load_run(_create_mock_run(tmp_path, "run_t2", seed=42, macro_f1=0.96, history_rounds=10)),
    ]
    agg = aggregate_runs(runs)

    # Across-seed SD
    seed_sd = agg.metrics["openset_f1_macro"].std_across_seeds
    assert seed_sd > 0.0

    # Temporal last-10 SD
    temporal = agg.temporal_metrics
    assert "temporal_last10_mean" in temporal
    assert "temporal_last10_sd" in temporal
    # They are distinct fields in the data structure and not conflated
    assert seed_sd != temporal["temporal_last10_sd"]


def test_open_set_score_export(tmp_path: Path):
    """Verify C14 OSR sample-level schema contract export."""
    scores_df = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "y_true": [0, 1, 99],
            "raw_pred": [0, 1, 0],
            "y_pred": [0, 1, 99],
            "known_or_unknown": ["known", "known", "unknown"],
            "recon_error": [0.1, 0.2, 0.9],
            "prototype_rank_score": [0.15, 0.25, 0.95],
            "final_reject": [0, 0, 1],
        }
    )
    out_p = tmp_path / "test_osr_scores.csv"
    contract_df = export_osr_sample_contract(scores_df, out_p)

    expected_cols = [
        "sample_id",
        "true_label",
        "closed_pred",
        "open_pred",
        "unknown_flag",
        "raw_score",
        "rank_score",
        "is_rejected",
    ]
    assert list(contract_df.columns) == expected_cols
    assert out_p.exists()
    assert len(contract_df) == 3
    assert list(contract_df["unknown_flag"]) == [0, 0, 1]
    assert list(contract_df["is_rejected"]) == [0, 0, 1]


def test_client_metric_export(tmp_path: Path):
    """Verify C15 client-level performance schema contract export."""
    client_metrics_df = pd.DataFrame(
        {
            "round": [5, 5],
            "client_id": [0, 1],
            "num_examples": [200, 300],
            "class_count": [4, 4],
            "class_coverage": [0.8, 0.8],
            "accuracy": [0.92, 0.94],
            "macro_f1": [0.91, 0.93],
        }
    )
    out_p = tmp_path / "test_client_metrics.csv"
    contract_df = export_client_level_contract(client_metrics_df, out_p)

    expected_cols = [
        "round",
        "client_id",
        "sample_count",
        "class_count",
        "class_coverage",
        "accuracy",
        "macro_f1",
    ]
    assert list(contract_df.columns) == expected_cols
    assert out_p.exists()
    assert len(contract_df) == 2


def test_publication_bundle_schema(tmp_path: Path):
    """Verify the versioned FedTROS -> separate plots publication-bundle contract."""
    runs_dir = tmp_path / "outputs"
    for s in (17, 42, 73):
        _create_mock_run(runs_dir, f"mock_run_s{s}", seed=s, alpha=0.1)

    from scripts.export_publication_bundle import SCHEMA_NAME, SCHEMA_VERSION, export

    target_root = tmp_path / "publication_exports"
    bundle_dir = export(
        outputs=runs_dir,
        target_root=target_root,
        freeze_id="test-freeze",
        include_stages=["paper_final"],
    )

    manifest_path = bundle_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_name"] == SCHEMA_NAME
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["method"] == "FedTROS-PR"
    assert manifest["config_freeze_id"] == "test-freeze"
    assert set(manifest["source_run_ids"]) == {"mock_run_s17", "mock_run_s42", "mock_run_s73"}
    assert (bundle_dir / "E4-NIID-FOSR" / "summary.csv").exists()
    assert (bundle_dir / "E4-NIID-FOSR" / "scores.csv").exists()
    assert (bundle_dir / "provenance" / "artifact_sources.json").exists()


def test_efficiency_curve_joins_real_communication_and_round_history(tmp_path: Path):
    """E7 joins actual communication bytes to recorded validation history by round."""
    rdir = _create_mock_run(
        tmp_path,
        "e7_real_style",
        study="E7-EFFICIENCY",
        seed=42,
        alpha=0.5,
        history_rounds=4,
    )
    # Match the real FedTROS communication recorder: bytes only, no duplicated
    # validation-performance column.
    metrics_dir = rdir / "metrics"
    pd.DataFrame(
        {
            "round": [1, 2, 3, 4],
            "communication/downlink_bytes": [1000, 1000, 1000, 1000],
            "communication/uplink_bytes": [1000, 1000, 1000, 1000],
            "communication/round_bytes": [2000, 2000, 2000, 2000],
            "communication/cumulative_bytes": [2000, 4000, 6000, 8000],
        }
    ).to_csv(metrics_dir / "communication_round.csv", index=False)
    # Multiple events can exist per round.  The contract must prefer the central
    # validation Macro-F1 series and never derive performance from communication.
    rows = []
    for rnd, score in enumerate([0.70, 0.76, 0.81, 0.84], start=1):
        rows.append({"federated/round": rnd, "federated/phase": "fit", "student/loss": 1.0 / rnd})
        rows.append(
            {"federated/round": rnd, "federated/phase": "central_validation", "val/macro_f1": score}
        )
    pd.DataFrame(rows).to_csv(metrics_dir / "round_metrics.csv", index=False)

    curve = build_efficiency_curve([load_run(rdir)])
    assert list(curve["round"]) == [1, 2, 3, 4]
    assert list(curve["communication/cumulative_bytes"]) == [2000, 4000, 6000, 8000]
    assert np.allclose(curve["performance_value"], [0.70, 0.76, 0.81, 0.84])
    assert set(curve["performance_metric"]) == {"val/macro_f1"}


def test_e7_publication_bundle_exports_efficiency_curve(tmp_path: Path):
    """The two-repository contract contains the standardized E7 curve file."""
    runs_dir = tmp_path / "outputs"
    rdir = _create_mock_run(
        runs_dir,
        "e7_bundle_run",
        study="E7-EFFICIENCY",
        seed=42,
        alpha=0.5,
        history_rounds=3,
    )
    # Synthetic helper already records performance beside communication; the
    # exporter must normalize it into the canonical efficiency contract.
    from scripts.export_publication_bundle import export

    bundle = export(
        outputs=runs_dir,
        target_root=tmp_path / "publication_exports",
        freeze_id="e7-contract",
        include_stages=["paper_final"],
    )
    path = bundle / "E7-EFFICIENCY" / "efficiency_curve.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert {"communication/cumulative_bytes", "performance_metric", "performance_value"}.issubset(
        frame.columns
    )


def test_existing_plot_adapter_exports_complete_29_figure_contract(tmp_path: Path):
    """Canonical run outputs are transformed into every input used by the existing renderer."""
    runs_dir = tmp_path / "outputs"

    def make(run_id: str, **kwargs):
        path = _create_mock_run(runs_dir, run_id, **kwargs)
        record = load_run(path)
        distribution = pd.DataFrame({
            "client_id": range(1, record.num_clients + 1),
            "Normal": [60] * record.num_clients,
            "BP": [10] * record.num_clients,
            "DoS": [10] * record.num_clients,
            "MitM": [10] * record.num_clients,
            "FoT": [10] * record.num_clients,
        })
        (path / "data").mkdir(exist_ok=True)
        distribution.to_csv(path / "data" / "client_class_distribution.csv", index=False)
        return path

    paths = [
        make("e1", study="E1-IID-CS", method="FedTROS-PR", alpha=1.0, num_clients=10),
        make("e2", study="E2-IID-OSR", method="FedTROS-PR", alpha=1.0, num_clients=10),
        make("e3_a01", study="E3-NIID-CS", method="FedTROS-PR", alpha=0.1, num_clients=10),
        make("e3_a05", study="E3-NIID-CS", method="FedTROS-PR", alpha=0.5, num_clients=10),
        make("e3_a1", study="E3-NIID-CS", method="FedTROS-PR", alpha=1.0, num_clients=10),
        make("e3_a1_avg", study="E3-NIID-CS", method="FedAvg", alpha=1.0, num_clients=10),
        make("e3_a1_prox", study="E3-NIID-CS", method="FedProx", alpha=1.0, num_clients=10),
        make("e6_10", study="E6-SCALE", method="FedTROS-PR", num_clients=10),
        make("e6_50", study="E6-SCALE", method="FedTROS-PR", num_clients=50),
        make("e6_100", study="E6-SCALE", method="FedTROS-PR", num_clients=100),
        make("e7", study="E7-EFFICIENCY", method="FedTROS-PR", num_clients=10),
        make("a1", study="A1-TEACHER", method="FedTROS-PR", num_clients=10),
    ]
    e2 = paths[1]
    projection_dir = e2 / "artifacts"
    projection_dir.mkdir(exist_ok=True)
    pd.DataFrame({
        "x": [0.0, 1.0, 0.5], "y": [0.0, 1.0, 0.5],
        "point_type": ["sample", "positive_prototype", "negative_prototype"],
        "label": ["Normal", "Normal", "Boundary"],
        "is_unknown": [0, 0, 1], "final_reject": [0, 0, 1], "sample_id": [0, pd.NA, pd.NA],
    }).to_csv(projection_dir / "prototype_rank_latent_projection.csv", index=False)

    from scripts.export_plot_data import REQUIRED_FILES, REQUIRED_SCALABILITY_COLUMNS, export_plot_data

    output_dir = tmp_path / "plot_data"
    manifest = export_plot_data([load_run(path) for path in paths], output_dir, strict=True)
    assert manifest["status"] == "COMPLETE"
    assert REQUIRED_FILES.issubset({path.name for path in output_dir.iterdir()})
    scores = pd.read_csv(output_dir / "exp2_scores.csv")
    assert {"known_or_unknown", "prototype_rank_score", "selected_threshold_used", "final_reject"}.issubset(scores.columns)
    scalability = pd.read_csv(output_dir / "scalability_100_clients.csv")
    assert REQUIRED_SCALABILITY_COLUMNS.issubset(scalability.columns)
    communication = pd.read_csv(output_dir / "communication_alpha1.csv")
    assert {"round", "method", "cumulative_mb", "accuracy_percent"}.issubset(
        communication.columns
    )
