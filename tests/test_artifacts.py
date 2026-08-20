from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.artifacts.communication import build_communication_metrics
from src.artifacts.embeddings import export_latent_embeddings
from src.artifacts.suite import build_suite_artifacts
from src.evaluation.compare import compare_runs


class SelectFirstTwo(nn.Module):
    def forward(self, features):
        mu = features[:, :2].contiguous()
        return mu, torch.zeros_like(mu)


def _write_resolved_config(
    run_dir: Path,
    *,
    run_id: str,
    dataset: str,
    source_labels: list[str],
    known_labels: list[str],
    method: str,
    num_clients: int,
    seed: int,
    alpha: float,
    iid: bool = False,
) -> None:
    cfg = OmegaConf.create(
        {
            "seed": seed,
            "dataset": {
                "name": dataset,
                "source_labels": source_labels,
                "known_labels": known_labels,
                "preprocessing": {
                    "output_dir": str(run_dir / "processed"),
                    "known_labels": known_labels,
                    "alpha": alpha,
                    "iid": iid,
                },
            },
            "federated": {"num_clients": num_clients},
            "experiment": {"name": "baseline", "method": method},
            "model": {"name": "openset_qchain"},
        }
    )
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "num_clients": num_clients,
                "alpha": alpha,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_evaluation_metrics(
    run_dir: Path,
    *,
    test_accuracy: float,
    macro_f1: float,
    open_set_auroc: float,
) -> None:
    metrics = {
        "timestamp_utc": "2026-05-20T00:00:00+00:00",
        "test/accuracy": test_accuracy,
        "test/macro_f1": macro_f1,
        "open_set/auroc": open_set_auroc,
        "openset_auroc": open_set_auroc,
        "openset_f1_macro": macro_f1,
        "openset_overall_acc": test_accuracy,
    }
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)


def _write_communication_csv(run_dir: Path, *, method: str, seed: int) -> None:
    pd.DataFrame(
        {
            "method": [method, method],
            "round": [1, 2],
            "cumulative_mb": [1.5 + seed * 0.01, 3.0 + seed * 0.01],
            "accuracy": [0.70 + seed * 0.001, 0.72 + seed * 0.001],
            "source_run_dir": [str(run_dir), str(run_dir)],
        }
    ).to_csv(run_dir / "communication_metrics.csv", index=False)


def _write_latent_csv(run_dir: Path) -> None:
    pd.DataFrame(
        {
            "x": [0.1, 0.2, 1.1, 1.2, 2.1],
            "y": [0.5, 0.6, 1.5, 1.6, 2.5],
            "label": ["Normal", "DoS", "MitM", "Unknown", "Unknown"],
        }
    ).to_csv(run_dir / "latent_embeddings.csv", index=False)


def _write_run(
    tmp_path: Path,
    *,
    run_id: str,
    dataset: str,
    source_labels: list[str],
    known_labels: list[str],
    method: str,
    num_clients: int,
    seed: int,
    alpha: float,
    test_accuracy: float,
    macro_f1: float,
    open_set_auroc: float,
    include_latent: bool = False,
) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(
        run_dir,
        run_id=run_id,
        dataset=dataset,
        source_labels=source_labels,
        known_labels=known_labels,
        method=method,
        num_clients=num_clients,
        seed=seed,
        alpha=alpha,
    )
    _write_evaluation_metrics(
        run_dir,
        test_accuracy=test_accuracy,
        macro_f1=macro_f1,
        open_set_auroc=open_set_auroc,
    )
    _write_communication_csv(run_dir, method=method, seed=seed)
    if include_latent:
        _write_latent_csv(run_dir)
    return run_dir


def test_export_latent_embeddings_writes_projection_csv(tmp_path):
    output_path = tmp_path / "latent_embeddings.csv"
    frame = export_latent_embeddings(
        model=SelectFirstTwo(),
        features=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        labels=torch.tensor([0, 1, -1]),
        class_names={0: "Normal", 1: "DoS"},
        output_path=output_path,
        batch_size=2,
        max_points=10,
    )

    assert output_path.exists()
    assert list(frame.columns) == ["x", "y", "label"]
    assert frame["label"].tolist() == ["Normal", "DoS", "Unknown"]
    assert frame["x"].tolist() == [1.0, 4.0, 7.0]
    assert frame["y"].tolist() == [2.0, 5.0, 8.0]


def test_build_communication_metrics_uses_logical_rounds(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_resolved_config(
        run_dir,
        run_id="run",
        dataset="B-NAT",
        source_labels=["Normal", "BP", "DoS", "MitM", "FoT"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="FedTROS",
        num_clients=2,
        seed=42,
        alpha=0.1,
    )
    checkpoint = {
        "student_model": {"w": torch.ones(4, 4)},
        "teacher": {"w": torch.ones(4, 4)},
        "teacher_to_student_aligner": {"w": torch.ones(4, 4)},
    }
    torch.save(checkpoint, run_dir / "best_model.pt")
    (run_dir / "fedtros_monitoring.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "phase_a_selection",
                        "logical_round": 1,
                        "selected_clients": 2,
                    }
                ),
                json.dumps(
                    {
                        "event": "phase_b_aggregation",
                        "logical_round": 1,
                        "uploads": [{"cid": "1"}],
                    }
                ),
                json.dumps(
                    {
                        "event": "phase_a_selection",
                        "logical_round": 2,
                        "selected_clients": 2,
                    }
                ),
                json.dumps(
                    {
                        "event": "phase_b_aggregation",
                        "logical_round": 2,
                        "uploads": [{"cid": "1"}, {"cid": "2"}],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    history = pd.DataFrame(
        {
            "round": [1, 2],
            "metric_name": ["accuracy", "accuracy"],
            "metric_value": [0.7, 0.8],
        }
    )

    frame = build_communication_metrics(run_dir=run_dir, project_root=tmp_path, history_frame=history)

    assert frame["round"].tolist() == [1, 2]
    assert frame["accuracy"].tolist() == [0.7, 0.8]
    assert frame["cumulative_mb"].is_monotonic_increasing


def test_build_suite_artifacts_generates_suite_csvs(tmp_path):
    run_1 = _write_run(
        tmp_path,
        run_id="scale_bnat_seed42",
        dataset="B-NAT",
        source_labels=["Normal", "BP", "DoS", "MitM", "FoT"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="FedTROS",
        num_clients=3,
        seed=42,
        alpha=0.1,
        test_accuracy=0.91,
        macro_f1=0.82,
        open_set_auroc=0.77,
        include_latent=True,
    )
    run_2 = _write_run(
        tmp_path,
        run_id="scale_bnat_seed43",
        dataset="B-NAT",
        source_labels=["Normal", "BP", "DoS", "MitM", "FoT"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="FedTROS",
        num_clients=10,
        seed=43,
        alpha=0.1,
        test_accuracy=0.88,
        macro_f1=0.80,
        open_set_auroc=0.79,
    )
    run_3 = _write_run(
        tmp_path,
        run_id="cross_toniot_seed44",
        dataset="ToN-IoT",
        source_labels=["Normal", "BP", "DoS", "MitM", "UnknownA", "UnknownB"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="FedTROS",
        num_clients=20,
        seed=44,
        alpha=0.5,
        test_accuracy=0.86,
        macro_f1=0.78,
        open_set_auroc=0.75,
    )
    run_4 = _write_run(
        tmp_path,
        run_id="seed_bnat_seed45",
        dataset="B-NAT",
        source_labels=["Normal", "BP", "DoS", "MitM", "FoT"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="FedTROS",
        num_clients=3,
        seed=45,
        alpha=0.1,
        test_accuracy=0.89,
        macro_f1=0.81,
        open_set_auroc=0.78,
    )
    run_5 = _write_run(
        tmp_path,
        run_id="ablation_full",
        dataset="B-NAT",
        source_labels=["Normal", "BP", "DoS", "MitM", "FoT"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="Proposed (FL + OSR)",
        num_clients=3,
        seed=42,
        alpha=0.1,
        test_accuracy=0.92,
        macro_f1=0.94,
        open_set_auroc=0.81,
    )
    run_6 = _write_run(
        tmp_path,
        run_id="ablation_fedavg",
        dataset="B-NAT",
        source_labels=["Normal", "BP", "DoS", "MitM", "FoT"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="Base + FL (FedAvg)",
        num_clients=3,
        seed=42,
        alpha=0.1,
        test_accuracy=0.84,
        macro_f1=0.70,
        open_set_auroc=0.69,
    )
    run_7 = _write_run(
        tmp_path,
        run_id="ablation_central_osr",
        dataset="B-NAT",
        source_labels=["Normal", "BP", "DoS", "MitM", "FoT"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="Base + OSR (Centralized)",
        num_clients=1,
        seed=42,
        alpha=0.1,
        test_accuracy=0.90,
        macro_f1=0.91,
        open_set_auroc=0.80,
    )
    run_8 = _write_run(
        tmp_path,
        run_id="ablation_central_no_osr",
        dataset="B-NAT",
        source_labels=["Normal", "BP", "DoS", "MitM", "FoT"],
        known_labels=["Normal", "BP", "DoS", "MitM"],
        method="Base Model (Centralized, No OSR)",
        num_clients=1,
        seed=42,
        alpha=0.1,
        test_accuracy=0.76,
        macro_f1=0.72,
        open_set_auroc=0.60,
    )

    suite_dir = tmp_path / "suite"
    compare_cfg = OmegaConf.create({"runs": [str(run_1), str(run_2), str(run_3), str(run_4), str(run_5), str(run_6), str(run_7), str(run_8)], "run_dir": str(suite_dir)})
    compare_runs(compare_cfg, project_root=tmp_path)

    generated = build_suite_artifacts(
        run_dirs=[run_1, run_2, run_3, run_4, run_5, run_6, run_7, run_8],
        output_dir=suite_dir,
        project_root=tmp_path,
    )

    expected_files = {
        "scalability.csv",
        "openness_metrics.csv",
        "cross_dataset_metrics.csv",
        "seed_robustness.csv",
        "latent_embeddings.csv",
        "communication_metrics.csv",
        "ablation_metrics.csv",
        "suite_artifacts_manifest.json",
        "comparison_metrics.csv",
    }
    assert expected_files.issubset({path.name for path in generated.values()})
    for filename in expected_files:
        assert (suite_dir / filename).exists()

    scalability = pd.read_csv(suite_dir / "scalability.csv")
    assert {3, 10, 20}.issubset(set(scalability["num_clients"].tolist()))

    openness = pd.read_csv(suite_dir / "openness_metrics.csv")
    assert {"method", "openness", "auroc"}.issubset(openness.columns)

    cross_dataset = pd.read_csv(suite_dir / "cross_dataset_metrics.csv")
    assert {"dataset", "metric", "metric_value"}.issubset(cross_dataset.columns)

    latent = pd.read_csv(suite_dir / "latent_embeddings.csv")
    assert {"x", "y", "label"}.issubset(latent.columns)
