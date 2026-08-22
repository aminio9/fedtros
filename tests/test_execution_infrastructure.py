"""Comprehensive execution infrastructure tests for FedTROS-PR (Item B30)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.infrastructure.checkpointing import (
    CheckpointState,
    IncompatibleCheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from src.infrastructure.hardware import get_full_environment_provenance
from src.infrastructure.instrumentation import (
    CommunicationTracker,
    RuntimeTracker,
)
from src.infrastructure.logging import configure_logging
from src.infrastructure.manifests import (
    RunStatus,
    create_initial_run_manifest,
    initialize_run_directories,
    update_run_manifest_status,
)
from src.infrastructure.run_id import (
    RunCollisionError,
    compute_scientific_config_hash,
    generate_run_id,
    validate_run_collision,
)
from src.infrastructure.study import (
    CANONICAL_SEEDS,
    DryRunSummary,
    PlannedRun,
    expand_study_matrix,
    filter_missing_runs,
    get_paired_partition_path,
    load_study_config,
    perform_dry_run,
)
from src.infrastructure.tracking import (
    NullTracker,
    create_tracker,
)
from src.models.bundle import FedTROSModelBundle
from src.models.student import StudentIDSModel
from src.models.variational_teacher import VariationalClassifierTeacher


@pytest.fixture
def dummy_cfg():
    return OmegaConf.create({
        "seed": 42,
        "stage": "development",
        "experiment": {
            "id": "E0-VERIFY",
            "name": "E0_verify",
            "method": "FedTROS-PR",
            "pipeline": "full",
        },
        "strategy": {
            "name": "fedtros_pr",
            "student_aggregation_mode": "reliability_weighted_average",
        },
        "dataset": {
            "name": "bnat",
            "preprocessing": {
                "known_labels": ["Normal", "BP", "DoS", "MitM"],
                "unknown_labels": ["FoT"],
                "alpha": 0.1,
                "iid": False,
                "num_clients": 10,
            },
        },
        "model": {
            "name": "fedtros",
            "latent_dim": 64,
            "hidden_dims": [512, 256, 128],
            "num_actions": 4,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "local_epochs": 1,
            "fedtros_global_anchor_weight": 0.1,
            "fedtros_student_hidden_dims": [512, 256, 128],
            "fedtros_teacher_to_student_start_round": 1,
            "fedtros_alignment_start_round": 1,
            "student_osr_enabled": True,
        },
        "federated": {
            "num_rounds": 10,
            "num_clients": 10,
        },
        "open_set": {
            "evt": {
                "enabled": True,
                "backend": "fedtros_pr",
            },
            "fedtros_osr": {
                "enabled": True,
            },
            "prototype_rank": {
                "top_k": 3,
                "temperature": 1.0,
            },
        },
        "evaluation": {
            "mode": "open_set",
        },
        "logging": {
            "level": "INFO",
            "debug_level": "DEBUG",
        },
        "tracking": {
            "backend": "local",
        },
    })


def test_study_composition():
    """B30.1: Test that all study YAML configs are well-formed and loadable."""
    project_root = Path(__file__).resolve().parent.parent
    study_dir = project_root / "configs" / "study"
    assert study_dir.exists(), "configs/study directory must exist"

    expected_studies = [
        "E0_verify", "E1_iid_closed", "E2_iid_osr", "E3_noniid_closed",
        "E4_noniid_fosr", "E5_datasetwise", "E6_scalability", "E7_efficiency",
        "E8_leave_one_attack_out", "A1_teacher", "A2_anchor", "A3_transfer",
        "A4_prototype_rank", "A5_feature_source", "S1_sensitivity"
    ]

    for study_name in expected_studies:
        cfg = load_study_config(study_name, project_root=project_root)
        assert "study_id" in cfg or "name" in cfg
        assert "methods" in cfg
        assert "datasets" in cfg
        assert len(cfg["methods"]) > 0


def test_study_matrix_expansion(dummy_cfg):
    """B30.2: Test combinatorial matrix expansion across methods, alphas, and seeds."""
    study_cfg = {
        "study_id": "E4-NIID-FOSR",
        "methods": ["fedtros_pr", "fedavg"],
        "datasets": ["bnat"],
        "alphas": [0.1, 0.5],
        "iids": [False],
        "unknown_label_sets": [["FoT"]],
        "seeds": [17, 42],
        "num_clients": 10,
    }
    planned = expand_study_matrix(study_cfg, stage="development")
    # 2 methods * 1 dataset * 2 alphas * 1 iid * 1 unk * 2 seeds = 8 runs
    assert len(planned) == 8
    for run in planned:
        assert isinstance(run, PlannedRun)
        assert run.study_id == "E4-NIID-FOSR"
        assert run.seed in (17, 42)
        assert run.alpha in (0.1, 0.5)
        assert len(run.run_id) > 0


def test_run_id_generation(dummy_cfg):
    """B30.3: Test deterministic run ID slug and human name generation."""
    run_id, human_name, config_hash = generate_run_id(dummy_cfg, study_id="E4-NIID-FOSR")
    assert "e4" in run_id.lower()
    assert "bnat" in run_id.lower()
    assert "s42" in run_id.lower()
    assert len(config_hash) == 64
    assert "E4-NIID-FOSR" in human_name
    assert "BNaT" in human_name or "BNAT" in human_name
    assert "s=42" in human_name


def test_dataset_display_names_and_registry_names_share_run_identity():
    base = {
        "experiment.id": "E1-IID-CS",
        "experiment.method": "FedTROS-PR",
        "dataset.preprocessing.iid": True,
        "dataset.preprocessing.unknown_labels": [],
        "federated.num_clients": 2,
        "seed": 42,
    }
    for registry, display in (
        ("bnat", "B-NAT"),
        ("btat", "B-TAT"),
        ("cicids2017", "CIC-IDS2017"),
        ("toniot", "ToN-IoT"),
    ):
        planner_id = generate_run_id({**base, "dataset": registry}, study_id="E1-IID-CS")[0]
        runtime_id = generate_run_id({**base, "dataset.name": display}, study_id="E1-IID-CS")[0]
        assert planner_id == runtime_id


def test_config_hash(dummy_cfg):
    """B30.4: Test that scientific config changes alter hash, while cosmetic changes do not."""
    base_hash = compute_scientific_config_hash(dummy_cfg)

    # Cosmetic change (logging level)
    cfg_cosmetic = OmegaConf.create(OmegaConf.to_container(dummy_cfg, resolve=True))
    cfg_cosmetic.logging.level = "DEBUG"
    assert compute_scientific_config_hash(cfg_cosmetic) == base_hash

    # Scientific change (learning rate)
    cfg_scientific = OmegaConf.create(OmegaConf.to_container(dummy_cfg, resolve=True))
    cfg_scientific.training.learning_rate = 0.005
    assert compute_scientific_config_hash(cfg_scientific) != base_hash


def test_duplicate_run_detection(dummy_cfg, tmp_path):
    """B30.5: Test that conflicting config hash on existing run directory raises RunCollisionError."""
    run_id, _, config_hash = generate_run_id(dummy_cfg)
    run_dir = tmp_path / "runs" / run_id
    initialize_run_directories(run_dir)

    manifest = create_initial_run_manifest(
        dummy_cfg,
        run_id=run_id,
        study_id="E0",
        stage="development",
        config_hash=config_hash,
        project_root=tmp_path,
    )
    manifest.save(run_dir)

    # Same hash: valid
    validate_run_collision(run_dir, config_hash)

    # Different hash: must raise
    with pytest.raises(RunCollisionError):
        validate_run_collision(run_dir, "mismatched_different_hash_123456")


def test_status_transition(dummy_cfg, tmp_path):
    """B30.6: Test lifecycle status updates."""
    run_dir = tmp_path / "runs" / "test_run_status"
    initialize_run_directories(run_dir)
    manifest = create_initial_run_manifest(
        dummy_cfg,
        run_id="test_run_status",
        study_id="E0",
        stage="development",
        config_hash="abc123hash",
        project_root=tmp_path,
    )
    assert manifest.status == RunStatus.CREATED.value
    manifest.save(run_dir)

    update_run_manifest_status(run_dir, RunStatus.RUNNING)
    loaded = json.loads((run_dir / "metadata" / "run_manifest.json").read_text(encoding="utf-8"))
    assert loaded["status"] == RunStatus.RUNNING.value

    update_run_manifest_status(run_dir, RunStatus.COMPLETED)
    loaded_done = json.loads((run_dir / "metadata" / "run_manifest.json").read_text(encoding="utf-8"))
    assert loaded_done["status"] == RunStatus.COMPLETED.value
    assert loaded_done["finished_at"] is not None


def test_output_creation(dummy_cfg, tmp_path):
    """B30.7: Test standardized output directory layout."""
    run_dir = tmp_path / "runs" / "contract_run"
    subdirs = initialize_run_directories(run_dir)
    for expected in ("config", "logs", "metrics", "checkpoints", "predictions", "artifacts", "metadata"):
        assert expected in subdirs
        assert (run_dir / expected).exists()
        assert (run_dir / expected).is_dir()


def test_tracker_initialization(dummy_cfg, tmp_path):
    """B30.8: Test WandBTracker and NullTracker initialization and metrics logging."""
    run_dir = tmp_path / "runs" / "test_tracker"
    initialize_run_directories(run_dir)

    disabled_cfg = OmegaConf.create(OmegaConf.to_container(dummy_cfg, resolve=True))
    disabled_cfg.tracking.backend = "wandb"
    disabled_cfg.tracking.mode = "disabled"

    tracker = create_tracker(
        disabled_cfg,
        run_dir=run_dir,
        run_id="test_tracker",
        study_id="E0",
        stage="development",
    )
    assert tracker.tracker_id == "null"
    tracker.log_metrics({"closed_set/accuracy": 0.95, "open_set/auroc": 0.98}, step=1)
    tracker.set_summary({"final_auroc": 0.98})
    tracker.finish(status="COMPLETED")

    null_tracker = NullTracker(run_id="null_test")
    assert null_tracker.tracker_id == "null"
    null_tracker.log_metrics({"loss": 0.5})
    null_tracker.finish()


def test_unknown_label_validation(dummy_cfg):
    """B30.9: Test that overlapping known and unknown labels raise ValueError."""
    from scripts.run import validate_experiment_config

    invalid_cfg = OmegaConf.create(OmegaConf.to_container(dummy_cfg, resolve=True))
    invalid_cfg.dataset.preprocessing.known_labels = ["Normal", "BP", "FoT"]
    invalid_cfg.dataset.preprocessing.unknown_labels = ["FoT"]

    with pytest.raises(ValueError, match="Known and unknown label sets must be strictly disjoint"):
        validate_experiment_config(invalid_cfg)


def test_resume_compatibility(dummy_cfg, tmp_path):
    """B30.10: Test Schema v2 checkpoint loading and rejection of legacy DQN checkpoints."""
    student = StudentIDSModel(input_dim=10, num_classes=4, hidden_dims=[32, 16])
    teacher = VariationalClassifierTeacher(input_dim=10, num_classes=4, latent_dim=16)

    class DummyAgent:
        def __init__(self):
            self.student_model = student
            self.teacher = teacher
            self.student_anchor_model = None
            self.teacher_to_student_aligner = None
            self.optimizer_student = None
            self.optimizer_teacher = None

    agent = DummyAgent()
    state = CheckpointState(epoch=5, global_step=50, metrics={"open_set/auroc": 0.92}, round_num=5)

    ckpt_path = tmp_path / "checkpoints" / "v2_test.pt"
    save_checkpoint(agent, dummy_cfg, ckpt_path, state, config_hash="hash123")
    assert ckpt_path.exists()

    # Load valid Schema v2
    loaded = load_checkpoint(agent, ckpt_path, device="cpu")
    assert loaded["schema_version"] == 2
    assert loaded["round"] == 5

    # Create dummy legacy DQN checkpoint
    dqn_path = tmp_path / "checkpoints" / "legacy_dqn.pt"
    dqn_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prior_net": torch.randn(10, 5),
        "value_net_main": torch.randn(5, 2),
        "round": 3,
    }, dqn_path)

    # Must raise IncompatibleCheckpointError
    with pytest.raises(IncompatibleCheckpointError):
        load_checkpoint(agent, dqn_path, device="cpu")


def test_dry_run_no_gpu(dummy_cfg, tmp_path):
    """B30.11: Test that dry run produces execution summary without allocating GPU."""
    study_cfg = {
        "study_id": "E0-VERIFY",
        "methods": ["fedtros_pr"],
        "datasets": ["bnat"],
        "alphas": [0.1],
        "iids": [False],
        "unknown_label_sets": [["FoT"]],
        "seeds": [17, 42],
        "num_clients": 10,
    }
    planned = expand_study_matrix(study_cfg, stage="smoke")
    summary = perform_dry_run(planned, output_base_dir=tmp_path)
    assert isinstance(summary, DryRunSummary)
    assert summary.total_runs == 2
    assert summary.new_runs == 2
    assert summary.completed_runs == 0


def test_communication_instrumentation(tmp_path):
    """B30.12: Test communication tracker recording tensors and computing summaries."""
    comm = CommunicationTracker(output_dir=tmp_path)
    tensor1 = torch.randn(64, 32)
    tensor2 = torch.randn(32, 4)

    comm.record_tensor(
        round_num=1,
        client_id="client_0",
        direction="uplink",
        tensor_name="fc1.weight",
        tensor=tensor1,
    )
    comm.record_tensor(
        round_num=1,
        client_id="server",
        direction="downlink",
        tensor_name="fc2.weight",
        tensor=tensor2,
    )

    summary = comm.end_round(round_num=1)
    assert summary["communication/downlink_bytes"] == 32 * 4 * 4  # 32*4 float32
    assert summary["communication/uplink_bytes"] == 64 * 32 * 4   # 64*32 float32
    assert summary["communication/round_bytes"] > 0
    assert (tmp_path / "metrics" / "communication_round.csv").exists()


def test_timing_instrumentation(tmp_path):
    """B30.13: Test runtime timing measurement and output writing."""
    runtime = RuntimeTracker(output_dir=tmp_path)
    runtime.start_round(1)

    with runtime.time_stage("teacher_training"):
        _ = sum(i * i for i in range(1000))

    with runtime.time_stage("student_training"):
        _ = sum(i * i for i in range(1000))

    summary = runtime.end_round(total_round_seconds=0.05)
    assert summary["runtime/round_seconds"] == 0.05
    assert (tmp_path / "metrics" / "timing_round.csv").exists()


def test_paired_partition_path(tmp_path):
    """B30.14: Test paired partition contract paths."""
    path_alpha = get_paired_partition_path(tmp_path, "bnat", 0.1, 42, iid=False)
    assert path_alpha.name.startswith("alpha_0p1_")
    assert path_alpha.name.endswith("_seed_42.json")
    assert "partitions" in path_alpha.parts

    path_iid = get_paired_partition_path(tmp_path, "bnat", 1.0, 42, iid=True)
    assert path_iid.name.startswith("iid_")
    assert path_iid.name.endswith("_seed_42.json")
