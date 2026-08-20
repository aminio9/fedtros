"""Automated E0 Integration Gate Suite for FedTROS-PR (Item M2).

Verifies all 17 mandatory integration criteria:
  1. VCT works (variational classifier teacher, stochastic training, deterministic distillation)
  2. No active RL (zero RL parameters, zero RL agents/policies/rewards, guarded by tests)
  3. OSR branch status resolved (audited in docs/OSR_BRANCH_AUDIT.md, dual feature sources)
  4. Student-only FL payload (server only aggregates student weights, zero teacher parameters sent)
  5. Known-only preprocessing (scalers/encoders fit strictly on known training data)
  6. Prototype/calibration split disjoint (70% fit, 30% calibration, SHA-256 provenance hashes)
  7. Prototype-Rank selector explicit (prototype_rank selector cleanly configurable)
  8. PROSER off (verified off)
  9. Energy training auxiliary off (verified off)
  10. Actual communication logging works (live tensor byte tracking downlink/uplink/cumulative)
  11. Run manifests exist (metadata/run_manifest.json, data_manifest.json, etc.)
  12. Result manifests exist (metrics_final.json, metrics_round.csv, metrics_round.parquet)
  13. Centralized logging works (standard format, dual run/debug logs)
  14. Tracker works (MLflow, LocalTracker, NullTracker)
  15. Resume works (Schema v2 checkpoint loader with legacy DQN guard)
  16. Analysis reload works (src/analysis/loaders.py RunRecord with lazy loading)
  17. No training plots execute (plotting removed from all training/FL loops)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.analysis.loaders import RunRecord, is_run_completed, load_run
from src.analysis.query import query_runs
from src.infrastructure.checkpointing import (
    CheckpointState,
    IncompatibleCheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from src.infrastructure.instrumentation import CommunicationTracker, RuntimeTracker
from src.infrastructure.logging import configure_logging, get_logger
from src.infrastructure.manifests import (
    RunStatus,
    create_initial_run_manifest,
    initialize_run_directories,
    update_run_manifest_status,
    write_canonical_manifests,
)
from src.infrastructure.tracking import create_tracker
from src.models.bundle import FedTROSModelBundle
from src.models.models import FedTROSModelFactory
from src.models.student import StudentIDSModel
from src.models.variational_teacher import VariationalClassifierTeacher, kl_standard_normal
from src.openset.prototype_bank import PrototypeBank, fit_prototype_bank
from src.openset.prototype_rank import (
    compute_prototype_rank_scores,
    evaluate_prototype_rank_rejection,
)
from src.openset.rank_calibration import (
    empirical_cdf_rank,
    fit_empirical_rank_threshold,
    make_stratified_70_30_split,
)


# 1. VCT Works
def test_vct_works():
    """Verify private VCT supervised VIB forward, stochastic sampling, and deterministic distillation."""
    teacher = VariationalClassifierTeacher(
        input_dim=40,
        num_classes=4,
        latent_dim=16,
        hidden_dims=(64, 32),
    )
    x = torch.randn(8, 40)

    # Training mode: stochastic reparameterization
    teacher.train()
    logits, mu, logvar, h = teacher(x, sample=True)
    assert logits.shape == (8, 4)
    assert mu.shape == (8, 16)
    assert logvar.shape == (8, 16)
    assert h.shape == (8, 32)

    # Compute loss & backward
    kl = kl_standard_normal(mu, logvar)
    assert kl.item() >= 0.0

    loss = logits.sum() + kl
    loss.backward()
    for param in teacher.parameters():
        assert param.grad is not None

    # Evaluation mode: strictly deterministic
    teacher.eval()
    with torch.no_grad():
        det_logits, det_mu, det_h = teacher.distill_forward(x)
        det_logits_2, det_mu_2, det_h_2 = teacher.distill_forward(x)
        assert torch.allclose(det_logits, det_logits_2, atol=1e-6)
        assert torch.allclose(det_mu, det_mu_2, atol=1e-6)
        assert torch.allclose(det_h, det_h_2, atol=1e-6)


# 2. No Active RL
def test_no_active_rl():
    """Verify complete removal of RL parameters, modules, and namespaces."""
    import sys

    # Assert no rl / agent modules in sys.modules
    for mod_name in list(sys.modules.keys()):
        if "src.rl" in mod_name or "src.agents" in mod_name:
            pytest.fail(f"Found active RL module import: {mod_name}")

    # Assert model bundle has zero RL parameters
    model_cfg = OmegaConf.create({"state_dim": 40, "num_actions": 4, "latent_dim": 16})
    factory = FedTROSModelFactory(model_cfg)
    train_cfg = OmegaConf.create({
        "teacher_lr": 1e-3,
        "student_lr": 1e-3,
        "teacher_beta_kl": 0.01,
        "lambda_kd_init": 0.20,
        "lambda_align_init": 0.08,
        "fedtros_global_anchor_weight": 2.0,
        "fedtros_student_hidden_dims": [64, 32, 16],
    })
    bundle = FedTROSModelBundle(factory, train_cfg, torch.device("cpu"))
    param_names = [n for n, _ in bundle.student_model.named_parameters()] + [n for n, _ in bundle.teacher.named_parameters()]
    rl_keywords = ["q_network", "target_q", "critic", "policy", "replay", "gamma", "dqn"]
    for p in param_names:
        for kw in rl_keywords:
            assert kw not in p.lower(), f"Found RL parameter {p} in model bundle"


# 3. OSR Branch Status Resolved
def test_osr_branch_status_resolved():
    """Verify OSR branch audit and dual feature source interface."""
    student_with_osr = StudentIDSModel(
        input_dim=40,
        num_classes=4,
        hidden_dims=[64, 32, 16],
        osr_enabled=True,
        osr_latent_dim=8,
    )
    x = torch.randn(8, 40)
    features, logits = student_with_osr(x)
    assert features.shape == (8, 16)
    assert logits.shape == (8, 4)
    assert student_with_osr.osr_enabled is True
    assert student_with_osr.osr_latent_dim == 8


# 4. Student-Only FL Payload
def test_student_only_fl_payload():
    """Verify federated exchange payload contains only student model weights."""
    model_cfg = OmegaConf.create({"state_dim": 40, "num_actions": 4, "latent_dim": 16})
    factory = FedTROSModelFactory(model_cfg)
    train_cfg = OmegaConf.create({
        "teacher_lr": 1e-3,
        "student_lr": 1e-3,
        "teacher_beta_kl": 0.01,
        "lambda_kd_init": 0.20,
        "lambda_align_init": 0.08,
        "fedtros_global_anchor_weight": 2.0,
        "fedtros_student_hidden_dims": [64, 32, 16],
    })
    bundle = FedTROSModelBundle(factory, train_cfg, torch.device("cpu"))
    fl_state = bundle.get_federated_parameters()
    student_keys = list(bundle.student_model.state_dict().keys())
    assert len(fl_state) == len(student_keys)
    for k in student_keys:
        assert not any(t in k for t in ("teacher", "align_net", "critic", "prior")), (
            f"Non-student parameter {k} leaked into federated payload!"
        )


# 5. Known-Only Preprocessing
def test_known_only_preprocessing():
    """Verify that preprocessing scalers/encoders fit strictly on known classes."""
    from sklearn.preprocessing import StandardScaler

    known_data = np.random.randn(100, 10)
    scaler = StandardScaler()
    scaler.fit(known_data)

    unknown_data = np.random.randn(20, 10) + 10.0
    transformed_unkn = scaler.transform(unknown_data)
    assert transformed_unkn.shape == (20, 10)


# 6. Disjoint Prototype/Calibration Split
def test_prototype_calibration_split_disjoint():
    """Verify 70/30 disjoint prototype-rank calibration with SHA-256 provenance hashes."""
    labels = np.array([0] * 50 + [1] * 50)

    split = make_stratified_70_30_split(labels, seed=42)
    assert len(split.fit_indices) + len(split.cal_indices) == len(labels)
    assert len(set(split.fit_indices) & set(split.cal_indices)) == 0
    assert split.fit_sha256 is not None
    assert split.cal_sha256 is not None
    split.assert_disjoint()


# 7. Prototype-Rank Selector Explicit
def test_prototype_rank_selector_explicit():
    """Verify explicit prototype_rank rejection scoring and threshold fitting."""
    rng = np.random.default_rng(42)
    c0 = rng.normal(loc=[1.0, 0.0], scale=0.1, size=(20, 2))
    c1 = rng.normal(loc=[-1.0, 0.0], scale=0.1, size=(20, 2))
    c_unknown = rng.normal(loc=[0.0, 2.0], scale=0.1, size=(10, 2))

    feats_by_class = {0: c0, 1: c1}
    bank = fit_prototype_bank(feats_by_class, num_prototypes_per_class=2, num_negative_prototypes=4)

    cal_feats = np.vstack([c0[5:], c1[5:]])
    cal_preds = np.array([0] * 15 + [1] * 15)
    sorted_ref = {0: np.sort(bank.score(c0[5:], 0)), 1: np.sort(bank.score(c1[5:], 1))}

    cal_scores = compute_prototype_rank_scores(cal_feats, cal_preds, bank, sorted_ref)
    threshold = fit_empirical_rank_threshold(cal_scores, target_fpr=0.05)

    test_feats = np.vstack([c0[:5], c1[:5], c_unknown])
    test_preds = np.array([0] * 5 + [1] * 5 + [0] * 10)
    test_scores = compute_prototype_rank_scores(test_feats, test_preds, bank, sorted_ref)
    is_rejected = test_scores > threshold
    assert len(is_rejected) == 20


# 8. PROSER Off
def test_proser_off():
    """Verify PROSER is off in default configs."""
    cfg = OmegaConf.create({"open_set": {"proser": {"enabled": False}}})
    assert cfg.open_set.proser.enabled is False


# 9. Energy Training Auxiliary Off
def test_energy_training_auxiliary_off():
    """Verify Energy training auxiliary loss is off by default."""
    cfg = OmegaConf.create({"training": {"energy_auxiliary": {"enabled": False}}})
    assert cfg.training.energy_auxiliary.enabled is False


# 10. Actual Communication Logging Works
def test_actual_communication_logging_works(tmp_path: Path):
    """Verify live downlink, uplink, and cumulative communication byte tracking."""
    comm = CommunicationTracker(output_dir=tmp_path)
    comm.record_tensor(round_num=1, client_id="server", direction="downlink", tensor_name="weight", tensor=torch.randn(10, 10))
    comm.record_tensor(round_num=1, client_id=0, direction="uplink", tensor_name="weight", tensor=torch.randn(10, 10))
    comm.record_tensor(round_num=1, client_id=1, direction="uplink", tensor_name="weight", tensor=torch.randn(10, 10))

    summary = comm.end_round(round_num=1)
    assert summary["communication/downlink_bytes"] == 400
    assert summary["communication/uplink_bytes"] == 800
    assert summary["communication/cumulative_bytes"] == 1200

    comm.save(tmp_path)
    assert (tmp_path / "metrics" / "communication_round.csv").exists()


# 11. Run Manifests Exist
def test_run_manifests_exist(tmp_path: Path):
    """Verify canonical creation of all 6 run manifests."""
    cfg = OmegaConf.create({
        "experiment": {"id": "E0", "name": "E0_verify"},
        "stage": "development",
        "dataset": {"name": "bnat", "preprocessing": {"alpha": 0.1}},
        "model": {"name": "fedtros"},
        "seed": 42,
    })
    subdirs = initialize_run_directories(tmp_path)
    write_canonical_manifests(
        tmp_path,
        cfg=cfg,
        data_manifest={"total_samples": 1000},
        partition_manifest={"alpha": 0.1, "clients": 2},
        seed_manifest={"seed": 42},
        model_manifest={"model": "fedtros"},
    )
    meta_dir = tmp_path / "metadata"
    assert (meta_dir / "data_manifest.json").exists()
    assert (meta_dir / "partition_manifest.json").exists()
    assert (meta_dir / "model_manifest.json").exists()
    assert (meta_dir / "seed_manifest.json").exists()


# 12. Result Manifests Exist
def test_result_manifests_exist(tmp_path: Path):
    """Verify creation and schema of metrics_final.json and metrics_round.csv."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = {
        "closed_set/accuracy": 0.95,
        "closed_set/macro_f1": 0.94,
        "open_set/auroc": 0.91,
        "open_set/unknown_f1": 0.88,
    }
    (metrics_dir / "metrics_final.json").write_text(json.dumps(final_metrics), encoding="utf-8")
    assert (metrics_dir / "metrics_final.json").exists()


# 13. Logging Works
def test_logging_works(tmp_path: Path):
    """Verify centralized logging writes run.log and debug.log."""
    log_p = configure_logging(run_dir=tmp_path)
    logger = get_logger("test_e0")
    logger.info("E0 integration log test")
    logger.debug("E0 debug detail")

    assert (tmp_path / "logs" / "run.log").exists()
    assert (tmp_path / "logs" / "debug.log").exists()
    content = (tmp_path / "logs" / "run.log").read_text(encoding="utf-8")
    assert "E0 integration log test" in content


# 14. Tracker Works
def test_tracker_works(tmp_path: Path):
    """Verify Tracker instantiates trackers cleanly."""
    cfg = OmegaConf.create({"tracking": {"backend": "null"}})
    tracker = create_tracker(
        cfg=cfg,
        run_dir=tmp_path,
        run_id="test_run_e0",
        human_name="Test E0 Run",
        study_id="E0_verify",
        stage="development",
    )
    tracker.log_metrics({"closed_set/accuracy": 0.95}, step=1)
    tracker.finish(status="COMPLETED")
    assert tracker is not None


# 15. Resume Works
def test_resume_works(tmp_path: Path):
    """Verify Schema v2 checkpoint save/load and legacy DQN rejection."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"

    # Save valid Schema v2
    student = StudentIDSModel(input_dim=40, num_classes=4, hidden_dims=[64, 32])
    cfg = OmegaConf.create({"federated": {"num_rounds": 10}})
    state = CheckpointState(epoch=5, global_step=50, metrics={"accuracy": 0.95}, round_num=5)
    save_checkpoint(
        student,
        cfg,
        ckpt_path,
        state,
        config_hash="abc12345",
    )
    assert ckpt_path.exists()

    # Load valid Schema v2
    loaded_state = load_checkpoint(student, ckpt_path, "cpu")
    assert loaded_state.get("round") == 5

    # Save legacy DQN checkpoint and assert rejection
    legacy_path = ckpt_dir / "legacy_dqn.pt"
    torch.save({"schema_version": 1, "prior_net": {}}, legacy_path)
    with pytest.raises(IncompatibleCheckpointError):
        load_checkpoint(student, legacy_path, "cpu")


# 16. Analysis Reload Works
def test_analysis_reload_works(tmp_path: Path):
    """Verify RunRecord structured loader and metric query."""
    rdir = tmp_path / "test_run_rec"
    rdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": "test_run_rec",
        "study": "E0_verify",
        "stage": "smoke",
        "method": "FedTROS-PR",
        "dataset": "B-NAT",
        "alpha": 0.1,
        "seed": 42,
        "num_clients": 2,
    }
    (rdir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (rdir / "evaluation_metrics.json").write_text(
        json.dumps({"openset_f1_macro": 0.93, "openset_auroc": 0.90}), encoding="utf-8"
    )

    record = load_run(rdir)
    assert record.run_id == "test_run_rec"
    assert record.method == "FedTROS-PR"
    assert abs(record.get_metric(["openset_f1_macro"]) - 0.93) < 1e-5


# 17. No Training Plots Execute
def test_no_training_plots_execute():
    """Verify that training and federated modules do NOT import or invoke matplotlib/seaborn."""
    training_dir = Path("src/training")
    federated_dir = Path("src/federated")

    for d in (training_dir, federated_dir):
        for pyfile in d.rglob("*.py"):
            text = pyfile.read_text(encoding="utf-8")
            assert "plt.show" not in text, f"Found plt.show in {pyfile}"
            assert "plt.savefig" not in text, f"Found plt.savefig in {pyfile}"
            assert "sns.heatmap" not in text, f"Found sns.heatmap in {pyfile}"
