from pathlib import Path
from omegaconf import OmegaConf

from src.experiment.result_store import ResultStore
from src.infrastructure.tracking.factory import create_tracker
from src.infrastructure.tracking.null_tracker import NullTracker

ROOT = Path(__file__).resolve().parents[1]


def test_disabled_tracking_uses_null_tracker(tmp_path):
    cfg = OmegaConf.create({"tracking": {"backend": "wandb", "mode": "disabled"}})
    tracker = create_tracker(cfg, run_dir=tmp_path, run_id="test_run", study_id="E0-VERIFY")
    assert isinstance(tracker, NullTracker)
    assert tracker.run_id == "test_run"


def test_result_store_is_independent_of_tracker(tmp_path):
    store = ResultStore(tmp_path / "run", "run_1")
    store.append_round_metrics({"closed_set/accuracy": 0.5}, step=1)
    store.save_final_metrics({"closed_set/accuracy": 0.6})
    store.finalize_result_manifest(status="COMPLETED", final_metrics={"closed_set/accuracy": 0.6})
    assert (tmp_path / "run/metrics/round_metrics.csv").exists()
    assert (tmp_path / "run/metrics/final_metrics.json").exists()
    assert (tmp_path / "run/result_manifest.json").exists()


def test_artifact_inventory_excludes_mutable_logs_and_manifests(tmp_path):
    store = ResultStore(tmp_path / "run", "run_1")
    (store.run_dir / "logs").mkdir(exist_ok=True)
    (store.run_dir / "logs" / "run.log").write_text("running", encoding="utf-8")
    (store.run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    store.save_final_metrics({"closed_set/accuracy": 0.6})

    inventory = store.artifact_inventory()

    assert "metrics/final_metrics.json" in inventory
    assert "logs/run.log" not in inventory
    assert "run_manifest.json" not in inventory


def test_resume_config_does_not_overwrite_frozen_resolved_config(tmp_path):
    store = ResultStore(tmp_path / "run", "run_1")
    original = OmegaConf.create({"federated": {"num_rounds": 100}, "seed": 42})
    store.save_config(original)
    frozen = (tmp_path / "run/resolved_config.yaml").read_text(encoding="utf-8")

    resumed = OmegaConf.create({"federated": {"num_rounds": 35}, "seed": 42})
    resume_path = store.save_resume_config(resumed, resumed_from_round=65)

    assert resume_path.name == "resume_from_round_0065.yaml"
    assert resume_path.exists()
    assert (tmp_path / "run/resolved_config.yaml").read_text(encoding="utf-8") == frozen


def test_only_one_active_tracker_backend():
    active = ROOT / "src/infrastructure/tracking"
    names = {p.name for p in active.glob("*.py")}
    assert "wandb_tracker.py" in names
    assert "mlflow_tracker.py" not in names
    assert "composite_tracker.py" not in names
    assert "local_tracker.py" not in names
    assert not (ROOT / "src/tracking").exists()


def test_federated_round_metrics_use_generic_sink_not_wandb_imports():
    server = (ROOT / "src/federated/server.py").read_text(encoding="utf-8")
    run = (ROOT / "src/federated/run.py").read_text(encoding="utf-8")
    assert "import wandb" not in server
    assert "_emit_round_metrics" in server
    assert "metrics_sink=tracker" in run
