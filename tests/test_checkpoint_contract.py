from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_paths_are_run_scoped():
    text = (ROOT / "src/configs/checkpointing/default.yaml").read_text(encoding="utf-8")
    assert "${tracking.run_dir}/checkpoints" in text
    assert "latest.pt" in text


def test_active_checkpoint_source_mentions_vct_schema_and_config_hash():
    server = (ROOT / "src/federated/server.py").read_text(encoding="utf-8")
    assert '"schema_version": 2' in server
    assert '"method_id": "fedtros_pr"' in server
    assert '"teacher_type": "variational_classifier"' in server
    assert '"config_hash"' in server
    assert '"rng_state"' in server
    assert "self.cfg.checkpointing.latest_checkpoint_path" in server
