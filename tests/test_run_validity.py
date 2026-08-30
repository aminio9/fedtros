import json

from src.analysis.loaders import load_run
from src.analysis.query import query_runs
from scripts.audit_fedtros_runs import audit_run


def test_fedtros_mc_fedavg_fallback_is_marked_invalid_and_filtered(tmp_path):
    run_dir = tmp_path / "runs" / "bad_run"
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "metadata" / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "bad_run",
                "status": "COMPLETED",
                "method": "FedTROS-MC",
                "method_id": "fedtros_mc",
                "study_id": "E1-IID-CS",
                "stage": "paper_final",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "logs" / "run.log").write_text(
        "--- Strategy: FedAvg with Model Saving ---\n"
        "Standard baseline local training\n",
        encoding="utf-8",
    )

    reasons = audit_run(run_dir)

    assert reasons == ["fedtros_mc_dispatched_to_fedavg"]
    record = load_run(run_dir)
    assert record.validity_status == "INVALID"
    assert query_runs(outputs_dir=tmp_path) == []
    assert [r.run_id for r in query_runs(outputs_dir=tmp_path, include_invalid=True)] == ["bad_run"]

