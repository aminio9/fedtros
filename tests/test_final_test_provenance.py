import hashlib
import pandas as pd

from src.openset.multicenter_conformal_pipeline import persist_final_test_identifiers


def test_final_test_identifier_artifact_is_traceable(tmp_path):
    scores = pd.DataFrame({
        "sample_id": ["k1", "u1", "k2"],
        "known_or_unknown": ["known", "unknown", "known"],
    })
    path, digest = persist_final_test_identifiers(scores, tmp_path)
    assert path == tmp_path / "osr" / "final_test_identifiers.csv"
    assert path.exists()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    saved = pd.read_csv(path)
    assert saved.to_dict("records") == [
        {"sample_id": "k1", "partition": "known"},
        {"sample_id": "u1", "partition": "unknown"},
        {"sample_id": "k2", "partition": "known"},
    ]