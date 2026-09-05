import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def test_bundle_contract_constants_match_plots_repo_when_present():
    # FedTROS side contract is source-level and intentionally file-based.
    exporter = Path(__file__).resolve().parents[1] / "scripts/export_publication_bundle.py"
    text = exporter.read_text(encoding="utf-8")
    assert 'SCHEMA_NAME="fedtros_mc_publication_bundle"' in text
    assert 'SCHEMA_VERSION=2' in text
    assert '"method":"FedTROS-MC"' in text
