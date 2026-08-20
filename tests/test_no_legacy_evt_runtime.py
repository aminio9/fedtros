from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_active_evt_runtime_module_or_imports():
    assert not (ROOT / "src" / "openset" / "evt.py").exists()
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "src.openset.evt" in text or "EVTModel" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Legacy EVT runtime references remain: {offenders}"


def test_no_active_evt_config_block():
    offenders = []
    for path in (ROOT / "src" / "configs").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "\n  evt:" in text or "\nevt:" in text or "evt_dir:" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Legacy EVT config remains active: {offenders}"
