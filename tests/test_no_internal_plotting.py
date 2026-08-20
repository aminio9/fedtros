from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("matplotlib", "seaborn", "plotly", "pyplot")


def test_no_internal_plotting_imports():
    hits = []
    for path in (ROOT / "src").rglob("*.py"):
        if "archive" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in FORBIDDEN:
            if f"import {name}" in text or f"from {name}" in text:
                hits.append((str(path.relative_to(ROOT)), name))
    assert not hits, f"FedTROS scientific source must not render figures: {hits}"


def test_no_plotting_config_group():
    cfg = (ROOT / "src/configs/config.yaml").read_text(encoding="utf-8")
    assert "plotting:" not in cfg
    assert "figures_dir" not in cfg
