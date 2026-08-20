#!/usr/bin/env python3
"""Environment and repository health checks for FedTROS-PR GPU execution."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _status(level: str, name: str, detail: str) -> tuple[str, str, str]:
    return level, name, detail


def _import_check(module: str, required: bool = True) -> tuple[str, str, str]:
    try:
        mod = importlib.import_module(module)
        return _status("PASS", module, str(getattr(mod, "__version__", "installed")))
    except Exception as exc:
        return _status("FAIL" if required else "WARN", module, f"{type(exc).__name__}: {exc}")


def _no_internal_plotting() -> tuple[str, str, str]:
    forbidden = ("matplotlib", "seaborn", "plotly")
    hits: list[str] = []
    for source_root in (_ROOT / "src", _ROOT / "scripts"):
        for p in source_root.rglob("*.py"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            if any(f"import {name}" in text or f"from {name}" in text for name in forbidden):
                hits.append(str(p.relative_to(_ROOT)))
    return _status("PASS" if not hits else "FAIL", "FedTROS internal plotting", "none" if not hits else ", ".join(hits))


def _study_check() -> tuple[str, str, str]:
    study_dir = _ROOT / "src" / "configs" / "study"
    expected = {"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "A1", "A2", "A3", "A4", "A5", "S1"}
    stems = {p.stem.split("_")[0].upper() for p in study_dir.glob("*.yaml")}
    missing = sorted(expected - stems)
    return _status("PASS" if not missing else "FAIL", "Study contract", "all canonical studies present" if not missing else f"missing={missing}")


def _output_writable(output: Path) -> tuple[str, str, str]:
    try:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output, prefix="doctor_", delete=True) as handle:
            handle.write(b"ok"); handle.flush()
        return _status("PASS", "Output directory", str(output.resolve()))
    except Exception as exc:
        return _status("FAIL", "Output directory", str(exc))


def _wandb_check(mode: str) -> tuple[str, str, str]:
    if mode == "disabled":
        return _status("PASS", "W&B", "disabled mode requested; local ResultStore remains authoritative")
    try:
        import wandb  # noqa: F401
    except Exception as exc:
        return _status("FAIL", "W&B", f"package unavailable: {exc}")
    if mode == "offline":
        return _status("PASS", "W&B", "offline mode available")
    key_present = bool(os.environ.get("WANDB_API_KEY")) or (Path.home() / ".netrc").exists()
    return _status("PASS" if key_present else "WARN", "W&B", "authentication detected" if key_present else "package installed; authenticate with `wandb login` before online runs")


def _plot_repo_check(path: Path | None) -> tuple[str, str, str]:
    if path is None:
        return _status("WARN", "Plots repository", "not supplied; use --plots-repo to validate publication integration")
    required = [path / "scripts" / "generate_all.py", path / "src" / "data" / "fedtros_bundle.py"]
    missing = [str(p) for p in required if not p.exists()]
    return _status("PASS" if not missing else "FAIL", "Plots repository", str(path.resolve()) if not missing else f"missing={missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check FedTROS-PR environment before server experiments.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "outputs")
    parser.add_argument("--plots-repo", type=Path, default=None)
    args = parser.parse_args()

    checks: list[tuple[str, str, str]] = []
    py_ok = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.append(_status("PASS" if py_ok else "FAIL", "Python", f"{sys.version.split()[0]} (project requires >=3.11,<3.13)"))
    for module in ("torch", "hydra", "omegaconf", "flwr", "numpy", "pandas", "sklearn"):
        checks.append(_import_check(module))
    checks.append(_wandb_check(args.wandb_mode))
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        detail = "CUDA available" if cuda else "CUDA unavailable; use runtime=cpu for tests or provision a GPU server"
        checks.append(_status("PASS" if cuda else "WARN", "CUDA", detail))
        if cuda:
            for idx in range(torch.cuda.device_count()):
                checks.append(_status("PASS", f"GPU {idx}", torch.cuda.get_device_name(idx)))
    except Exception as exc:
        checks.append(_status("FAIL", "CUDA", str(exc)))
    checks.append(_output_writable(args.output_dir))
    checks.append(_study_check())
    checks.append(_no_internal_plotting())
    checks.append(_plot_repo_check(args.plots_repo))

    print("\nFedTROS-PR doctor")
    print("=" * 96)
    for level, name, detail in checks:
        print(f"[{level:<4}] {name:<30} {detail}")
    print("=" * 96)
    fails = sum(1 for level, _, _ in checks if level == "FAIL")
    warns = sum(1 for level, _, _ in checks if level == "WARN")
    print(f"Result: {fails} FAIL, {warns} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
