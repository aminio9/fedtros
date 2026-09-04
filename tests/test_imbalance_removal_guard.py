"""Guard tests: Ensure all class-imbalance handling mechanisms are completely removed."""

from pathlib import Path
import pytest
from src.utils.utils import project_root

FORBIDDEN_SRC_TERMS = [
    "class_balanced_cross_entropy",
    "effective_number_class_weights",
    "EffectiveNumberClassBalance",
    "class_balance_beta",
    "class_weight_min",
    "class_weight_max",
    "L_CBCE",
]

FORBIDDEN_CONFIG_TERMS = [
    "class_balance_beta",
    "class_weight_min",
    "class_weight_max",
    "use_cb_loss",
    "imbalance_factor",
]


def test_no_class_balance_file():
    """Verify src/training/class_balance.py is completely deleted."""
    cb_path = project_root() / "src" / "training" / "class_balance.py"
    assert not cb_path.exists(), f"class_balance.py exists at {cb_path}; it must be removed."


def test_no_class_imbalance_terms_in_src():
    """Verify no class imbalance functions, classes, or parameters in src/."""
    src_dir = project_root() / "src"
    py_files = list(src_dir.glob("**/*.py"))
    assert len(py_files) > 0, "No python files found in src/"

    violations = []
    for py_file in py_files:
        text = py_file.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), 1):
            for term in FORBIDDEN_SRC_TERMS:
                if term in line:
                    violations.append(f"{py_file.name}:{idx} -> '{term}'")

    assert not violations, "Found leftover class-imbalance terms in src/: " + ", ".join(violations)


def test_no_class_imbalance_in_configs():
    """Verify YAML configurations do not contain class imbalance parameters."""
    cfg_dir = project_root() / "src" / "configs"
    yaml_files = list(cfg_dir.glob("**/*.yaml"))
    assert len(yaml_files) > 0, "No yaml files found in src/configs/"

    violations = []
    for yaml_file in yaml_files:
        text = yaml_file.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), 1):
            for term in FORBIDDEN_CONFIG_TERMS:
                if term in line:
                    violations.append(f"{yaml_file.name}:{idx} -> '{term}'")

    assert not violations, "Found leftover class-imbalance terms in configs: " + ", ".join(violations)
