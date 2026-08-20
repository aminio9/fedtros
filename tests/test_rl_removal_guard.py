"""CI Guard Test: Ensure all Reinforcement Learning and Q-network terms are completely removed."""

from pathlib import Path
import pytest

from src.utils.utils import project_root

FORBIDDEN_TERMS = [
    "BlockchainIntrusionEnv",
    "ExperienceReplayBuffer",
    "EpsilonGreedyPolicy",
    "EpsilonScheduler",
    "PriorNetwork",
    "RecognitionNetwork",
    "MainQNetwork",
    "TargetQNetwork",
    "GenerationNetwork",
    "value_net_main",
    "value_net_target",
    "optimizer_q_rl",
    "optimizer_prior",
    "use_double_dqn",
    "prior_net",
    "recognition_net",
    "generation_net",
]

SRC_DIR = project_root() / "src"


def test_no_forbidden_rl_terms_in_src():
    """Verify that no forbidden RL classes, attributes, or functions exist in src/."""
    violations = []
    py_files = list(SRC_DIR.glob("**/*.py"))

    assert len(py_files) > 0, "No python files found in src/"

    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        for idx, line in enumerate(lines, 1):
            # Skip pure comments
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            for term in FORBIDDEN_TERMS:
                if term in line:
                    violations.append(f"{py_file.relative_to(project_root())}:{idx} -> contains '{term}'")

    assert not violations, "Found leftover RL/DQN terms in src/:\n" + "\n".join(violations)


def test_vct_and_student_present():
    """Verify that canonical VCT and Student models are present and importable."""
    from src.models.variational_teacher import VariationalClassifierTeacher
    from src.models.student import StudentIDSModel
    from src.models.bundle import FedTROSModelBundle, Agent

    assert VariationalClassifierTeacher is not None
    assert StudentIDSModel is not None
    assert FedTROSModelBundle is not None
    assert Agent is not None


def test_no_rl_directory_in_src():
    """Verify that no src/rl/ directory exists in the codebase."""
    rl_dir = project_root() / "src" / "rl"
    assert not rl_dir.exists(), f"src/rl directory exists: {rl_dir}; it must be completely removed."


def test_no_agents_directory_in_src():
    """Verify that no src/agents/ directory exists in the codebase."""
    agents_dir = project_root() / "src" / "agents"
    assert not agents_dir.exists(), f"src/agents directory exists: {agents_dir}; it must be completely removed."


def test_no_agent_configs():
    """Verify that no src/configs/agent directory exists in the codebase."""
    agent_cfg_dir = project_root() / "src" / "configs" / "agent"
    assert not agent_cfg_dir.exists(), f"src/configs/agent directory exists: {agent_cfg_dir}; it must be completely removed."
