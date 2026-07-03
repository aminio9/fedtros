"""Lightweight source-level guard for FMRL-AVA local proximal support.

Importing the Flower client is intentionally avoided here because full Flower
construction is too heavy for minimal unit tests and may be absent in cheap CI.
"""

from pathlib import Path


def test_fmrl_ava_fit_uses_strategy_local_proximal_mu():
    source = Path("src/federated/client.py").read_text()

    assert 'elif strategy_name == "fmrl_ava"' in source
    assert 'getattr(self.cfg.strategy, "local_proximal_mu", 0.0)' in source
    assert "_perform_training_loop(proximal_mu=proximal_mu)" in source


def test_fmrl_ava_config_declares_local_proximal_mu():
    fmrl_cfg = Path("src/configs/method/fmrl_ava.yaml").read_text()
    glow_cfg = Path("src/configs/method/fmrl_ava_glow.yaml").read_text()

    assert "local_proximal_mu: 0.001" in fmrl_cfg
    assert "local_proximal_mu: 0.001" in glow_cfg
