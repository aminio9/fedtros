import sys, traceback
try:
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    config_dir = str((Path('src/configs')).resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config_fl', overrides=['experiment=exp3', '+method=fmrl_ava_glow'])
    t = cfg.training
    s = cfg.federated.strategy
    assert t.rl_mode == 'contextual_bandit', f'rl_mode={t.rl_mode}'
    assert t.gamma == 0.0, f'gamma={t.gamma}'
    assert t.loss_weights.bandit_q == 1.0, f'bandit_q={t.loss_weights.bandit_q}'
    assert t.loss_weights.q_td == 0.25, f'q_td={t.loss_weights.q_td}'
    assert t.classification_loss.focal_gamma == 1.5, f'focal_gamma={t.classification_loss.focal_gamma}'
    assert t.imbalance.class_balanced_sampling == True
    assert s.server_optimizer == 'none', f'server_optimizer={s.server_optimizer}'
    assert s.local_proximal_mu == 0.001, f'local_proximal_mu={s.local_proximal_mu}'
    assert s.profile_balance_strength == 0.0
    assert s.profile_quality_blend == 0.0
    assert s.profile_cluster_strength == 0.0
    assert s.profile_min_multiplier == 1.0
    assert s.profile_max_multiplier == 1.0
    print('ALL FMRL-AVA-GLOW CONFIG ASSERTIONS PASSED')
except Exception:
    traceback.print_exc()
    sys.exit(1)
