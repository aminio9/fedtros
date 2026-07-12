from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.models.student import StudentIDSModel
from src.openset.digos_eval import calibrate_fed_digos, evaluate_fed_digos


def _cfg():
    return SimpleNamespace(
        unknown_label_id=-1,
        open_set_label_id=99,
        latent_nll_weight=0.05,
        energy=SimpleNamespace(enabled=True, temperature=1.0, rank_direction="low"),
        score_fusion=SimpleNamespace(
            method="mean_rank",
            calibration_scope="global",
            generator_score_column="recon_error",
            decision_rule="fused_rank_threshold",
        ),
        pnpff=SimpleNamespace(
            feature_dim=8,
            num_positive_prototypes=7,
            fit_fraction=0.70,
            seed=7,
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            threshold_mode="fixed",
            tau=0.5,
            checkpoint_output="pnpff_state.pt",
        ),
        prototype=SimpleNamespace(
            enabled=True, num_prototypes_per_class=2, min_samples_per_prototype=5
        ),
        evt=SimpleNamespace(
            threshold_method="quantile",
            tail_size_percent=0.20,
            min_errors_per_class=5,
            min_tail_size=3,
            target_known_fpr=0.10,
            fit_correct_only=False,
            mef_min_quantile=0.70,
            mef_max_quantile=0.98,
            mef_num_candidates=10,
        ),
    )


def test_student_osr_branch_shapes_and_no_decoder_leak():
    model = StudentIDSModel(
        input_dim=6,
        num_classes=3,
        hidden_dims=[12, 8],
        osr_enabled=True,
        osr_latent_dim=4,
        osr_hidden_dims=[10],
        osr_decoder_hidden_dims=[10],
    )
    x = torch.randn(5, 6)
    y = torch.tensor([0, 1, 2, 1, 0])
    h, logits = model(x)
    out = model.osr_score(x, y)
    assert h.shape == (5, 8)
    assert logits.shape == (5, 3)
    assert out["recon"].shape == x.shape
    assert out["score"].shape == (5,)
    assert len(list(model.osr_parameters())) > 0
    assert not any(name.startswith("decoder") for name, _ in model.named_parameters())


def test_fed_digos_eval_writes_artifacts(tmp_path):
    torch.manual_seed(3)
    model = StudentIDSModel(
        input_dim=4,
        num_classes=2,
        hidden_dims=[16, 8],
        osr_enabled=True,
        osr_latent_dim=3,
        osr_hidden_dims=[8],
        osr_decoder_hidden_dims=[8],
    )
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias[:] = torch.tensor([0.1, -0.1])
    x0 = torch.randn(30, 4) * 0.1 - 1.0
    x1 = torch.randn(30, 4) * 0.1 + 1.0
    x_cal = torch.cat([x0, x1], dim=0)
    y_cal = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    cfg = _cfg()
    models, detector, calibration_df, _meta = calibrate_fed_digos(
        x_cal, y_cal, student_model=model, batch_size=16, device=torch.device("cpu"), cfg=cfg
    )
    x_unknown = torch.randn(10, 4) + 4.0
    x_test = torch.cat([x0[:10], x1[:10], x_unknown], dim=0)
    y_test = torch.cat([torch.zeros(10), torch.ones(10), torch.full((10,), -1)]).long()
    metrics = evaluate_fed_digos(
        x_test,
        y_test,
        student_model=model,
        batch_size=8,
        device=torch.device("cpu"),
        cfg=cfg,
        class_names={0: "A", 1: "B"},
        output_dir=tmp_path,
        evt_models=models,
        pnpff_detector=detector,
        calibration_df=calibration_df,
    )
    assert "openset_auroc" in metrics
    assert "openset_oscr" in metrics
    for name in (
        "open_set_scores.csv",
        "fed_digos_evt_thresholds.json",
        "fed_digos_rank_calibration.json",
        "fed_digos_component_aurocs.json",
        "score_overlap_report.json",
        "pnpff_state.pt",
        "pnpff_metadata.json",
    ):
        assert (tmp_path / name).exists()
    scores = pd.read_csv(tmp_path / "open_set_scores.csv")
    assert {
        "pnpff_raw_unknown_score",
        "prototype_score",
        "pnpff_embedding_norm",
        "pnpff_healthy",
    }.issubset(scores.columns)

    cfg.score_fusion.method = "prototype_rank"
    detector.health = {"healthy": False, "reasons": ["inverted_calibration"]}
    with pytest.raises(RuntimeError, match="inverted_calibration"):
        evaluate_fed_digos(
            x_test,
            y_test,
            student_model=model,
            batch_size=8,
            device=torch.device("cpu"),
            cfg=cfg,
            class_names={0: "A", 1: "B"},
            output_dir=tmp_path / "unhealthy",
            evt_models=models,
            pnpff_detector=detector,
            calibration_df=calibration_df,
        )
