from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.agents.policy import EpsilonGreedyPolicy, EpsilonScheduler
from src.checkpointing.checkpoints import (
    CheckpointState,
    load_agent_checkpoint,
    save_agent_checkpoint,
    select_checkpoint_metric,
)
from src.data.io import load_tensor_dataset
from src.data.preprocessing import run_preprocessing
from src.evaluation.closed_set import evaluate_closed_set, load_class_names
from src.openset.scorers import (
    EnergyScorer,
    MahalanobisDistanceScorer,
    MSPScorer,
    PrototypeDistanceScorer,
    energy_unknown_score,
    fit_class_prototypes,
    msp_unknown_score,
    prototype_distance_unknown_score,
    select_threshold_from_validation,
)
from src.evaluation.open_set import evaluate_open_set
from src.models.cvae_dqn import OpenSetQChainModelFactory
from src.openset.evt import EVTModel
from src.openset.thresholding import select_validation_threshold as select_scorer_threshold
from src.plotting.plots import render_required_plots
from src.rl.environment import BlockchainIntrusionEnv
from src.rl.local_training import run_local_training_round
from src.rl.replay_buffer import ExperienceReplayBuffer
from src.training.losses import (
    center_compactness_loss,
    supervised_contrastive_loss,
)
from src.utils.config import validate_config
from src.utils.imbalance import compute_class_weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConstantPrior(nn.Module):
    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(states.size(0), 2, device=states.device), torch.zeros(
            states.size(0), 2, device=states.device
        )


class ConstantRecognition(nn.Module):
    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = actions
        return torch.zeros(states.size(0), 2, device=states.device), torch.zeros(
            states.size(0), 2, device=states.device
        )


class FeatureRuleQ(nn.Module):
    num_actions = 2

    def forward(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        _ = z
        class_one = states[:, 0] >= 0.5
        logits = torch.zeros(states.size(0), 2, device=states.device)
        negative = torch.tensor(-4.0, device=states.device)
        positive = torch.tensor(4.0, device=states.device)
        logits[:, 0] = torch.where(class_one, negative, positive)
        logits[:, 1] = torch.where(class_one, positive, negative)
        return logits


class ZeroGenerator(nn.Module):
    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        _ = actions
        return torch.zeros(z.size(0), 3, device=z.device)


def _write_synthetic_raw(path: Path) -> None:
    rows = []
    known_labels = ["Normal", "BP", "DoS", "MitM"]
    for class_id, label in enumerate(known_labels):
        for idx in range(24):
            rows.append(
                {
                    "feature_a": float(class_id * 10 + idx),
                    "feature_b": float(idx % 5) / 5.0,
                    "protocol": "tcp" if idx % 2 else "udp",
                    "service": f"svc_{class_id}_{idx % 3}",
                    "flag": "SF" if idx % 2 else "OTH",
                    "label": label,
                }
            )
    for idx in range(6):
        rows.append(
            {
                "feature_a": 100.0 + idx,
                "feature_b": 3.0 + idx,
                "protocol": "tcp",
                "service": "unknown_svc",
                "flag": "REJ",
                "label": "FoT",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _preprocess_cfg(raw_path: Path, output_dir: Path):
    return OmegaConf.create(
        {
            "seed": 7,
            "dataset": {
                "name": "synthetic_bnat_contract",
                "preprocessing": {
                    "output_dir": str(output_dir),
                    "raw_file": str(raw_path),
                    "label_column": "label",
                    "known_labels": ["Normal", "BP", "DoS", "MitM"],
                    "numerical_cols": None,
                    "categorical_cols": None,
                    "numeric_threshold": 0.9,
                    "validation_split": 0.25,
                    "closed_set_test_size": 0.25,
                    "num_clients": 2,
                    "alpha": 10.0,
                    "iid": False,
                    "unknown_label_id": -1,
                },
            },
        }
    )


def _load_config() -> tuple[object, str]:
    config_dir = str((PROJECT_ROOT / "src" / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config_fl",
            overrides=[
                "experiment=validation",
                "runtime=tiny",
                "tracking.run_id=cheap_validation_config",
            ],
        )
    validate_config(cfg)
    return cfg, config_dir


def _make_threshold_model() -> EVTModel:
    model = EVTModel(0.5)
    model.threshold_u = 1.0
    model.gpd_params = (0.0, 0.0, 1.0)
    return model


def main() -> None:
    run_root = PROJECT_ROOT / "outputs" / (
        "cheap_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    data_dir = run_root / "processed"
    eval_dir = run_root / "eval"
    plot_dir = run_root / "plots"
    run_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {"run_dir": str(run_root)}

    cfg, config_dir = _load_config()
    results["import_test"] = "ok"
    results["config_loading_test"] = {"status": "ok", "config_dir": config_dir}

    raw_path = run_root / "synthetic_raw.csv"
    _write_synthetic_raw(raw_path)
    metadata = run_preprocessing(_preprocess_cfg(raw_path, data_dir), project_root=PROJECT_ROOT)
    results["dataset_loading_test"] = {
        "status": "ok",
        "state_dim": metadata["state_dim"],
        "num_actions": metadata["num_actions"],
    }

    class_names = load_class_names(data_dir / "class_names.json", int(metadata["num_actions"]))
    known_train_x, known_train_y = load_tensor_dataset(data_dir / "known_train.pt")
    val_x, val_y = load_tensor_dataset(data_dir / "validation.pt")
    closed_x, closed_y = load_tensor_dataset(data_dir / "closed_set_test.pt")
    open_x, open_y = load_tensor_dataset(data_dir / "open_set_test.pt")
    results["label_mapping_test"] = {"status": "ok", "class_names": class_names}
    results["known_unknown_split_test"] = {
        "status": "ok",
        "train_unknown_count": int((known_train_y == -1).sum().item()),
        "validation_unknown_count": int((val_y == -1).sum().item()),
        "open_unknown_count": int((open_y == -1).sum().item()),
    }
    assert (known_train_y == -1).sum().item() == 0
    assert (val_y == -1).sum().item() == 0
    assert (open_y == -1).sum().item() > 0

    cfg.model.state_dim = int(metadata["state_dim"])
    cfg.model.num_actions = int(metadata["num_actions"])
    cfg.training.local_episodes_per_round = 1
    cfg.training.steps_per_episode = 3
    cfg.training.min_buffer_size = 2
    cfg.training.batch_size = 2
    cfg.training.replay_buffer_size = 64
    cfg.training.generator.enabled = False
    cfg.checkpointing.dir = str(run_root)
    cfg.checkpointing.latest_checkpoint_path = str(run_root / "latest_checkpoint.pt")
    cfg.checkpointing.best_model_path = str(run_root / "best_model.pt")
    cfg.checkpointing.last_model_path = str(run_root / "last_model.pt")
    cfg.checkpointing.final_model_path = str(run_root / "final_model.pt")

    device = torch.device("cpu")
    agent = Agent(OpenSetQChainModelFactory(cfg.model), cfg.training, device=device)
    with torch.no_grad():
        scorer_features, _ = agent.prior_net(known_train_x[:16].to(device))
        scorer_logits = agent.value_net_main(scorer_features, known_train_x[:16].to(device))
        q_values = scorer_logits[:2]
    assert q_values.shape == (2, int(metadata["num_actions"]))
    results["one_forward_pass"] = {"status": "ok", "q_shape": list(q_values.shape)}

    with torch.no_grad():
        known_features = scorer_features.detach().cpu()
        known_logits = scorer_logits.detach().cpu()
        prototypes = fit_class_prototypes(
            known_features,
            known_train_y[:16],
            num_classes=int(metadata["num_actions"]),
            normalize=True,
        )
        proto_scores = prototype_distance_unknown_score(
            known_features,
            prototypes,
            normalize=True,
        )
        threshold = select_threshold_from_validation(proto_scores, target_known_fpr=0.05)
        assert msp_unknown_score(known_logits).shape[0] == known_logits.shape[0]
        assert energy_unknown_score(known_logits).shape[0] == known_logits.shape[0]
        known_labels = list(range(int(metadata["num_actions"])))
        msp_scores = MSPScorer().score(logits=known_logits)
        energy_scores = EnergyScorer().score(logits=known_logits)
        proto_scorer = PrototypeDistanceScorer().fit(
            known_features,
            known_train_y[:16],
            known_labels=known_labels,
        )
        maha_scorer = MahalanobisDistanceScorer(regularization=1e-3).fit(
            known_features,
            known_train_y[:16],
            known_labels=known_labels,
        )
        proto_interface_scores = proto_scorer.score(known_features)
        maha_scores = maha_scorer.score(known_features)
        scorer_threshold = select_scorer_threshold(
            proto_interface_scores,
            known_train_y[:16],
            known_labels=known_labels,
            target_known_fpr=0.05,
        )
        assert np.isfinite(msp_scores).all()
        assert np.isfinite(energy_scores).all()
        assert np.isfinite(proto_interface_scores).all()
        assert np.isfinite(maha_scores).all()
    results["cvae_dqn_scorer_smoke_test"] = {
        "status": "ok",
        "feature_shape": list(known_features.shape),
        "logit_shape": list(known_logits.shape),
        "prototype_threshold": float(threshold),
    }
    results["open_set_scorer_interface_test"] = {
        "status": "ok",
        "msp_shape": list(msp_scores.shape),
        "energy_shape": list(energy_scores.shape),
        "prototype_threshold": float(scorer_threshold),
        "mahalanobis_max": float(np.max(maha_scores)),
    }

    prior_param_ids = {
        id(param) for group in agent.optimizer_prior.param_groups for param in group["params"]
    }
    q_param_ids = {
        id(param) for group in agent.optimizer_q_rl.param_groups for param in group["params"]
    }
    assert prior_param_ids == {id(param) for param in agent.prior_net.parameters()}
    assert q_param_ids == {
        id(param)
        for module in (agent.recognition_net, agent.value_net_main)
        for param in module.parameters()
    }
    results["optimizer_parameter_coverage_test"] = {
        "status": "ok",
        "prior_params": len(prior_param_ids),
        "q_rl_params": len(q_param_ids),
    }

    tiny_batch_size = 2
    tiny_batch = (
        known_train_x[:tiny_batch_size].to(device),
        known_train_y[:tiny_batch_size].view(-1, 1).to(device),
        torch.ones(tiny_batch_size, 1, device=device),
        known_train_x[1 : tiny_batch_size + 1].to(device),
        torch.zeros(tiny_batch_size, 1, device=device),
        known_train_y[:tiny_batch_size].view(-1, 1).to(device),
    )
    step_metrics = agent.train_step(tiny_batch)
    assert step_metrics["loss/total"] >= 0.0
    assert "loss/classification" in step_metrics
    assert "gradient/q_norm" in step_metrics
    aux_embeddings = torch.randn(6, 3, requires_grad=True)
    aux_labels = torch.tensor([0, 0, 1, 1, 2, 2])
    aux_loss = supervised_contrastive_loss(aux_embeddings, aux_labels)
    aux_loss = aux_loss + center_compactness_loss(aux_embeddings, aux_labels)
    aux_loss.backward()
    assert torch.isfinite(aux_embeddings.grad).all()
    class_weights = compute_class_weights(
        known_train_y,
        num_classes=int(metadata["num_actions"]),
        mode="inverse_frequency",
    )
    assert torch.isfinite(class_weights).all()
    results["one_loss_computation"] = {
        "status": "ok",
        "total_loss": step_metrics["loss/total"],
        "classification_loss": step_metrics["loss/classification"],
        "supcon_loss": step_metrics["loss/supervised_contrastive"],
        "center_loss": step_metrics["loss/center_compactness"],
    }
    results["one_backward_pass"] = {
        "status": "ok",
        "prior_grad_norm": step_metrics["gradient/prior_norm"],
        "q_grad_norm": step_metrics["gradient/q_norm"],
        "auxiliary_grad_norm": float(aux_embeddings.grad.norm().item()),
        "max_class_weight": float(class_weights.max().item()),
    }

    target_before = [param.detach().clone() for param in agent.value_net_target.parameters()]
    with torch.no_grad():
        next(agent.value_net_main.parameters()).add_(0.01)
    agent.update_target_network(tau=0.5)
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(target_before, agent.value_net_target.parameters(), strict=True)
    )
    results["target_network_update_test"] = {"status": "ok"}

    env = BlockchainIntrusionEnv(
        str(data_dir / "known_train.pt"),
        steps_per_episode=int(cfg.training.steps_per_episode),
        device=device,
        global_num_actions=int(metadata["num_actions"]),
        reward_correct=float(cfg.training.reward.correct),
        reward_incorrect=float(cfg.training.reward.incorrect),
        class_balanced_rewards=bool(cfg.training.reward.class_balanced),
        class_balance_power=float(cfg.training.reward.class_balance_power),
        imbalance_cfg=getattr(cfg.training, "imbalance", None),
    )
    buffer = ExperienceReplayBuffer(int(cfg.training.replay_buffer_size))
    policy = EpsilonGreedyPolicy(
        agent.prior_net,
        agent.value_net_main,
        int(metadata["num_actions"]),
        device,
    )
    scheduler = EpsilonScheduler(cfg.training)
    steps, train_metrics = run_local_training_round(
        agent,
        env,
        buffer,
        policy,
        scheduler,
        cfg.training,
        device,
    )
    assert int(train_metrics["train_steps"]) > 0
    assert "avg_total_loss" in train_metrics
    assert "avg_classification_loss" in train_metrics
    assert "gradient_norm_q" in train_metrics
    results["one_tiny_training_step"] = {
        "status": "ok",
        "steps": int(steps),
        "train_steps": int(train_metrics["train_steps"]),
    }
    results["loss_logging_test"] = {
        "status": "ok",
        "avg_total_loss": float(train_metrics["avg_total_loss"]),
        "avg_classification_loss": float(train_metrics["avg_classification_loss"]),
        "gradient_norm_q": float(train_metrics["gradient_norm_q"]),
        "learning_rate_q_rl": float(train_metrics["learning_rate_q_rl"]),
    }

    ckpt_path = run_root / "cheap_checkpoint.pt"
    save_agent_checkpoint(
        agent,
        cfg,
        ckpt_path,
        CheckpointState(
            epoch=1,
            global_step=int(steps),
            metrics={"cheap/train_steps": float(train_metrics["train_steps"])},
        ),
    )
    reloaded_agent = Agent(OpenSetQChainModelFactory(cfg.model), cfg.training, device=device)
    load_agent_checkpoint(reloaded_agent, ckpt_path, device, strict=True, load_optimizers=False)
    checkpoint_metadata = ckpt_path.parent / "checkpoint_metadata.json"
    selected_metric = select_checkpoint_metric(
        {"val/macro_f1": 0.5, "train/accuracy": 1.0},
        monitor_metric="val/macro_f1",
    )
    assert checkpoint_metadata.exists()
    assert selected_metric == ("val/macro_f1", 0.5)
    results["checkpoint_save_load_test"] = {
        "status": "ok",
        "checkpoint": str(ckpt_path),
        "metadata": str(checkpoint_metadata),
        "selected_metric": selected_metric[0],
    }

    closed_metrics = evaluate_closed_set(
        reloaded_agent,
        closed_x,
        closed_y,
        batch_size=4,
        device=device,
        class_names=class_names,
        output_dir=eval_dir,
        prefix="cheap_test",
        save_predictions=True,
    )
    results["evaluation_tiny_sample"] = {
        "status": "ok",
        "accuracy": closed_metrics["cheap_test/accuracy"],
        "macro_f1": closed_metrics["cheap_test/macro_f1"],
    }

    evt_cfg = OmegaConf.create(
        {
            "decision_threshold": 0.5,
            "error_scale_factor": 1.0,
            "unknown_label_id": -1,
            "open_set_label_id": 99,
        }
    )
    threshold_model = _make_threshold_model()
    assert threshold_model.predict_probability_unknown(0.5) == 0.0
    assert threshold_model.predict_probability_unknown(5.0) > 0.5
    results["threshold_sanity_test"] = {"status": "ok"}

    open_metrics = evaluate_open_set(
        features=torch.tensor(
            [[0.1, 0.0, 0.0], [0.8, 0.0, 0.0], [0.1, 5.0, 5.0], [0.8, 5.0, 5.0]],
            dtype=torch.float32,
        ),
        labels=torch.tensor([0, 1, -1, -1], dtype=torch.long),
        batch_size=4,
        prior_net=ConstantPrior(),
        recognition_net=ConstantRecognition(),
        value_net_main=FeatureRuleQ(),
        generation_net=ZeroGenerator(),
        evt_models={0: _make_threshold_model(), 1: _make_threshold_model()},
        evt_meta={
            "global_delta": 0.5,
            "error_scale_factor": 1.0,
            "unknown_label_id": -1,
            "open_set_label_id": 99,
        },
        class_names={0: "known_0", 1: "known_1"},
        output_dir=eval_dir,
        device=device,
        evt_cfg=evt_cfg,
    )
    assert open_metrics["open_set/unknown_detection_rate"] == 1.0
    assert "open_set/auoscr" in open_metrics
    results["metric_sanity_test"] = {
        "status": "ok",
        "unknown_detection_rate": open_metrics["open_set/unknown_detection_rate"],
        "auoscr": open_metrics["open_set/auoscr"],
    }

    # Plot generation with dummy/open-set artifacts. Missing optional inputs render
    # as explicit placeholder panels, which is still a useful smoke test for plotting.
    generated_plots = render_required_plots(
        eval_dir,
        plot_dir,
        ["png"],
        80,
        preprocess_dir=data_dir,
    )
    assert generated_plots
    results["plot_generation_test"] = {
        "status": "ok",
        "num_plots": len(generated_plots),
        "plot_dir": str(plot_dir),
    }

    (run_root / "cheap_validation_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
