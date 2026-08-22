import json
import logging
import random
from contextlib import suppress
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import Parameters, parameters_to_ndarrays
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.bundle import FedTROSModelBundle as Agent
from src.models.models import ModelFactory
from src.training.local_training import run_local_training_round
from src.utils.utils import get_device, project_root

PROJECT_ROOT = project_root()


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


class FlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient implementing FedTROS-PR and Student-based Federated Baselines."""

    def __init__(
        self,
        cid: str,
        cfg: DictConfig,
        data_path: str,
        device: torch.device | None = None,
        *,
        simulation_gpu_batching: bool = False,
        simulation_execution_device: torch.device | str | None = None,
    ):
        self.cid = cid
        self.logger = logging.getLogger(f"Client.{cid}")
        self.cfg = cfg
        self.data_path = data_path
        self.device = device if device is not None else get_device(logger=self.logger)
        self._simulation_gpu_batching = bool(simulation_gpu_batching)
        self._simulation_execution_device = (
            torch.device(simulation_execution_device)
            if simulation_execution_device is not None
            else self.device
        )
        self._simulation_rest_device = (
            torch.device("cpu") if self._simulation_gpu_batching else self.device
        )
        self._move_data_to_device = bool(cfg.device.move_data_to_device)

        self.logger.info("Client %s: Initializing...", cid)

        # Load client-local training dataset
        resolved_data_path = _resolve_project_path(data_path)
        data = torch.load(resolved_data_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            raw_features = data["features"] if "features" in data else data["X"]
            raw_labels = data["labels"] if "labels" in data else data["y"]
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            raw_features, raw_labels = data[0], data[1]
        else:
            raise ValueError(f"Unsupported data format in {data_path}")

        data_dev = self.device if self._move_data_to_device else torch.device("cpu")
        self.features = raw_features.to(data_dev).float()
        self.labels = raw_labels.to(data_dev).long()

        self.model_factory = ModelFactory(cfg.model)
        self.agent = Agent(self.model_factory, cfg.training, self.device, logger=self.logger)
        self._private_state_restored = False
        self._maybe_restore_private_state()

        self.cached_weights: list[np.ndarray] = []
        self.cached_metrics: dict[str, Any] = {}
        self.local_data_profile = self._build_local_data_profile()

        # Closed-set evaluation is numeric-only. Publication figures live in the separate
        # plotting repository and are generated from exported scientific artifacts.
        # Client evaluation therefore has no figure/output image directory.

        # Closed-set evaluation initialization
        self.eval_enabled: bool = False
        self.eval_loader: DataLoader | None = None
        self.eval_class_names: list[str] = []
        self.eval_output_dir: Path | None = None
        self.closed_set_data_path: Path | None = None
        self._init_closed_set_evaluation()

        self.logger.info(
            "Client %s: Initialization complete | samples=%d | classes=%d",
            cid,
            int(self.local_data_profile["local_num_examples"]),
            int(cfg.model.num_classes),
        )

    def _private_checkpoint_path(self) -> Path:
        run_dir = Path(str(OmegaConf.select(self.cfg, "tracking.run_dir", default="outputs/current")))
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        path = run_dir / "checkpoints" / "private" / f"client_{self.cid}_latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _resume_requested(self) -> bool:
        return bool(
            OmegaConf.select(self.cfg, "federated.resume_from", default=None)
            or int(OmegaConf.select(self.cfg, "federated.resume_round_offset", default=0) or 0) > 0
        )

    def _maybe_restore_private_state(self) -> None:
        if not self._resume_requested():
            return
        path = self._private_checkpoint_path()
        if not path.exists():
            raise RuntimeError(
                f"Exact VCT resume requested but private client checkpoint is missing: {path}. "
                "Do not claim exact continuation from a student-only checkpoint."
            )
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema_version") != 2 or str(payload.get("teacher_type", "")) != "variational_classifier":
            raise RuntimeError(f"Incompatible private teacher checkpoint: {path}")
        expected_hash = str(OmegaConf.select(self.cfg, "experiment.config_hash", default="") or "")
        checkpoint_hash = str(payload.get("config_hash", "") or "")
        if expected_hash and checkpoint_hash and expected_hash != checkpoint_hash:
            raise RuntimeError(
                f"Private checkpoint/config mismatch for client {self.cid}: "
                f"checkpoint={checkpoint_hash[:12]} current={expected_hash[:12]}"
            )
        self.agent.load_private_state_dict(payload["private_state"], strict=True)
        rng_state = payload.get("rng_state") or {}
        if rng_state:
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("torch_cpu") is not None:
                torch.set_rng_state(rng_state["torch_cpu"])
            if torch.cuda.is_available() and rng_state.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        self._private_state_restored = True
        self.logger.info("Restored private VCT state for client %s from %s", self.cid, path)

    def _save_private_state(self, round_num: int) -> None:
        path = self._private_checkpoint_path()
        payload = {
            "schema_version": 2,
            "method": "FedTROS-PR",
            "method_id": "fedtros_pr",
            "teacher_type": "variational_classifier",
            "client_id": str(self.cid),
            "round": int(round_num),
            "config_hash": str(OmegaConf.select(self.cfg, "experiment.config_hash", default="") or ""),
            "code_commit": str(OmegaConf.select(self.cfg, "experiment.git_commit", default="unknown_commit")),
            "private_state": self.agent.private_state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def _switch_runtime_device(self, device: torch.device | str) -> None:
        target_device = torch.device(device)
        if target_device == self.device:
            return
        self.agent.to(target_device)
        self.device = target_device

    def _enter_execution_device(self) -> tuple[torch.device, bool]:
        target_device = (
            self._simulation_execution_device if self._simulation_gpu_batching else self.device
        )
        switched = target_device != self.device
        if switched:
            self._switch_runtime_device(target_device)
        return target_device, switched

    def _exit_execution_device(self, target_device: torch.device, switched: bool) -> None:
        if not switched:
            return
        self._switch_runtime_device(self._simulation_rest_device)
        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        _ = config
        return self.agent.get_student_parameters()

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.agent.set_student_parameters(parameters)

    @staticmethod
    def _param_list_norm(parameters: list[np.ndarray]) -> float:
        total = 0.0
        for param in parameters:
            arr = np.asarray(param, dtype=np.float64)
            total += float(np.sum(arr * arr))
        return float(np.sqrt(max(total, 0.0)))

    @staticmethod
    def _param_list_distance(a: list[np.ndarray], b: list[np.ndarray]) -> float:
        total = 0.0
        for pa, pb in zip(a, b, strict=True):
            diff = np.asarray(pa, dtype=np.float64) - np.asarray(pb, dtype=np.float64)
            total += float(np.sum(diff * diff))
        return float(np.sqrt(max(total, 0.0)))

    def fit(
        self, parameters: list[np.ndarray] | Parameters, config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        phase = config.get("phase", "standard")
        round_num = config.get("server_round", "?")
        round_index = int(round_num) if str(round_num).isdigit() else 0
        strategy_name = str(self.cfg.strategy.name).lower()
        proximal_mu = float(self.cfg.server.proximal_mu) if strategy_name == "fedprox" else 0.0

        execution_device, switched = self._enter_execution_device()
        try:
            param_list = (
                parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
            )

            # =========================================================
            # FedTROS-PR: Private VCT teacher + Guided Federated Student
            # =========================================================
            if phase in {"fedtros_pr", "fedtros"}:
                self.logger.info(f"Client {self.cid} [FedTROS-PR]: Round {round_num}")
                student_before = self.agent.get_student_parameters()
                if param_list:
                    self.agent.set_student_parameters(param_list)
                student_after_load = self.agent.get_student_parameters()
                loaded_delta = self._param_list_distance(student_before, student_after_load)

                num_steps_trained, metrics = run_local_training_round(
                    agent=self.agent,
                    features=self.features,
                    labels=self.labels,
                    cfg_training=self.cfg.training,
                    device=self.device,
                    proximal_mu=0.0,
                    round_num=round_index,
                    client_id=self.cid,
                    is_fedtros=True,
                    logger=self.logger,
                )

                # Train Student OSR Branch if enabled
                prototype_rank_cfg = getattr(getattr(self.cfg, "open_set", None), "prototype_rank", None)
                if prototype_rank_cfg is not None and bool(getattr(prototype_rank_cfg, "enabled", False)):
                    try:
                        osr_metrics = self.agent.train_student_osr_on_dataset(
                            self.features, self.labels, prototype_rank_cfg, logger=self.logger
                        )
                        metrics.update(osr_metrics)
                    except Exception:
                        self.logger.exception("Client %s: Student OSR branch training failed", self.cid)

                student_after_train = self.agent.get_student_parameters()
                train_delta = self._param_list_distance(student_after_load, student_after_train)
                metrics.update(
                    {
                        "student_load_delta_norm": float(loaded_delta),
                        "student_train_delta_norm": float(train_delta),
                        "student_num_parameters": float(sum(p.size for p in student_after_train)),
                        "local_num_examples": float(self.local_data_profile["local_num_examples"]),
                        "raw_data_local_metrics": 1.0,
                    }
                )
                if bool(getattr(self.cfg.training, "evaluate_after_local_fit", False)):
                    metrics.update(self._evaluate_local_student())
                self._save_private_state(round_index)
                num_examples = int(self.local_data_profile["local_num_examples"])
                return student_after_train, num_examples, self._sanitize_server_metrics(metrics)

            # =========================================================
            # Standard Baselines: FedAvg-Student / FedProx-Student
            # =========================================================
            phase_name = "FedProx-Student" if proximal_mu > 0.0 else "FedAvg-Student"
            self.logger.info(f"Client {self.cid} [{phase_name}]: Round {round_num}")
            if param_list:
                self.agent.set_student_parameters(param_list)

            if bool(getattr(self.cfg.training, "reset_optimizer_each_round", False)):
                self.agent.reset_federated_optimizers()

            num_steps_trained, metrics = run_local_training_round(
                agent=self.agent,
                features=self.features,
                labels=self.labels,
                cfg_training=self.cfg.training,
                device=self.device,
                proximal_mu=proximal_mu,
                round_num=round_index,
                client_id=self.cid,
                is_fedtros=False,
                logger=self.logger,
            )
            if bool(getattr(self.cfg.training, "evaluate_after_local_fit", False)):
                metrics.update(self._evaluate_local_student())
            updated_params = self.agent.get_student_parameters()
            num_examples = int(self.local_data_profile["local_num_examples"])
            return updated_params, num_examples, metrics
        finally:
            self._exit_execution_device(execution_device, switched)

    _SERVER_LOCAL_ONLY_METRIC_KEYS = {
        "label_histogram",
        "class_entropy",
        "label_coverage",
        "missing_classes",
        "present_classes",
        "imbalance_ratio",
    }

    def _sanitize_server_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Strip client label distributions to keep client-local distribution details off the server."""
        sanitized = {
            k: v for k, v in (metrics or {}).items() if k not in self._SERVER_LOCAL_ONLY_METRIC_KEYS
        }
        sanitized["raw_data_local_metrics"] = 1.0
        return sanitized

    def _build_local_data_profile(self) -> dict[str, Any]:
        labels = self.labels.detach().cpu().long()
        num_classes = int(self.cfg.model.num_classes)
        counts = torch.bincount(labels.clamp(min=0), minlength=num_classes)[:num_classes]
        total = int(labels.numel())
        probs = counts.float() / max(total, 1)
        nonzero = probs[probs > 0]
        entropy = (
            float((-(nonzero * torch.log(nonzero))).sum().item() / np.log(num_classes))
            if num_classes > 1 and nonzero.numel() > 0
            else 0.0
        )
        coverage = float((counts > 0).sum().item() / max(num_classes, 1))
        missing = [int(i) for i in range(num_classes) if int(counts[i].item()) == 0]
        present = [int(i) for i in range(num_classes) if int(counts[i].item()) > 0]
        return {
            "local_num_examples": float(total),
            "class_entropy": entropy,
            "label_coverage": coverage,
            "missing_classes": json.dumps(missing),
            "present_classes": json.dumps(present),
            "label_histogram": json.dumps(
                {str(i): int(counts[i].item()) for i in range(num_classes)}, sort_keys=True
            ),
        }

    def evaluate(
        self, parameters: list[np.ndarray] | Parameters, config: dict[str, Any]
    ) -> tuple[float, int, dict[str, float]]:
        round_num = config.get("server_round", "?")
        round_index = int(round_num) if str(round_num).isdigit() else 0

        execution_device, switched = self._enter_execution_device()
        try:
            param_list = (
                parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
            )
            if param_list:
                self.agent.set_student_parameters(param_list)

            if not self.eval_enabled or self.eval_loader is None:
                return 0.0, 0, {}

            loss, num_examples, student_metrics = self._evaluate_closed_set(
                round_index, prefix="GLOBAL_STUDENT"
            )
            student_metrics["cid"] = self.cid
            return loss, num_examples, student_metrics
        finally:
            self._exit_execution_device(execution_device, switched)

    def _init_closed_set_evaluation(self) -> None:
        paths_cfg = self.cfg.paths
        test_path_candidates = [
            getattr(paths_cfg, "shared_closed_set_test_data", None),
            getattr(paths_cfg, "closed_set_test_data", None),
        ]
        test_data_path: Path | None = None
        for cand in test_path_candidates:
            if cand:
                p = _resolve_project_path(cand)
                if p.exists():
                    test_data_path = p
                    break

        class_names_rel = getattr(paths_cfg, "class_names", None)
        if not (test_data_path and class_names_rel):
            return

        class_names_path = _resolve_project_path(class_names_rel)
        try:
            data = torch.load(test_data_path, map_location="cpu", weights_only=True)
            features = data["features"].float()
            labels = data["labels"].long()
        except Exception as exc:
            self.logger.error("Client %s: failed to load test data: %s", self.cid, exc)
            return

        dataset = TensorDataset(features, labels)
        self.eval_loader = DataLoader(
            dataset, batch_size=int(self.cfg.training.batch_size), shuffle=False
        )
        self.eval_class_names = self._load_class_names(class_names_path, int(self.cfg.model.num_classes))
        self.eval_enabled = True

    def _load_class_names(self, path: Path, num_classes: int) -> list[str]:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as fp:
                    raw = json.load(fp)
                sorted_items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
                return [name for _, name in sorted_items]
            except Exception:
                pass
        return [f"class_{idx}" for idx in range(num_classes)]

    def _evaluate_closed_set(
        self, round_index: int, prefix: str = "GLOBAL"
    ) -> tuple[float, int, dict[str, float]]:
        assert self.eval_loader is not None
        self.agent.student_model.eval()
        total_loss = 0.0
        total_samples = 0
        all_true: list[int] = []
        all_pred: list[int] = []

        with torch.no_grad():
            for features, labels in self.eval_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                _, logits = self.agent.student_model(features)
                loss = F.cross_entropy(logits, labels, reduction="mean")
                preds = logits.argmax(dim=1)
                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        num_examples = int(y_true.size)
        accuracy = float((y_true == y_pred).mean()) if num_examples else 0.0
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if num_examples else 0.0
        balanced = float(balanced_accuracy_score(y_true, y_pred)) if num_examples else 0.0

        return (
            avg_loss,
            num_examples,
            {
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "f1_macro": f1,
                "num_examples": num_examples,
                "raw_data_local_metrics": 1.0,
            },
        )

    def _evaluate_local_student(self) -> dict[str, float]:
        """Measure the locally trained student on its client-local data."""
        dataset = TensorDataset(self.features.detach().cpu(), self.labels.detach().cpu())
        loader = DataLoader(
            dataset,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=False,
            drop_last=False,
        )
        was_training = self.agent.student_model.training
        self.agent.student_model.eval()
        all_true: list[int] = []
        all_pred: list[int] = []
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                _, logits = self.agent.student_model(features)
                all_true.extend(labels.cpu().numpy().tolist())
                all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        self.agent.student_model.train(was_training)

        y_true = np.asarray(all_true)
        y_pred = np.asarray(all_pred)
        if y_true.size == 0:
            return {"local_student_accuracy": 0.0, "local_student_f1_macro": 0.0}
        return {
            "local_student_accuracy": float((y_true == y_pred).mean()),
            "local_student_f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }
