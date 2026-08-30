"""SCAFFOLD local training routine for federated clients."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from src.models.bundle import FedTROSModelBundle as Agent
from src.training.class_balance import class_balanced_cross_entropy, effective_number_class_weights

logger = logging.getLogger("ScaffoldTraining")


def encode_control_variate(control: dict[str, torch.Tensor]) -> str:
    """Serialize SCAFFOLD control variates for Flower scalar config/metrics."""
    return json.dumps({name: tensor.detach().cpu().tolist() for name, tensor in control.items()}, separators=(",", ":"))


def decode_control_variate(payload: str | bytes | None) -> dict[str, torch.Tensor]:
    """Deserialize SCAFFOLD control variates; malformed payloads fail closed to empty."""
    if not payload:
        return {}
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        raw = json.loads(str(payload))
        return {str(name): torch.tensor(value, dtype=torch.float32) for name, value in raw.items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def run_scaffold_training_round(
    agent: Agent,
    features: torch.Tensor,
    labels: torch.Tensor,
    cfg_training: DictConfig,
    device: torch.device,
    server_control_variate: dict[str, torch.Tensor],
    client_control_variate: dict[str, torch.Tensor],
    *,
    round_num: int = 0,
    client_id: str | int = 0,
    logger: logging.Logger | None = None,
) -> tuple[int, dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Execute local SCAFFOLD training for a client over its local dataset.

    Args:
        agent: Local Agent instance (student).
        features: Tensor of local features [N, D].
        labels: Tensor of local labels [N].
        cfg_training: Training configuration.
        device: Active torch device.
        server_control_variate: Global control variate c.
        client_control_variate: Local control variate c_i.
        round_num: Current communication round index.
        client_id: Client identifier.
        logger: Logger instance.

    Returns:
        total_steps: Number of optimizer steps executed.
        metrics: Summary metrics dictionary for the round.
        new_client_control_variate: Updated c_i.
        delta_control_variate: delta_c_i = new_client_control_variate - client_control_variate.
    """
    active_logger = logger or logging.getLogger(f"ScaffoldTraining.{client_id}")
    num_classes = agent.num_classes

    class_weights = effective_number_class_weights(
        labels.detach().cpu(),
        num_classes,
        beta=float(getattr(cfg_training, "class_balance_beta", 0.999)),
        min_weight=float(getattr(cfg_training, "class_weight_min", 0.2)),
        max_weight=float(getattr(cfg_training, "class_weight_max", 5.0)),
        normalize=True,
        device=device,
    )

    batch_size = int(getattr(cfg_training, "batch_size", 64))
    local_epochs = int(getattr(cfg_training, "local_epochs", 2))
    label_smoothing = float(getattr(cfg_training, "label_smoothing", 0.0))
    max_grad_norm = float(getattr(cfg_training, "grad_clip_norm", 1.0))

    dataset = TensorDataset(features.float().to(device), labels.long().to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    agent.student_model.train()
    
    # Store initial weights to compute update for control variate
    initial_weights = {
        name: param.clone().detach() 
        for name, param in agent.student_model.named_parameters() 
        if param.requires_grad
    }

    total_loss = 0.0
    total_steps = 0
    student_start = time.perf_counter()

    for _ in range(max(1, local_epochs)):
        for bx, by in loader:
            agent.optimizer_student.zero_grad()
            _, logits = agent.student_model(bx)
            loss = class_balanced_cross_entropy(
                logits, by, weights=class_weights, label_smoothing=label_smoothing
            )
            loss.backward()
            
            # Apply SCAFFOLD correction: grad = grad - c_i + c
            # Only apply to classifier parameters that are federated
            for name, param in agent.student_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name in server_control_variate and name in client_control_variate:
                        c_global = server_control_variate[name].to(device)
                        c_local = client_control_variate[name].to(device)
                        param.grad.data += (c_global - c_local)
            
            torch.nn.utils.clip_grad_norm_(agent.student_model.classifier_parameters(), max_norm=max_grad_norm)
            agent.optimizer_student.step()

            total_loss += float(loss.detach().item())
            total_steps += 1

    # Update local control variate
    # c_i^+ = c_i - c + (x_i - x_i^+) / (K * eta)
    # where K is total_steps, eta is learning rate
    lr = agent.optimizer_student.param_groups[0]['lr']
    
    new_client_control_variate = {}
    delta_control_variate = {}
    
    with torch.no_grad():
        for name, param in agent.student_model.named_parameters():
            if param.requires_grad:
                if name in server_control_variate and name in client_control_variate:
                    c_global = server_control_variate[name].to(device)
                    c_local = client_control_variate[name].to(device)
                    x_initial = initial_weights[name].to(device)
                    x_final = param.data
                    
                    # Compute pseudo-gradient (x_i - x_i^+) / (K * eta)
                    pseudo_grad = (x_initial - x_final) / (total_steps * lr)
                    
                    c_local_new = c_local - c_global + pseudo_grad
                    
                    new_client_control_variate[name] = c_local_new.cpu()
                    delta_control_variate[name] = (c_local_new - c_local).cpu()

    denom = max(1, total_steps)
    student_seconds = float(time.perf_counter() - student_start)
    metrics = {
        "train_loss": total_loss / denom,
        "train_steps": float(total_steps),
        "is_scaffold_baseline": 1.0,
        "runtime/student_seconds": student_seconds,
    }
    
    active_logger.info(
        "SCAFFOLD baseline local training | steps=%d loss=%.4f",
        total_steps,
        metrics["train_loss"],
    )

    return total_steps, metrics, new_client_control_variate, delta_control_variate
