import torch
import torch.nn.functional as F

from src.utils.utils import EPS, LOGVAR_MAX, LOGVAR_MIN


def kl_warmup_weight(step: int, warmup_steps: int) -> float:
    """Linear KL warmup weight in [0, 1]."""
    warmup_steps = int(warmup_steps)
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(float(step), 0.0) / float(warmup_steps))


def diagonal_gaussian_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    *,
    free_nats: float = 0.0,
    reduce: str = "mean",
    clamp_logvar: bool = True,
) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussians with optional per-dimension free bits."""
    if clamp_logvar:
        logvar_q = logvar_q.clamp(LOGVAR_MIN, LOGVAR_MAX)
        logvar_p = logvar_p.clamp(LOGVAR_MIN, LOGVAR_MAX)

    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl_per_dim = 0.5 * (
        logvar_p
        - logvar_q
        + (var_q + (mu_q - mu_p).pow(2)) / var_p.clamp_min(EPS)
        - 1.0
    )
    if free_nats > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=float(free_nats))
    kl = kl_per_dim.sum(dim=1)
    if reduce == "none":
        return kl
    if reduce == "sum":
        return kl.sum()
    return kl.mean()


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy for imbalanced known-class classification."""
    targets = targets.view(-1).long()
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt).pow(float(gamma)) * ce).mean()


def smooth_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 reconstruction loss for scaled tabular features."""
    return F.smooth_l1_loss(recon, target, beta=float(beta))


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Supervised contrastive loss over a mini-batch of labeled embeddings."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D [B, D], got {tuple(embeddings.shape)}")
    labels = labels.view(-1).long()
    if labels.numel() != embeddings.size(0):
        raise ValueError("labels and embeddings batch size must match.")
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0

    temp = max(float(temperature), eps)
    features = F.normalize(embeddings, dim=1)
    logits = torch.matmul(features, features.T) / temp
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = embeddings.size(0)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not bool(valid_anchor.any()):
        return embeddings.sum() * 0.0

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(eps))
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts
    return -mean_log_prob_pos[valid_anchor].mean()


def center_compactness_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Batch prototype compactness loss: mean squared distance to batch class centers."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D [B, D], got {tuple(embeddings.shape)}")
    labels = labels.view(-1).long()
    if labels.numel() != embeddings.size(0):
        raise ValueError("labels and embeddings batch size must match.")
    if embeddings.size(0) == 0:
        return embeddings.sum() * 0.0

    loss = embeddings.sum() * 0.0
    class_count = 0
    for label in labels.unique(sorted=True):
        mask = labels.eq(label)
        if not bool(mask.any()):
            continue
        class_embeddings = embeddings[mask]
        center = class_embeddings.mean(dim=0, keepdim=True)
        loss = loss + (class_embeddings - center).pow(2).sum(dim=1).mean()
        class_count += 1
    if class_count == 0:
        return embeddings.sum() * 0.0
    return loss / float(class_count)


def energy_score(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    """Energy score where larger values indicate lower in-distribution confidence."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [B, C], got {tuple(logits.shape)}")
    temp = max(float(temperature), 1e-6)
    return -temp * torch.logsumexp(logits / temp, dim=1)
