from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Bidirectional InfoNCE with in-batch negatives.
    Each item i in z1 should match item i in z2.
    """
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)

    return (loss_12 + loss_21) / 2


def focal_info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.05,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal variant of bidirectional InfoNCE.
    Up-weights harder examples and down-weights easier ones.
    """
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    def focal_ce(logits_: torch.Tensor, labels_: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits_, dim=-1)
        probs = log_probs.exp()
        pt = probs[torch.arange(logits_.size(0), device=logits_.device), labels_]
        ce = F.nll_loss(log_probs, labels_, reduction="none")
        focal = ((1.0 - pt) ** gamma) * ce
        return focal.mean()

    loss_12 = focal_ce(logits, labels)
    loss_21 = focal_ce(logits.T, labels)

    return (loss_12 + loss_21) / 2


def alignment_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    """
    Pull parallel pairs closer in the embedding space.
    """
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


def total_embedding_loss(
    z_shw: torch.Tensor,
    z_es: torch.Tensor,
    use_focal: bool = False,
    use_alignment: bool = False,
    contrastive_weight: float = 1.0,
    alignment_weight: float = 0.3,
    temperature: float = 0.05,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Configurable total loss for all embedding experiments.
    """
    if use_focal:
        contrastive = focal_info_nce_loss(
            z_shw,
            z_es,
            temperature=temperature,
            gamma=gamma,
        )
    else:
        contrastive = info_nce_loss(
            z_shw,
            z_es,
            temperature=temperature,
        )

    loss = contrastive_weight * contrastive

    if use_alignment:
        loss = loss + alignment_weight * alignment_loss(z_shw, z_es)

    return loss