from __future__ import annotations

import torch
import torch.nn.functional as F


def bidirectional_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return (loss_12 + loss_21) / 2


def alignment_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()