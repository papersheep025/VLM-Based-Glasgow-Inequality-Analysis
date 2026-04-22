"""
Losses and multi-task weighting for Route A.

Per-head loss = Huber + λ₁·(1 − Pearson) + λ₂·SoftSpearman.

SoftSpearman is implemented via pairwise-sigmoid soft ranks (differentiable),
following the standard trick::

    rank_i  ≈  Σ_j  sigmoid(β · (p_j − p_i))      (1-based rank)

then Pearson between soft-ranks(pred) and ranks(target).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Correlation losses (operate on 1-D vectors along dim=0)
# ---------------------------------------------------------------------------

def _pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = torch.sqrt((x * x).sum() * (y * y).sum() + eps)
    return num / den.clamp(min=eps)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 − Pearson correlation, operating on 1-D tensors."""
    return 1.0 - _pearson(pred, target)


def _soft_rank(x: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    """Differentiable approximation of rank along dim=0.

    rank_i ≈ 1 + Σ_{j≠i} sigmoid(β (x_j − x_i)).  Lower values → smaller rank.
    Shape: (N,) → (N,).
    """
    diff = x.unsqueeze(0) - x.unsqueeze(1)          # (N, N)  diff[j, i] = x_j − x_i
    soft_lt = torch.sigmoid(beta * diff)
    ranks = 1.0 + soft_lt.sum(dim=0) - torch.diagonal(soft_lt)
    return ranks


def soft_spearman_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    """1 − Pearson(soft_rank(pred), rank(target)).  Differentiable in `pred`."""
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    target_ranks = target.argsort().argsort().float().detach()
    pred_soft_ranks = _soft_rank(pred, beta=beta)
    return 1.0 - _pearson(pred_soft_ranks, target_ranks)


# ---------------------------------------------------------------------------
# Multi-task composite loss
# ---------------------------------------------------------------------------

class HeadLoss(nn.Module):
    """Huber + λ₁·(1 − Pearson) + λ₂·SoftSpearman for a single head."""

    def __init__(self, huber_beta: float = 1.0, lam_pearson: float = 0.5, lam_spearman: float = 0.5, soft_beta: float = 5.0) -> None:
        super().__init__()
        self.huber_beta = huber_beta
        self.lam_p = lam_pearson
        self.lam_s = lam_spearman
        self.soft_beta = soft_beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber = F.smooth_l1_loss(pred, target, beta=self.huber_beta)
        p_term = pearson_loss(pred, target) if self.lam_p > 0 else pred.new_tensor(0.0)
        s_term = soft_spearman_loss(pred, target, beta=self.soft_beta) if self.lam_s > 0 else pred.new_tensor(0.0)
        return huber + self.lam_p * p_term + self.lam_s * s_term


class UncertaintyWeighting(nn.Module):
    """Kendall & Gal (2018) uncertainty weighting for multi-task regression.

    L_total = Σ_i [ 0.5 · exp(-2 s_i) · L_i + s_i ]  where s_i = log σ_i (learnable).
    """

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, per_task_losses: torch.Tensor) -> torch.Tensor:
        # per_task_losses: (T,)
        precision = 0.5 * torch.exp(-2.0 * self.log_sigma)
        return (precision * per_task_losses + self.log_sigma).sum()
