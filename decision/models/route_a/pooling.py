"""
Phrase-level attention pooling for variable-length evidence (e.g. streetview phrases).

Used only when the embedding-side pooling branch is chosen (plan §Route A):
we encode each phrase individually -> (B, N, d) and pool to (B, d), length-invariant.

The default ``build_dataset`` pipeline uses input-side dedup+join and does NOT need
this module.  It is provided so training scripts can opt in without refactoring.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Scaled dot-product attention pooling with a learnable query.

    Forward:
      phrases : (B, N, d)      phrase embeddings
      mask    : (B, N) bool    True = valid, False = pad
    Returns:
      pooled  : (B, d)
    """

    def __init__(self, d: int, use_mean_query: bool = True) -> None:
        super().__init__()
        self.d = d
        self.use_mean_query = use_mean_query
        # Learnable projections; when use_mean_query=True the query is
        # mean-of-phrases (per plan spec) but we still learn a projection.
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.empty = nn.Parameter(torch.zeros(d))  # placeholder for all-masked rows

    def forward(self, phrases: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, d = phrases.shape
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=phrases.device)

        # Build query per row: mean of valid phrases (fallback to learnt empty).
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        masked = phrases * mask.unsqueeze(-1)
        mean = masked.sum(dim=1) / valid_counts  # (B, d)
        q_src = mean if self.use_mean_query else phrases.mean(dim=1)
        q = self.q_proj(q_src).unsqueeze(1)  # (B, 1, d)

        k = self.k_proj(phrases)             # (B, N, d)
        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(1) / math.sqrt(d)  # (B, N)
        scores = scores.masked_fill(~mask, float("-inf"))

        all_masked = (~mask).all(dim=1)      # (B,)
        # Avoid NaN softmax on empty rows: replace with zeros then overwrite below.
        safe_scores = torch.where(all_masked.unsqueeze(1), torch.zeros_like(scores), scores)
        weights = F.softmax(safe_scores, dim=-1).unsqueeze(-1)  # (B, N, 1)
        pooled = (weights * phrases).sum(dim=1)                 # (B, d)

        if all_masked.any():
            pooled = torch.where(all_masked.unsqueeze(-1), self.empty.expand_as(pooled), pooled)
        return pooled


def pad_phrase_batch(
    phrase_lists: list[torch.Tensor],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length per-sample phrase tensors into (B, N_max, d) + mask.

    phrase_lists: list of tensors, each shape (n_i, d). Empty tensors allowed.
    """
    if not phrase_lists:
        raise ValueError("empty batch")
    d = phrase_lists[0].shape[-1]
    n_max = max((t.shape[0] for t in phrase_lists), default=0)
    n_max = max(n_max, 1)  # avoid N=0 -> masked softmax still needs a slot
    B = len(phrase_lists)
    out = torch.full((B, n_max, d), pad_value, dtype=phrase_lists[0].dtype)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, t in enumerate(phrase_lists):
        n = t.shape[0]
        if n > 0:
            out[i, :n] = t
            mask[i, :n] = True
    return out, mask
