"""
Target normalisation for SIMD domain scores.

SIMD_score.csv values: integers 1–10, higher = more deprived.
We map to logit-space for regression, then invert for evaluation.

  normalise(x)  : int 1-10  → float in logit space
  denormalise(y): float logit → float in [1, 10] (continuous; round for ordinal)
"""

from __future__ import annotations

import math
import numpy as np

DOMAINS = ["Income", "Employment", "Health", "Education", "Access", "Crime", "Housing"]
_EPS = 1e-3


def normalise(score: int | float) -> float:
    """Map SIMD score 1–10 to logit([eps, 1-eps])."""
    p = (float(score) - 1.0) / 9.0          # [0, 1]
    p = max(_EPS, min(1.0 - _EPS, p))
    return math.log(p / (1.0 - p))


def denormalise(logit: float) -> float:
    """Invert logit → SIMD score in [1, 10] (continuous)."""
    p = 1.0 / (1.0 + math.exp(-float(logit)))
    return p * 9.0 + 1.0


def normalise_row(row: dict[str, int]) -> dict[str, float]:
    """Normalise all 7 domain values in a SIMD row dict."""
    return {d: normalise(row[d]) for d in DOMAINS if d in row}


def denormalise_row(row: dict[str, float]) -> dict[str, float]:
    return {d: denormalise(v) for d, v in row.items()}


# numpy batched helpers
def normalise_array(arr: np.ndarray) -> np.ndarray:
    """arr shape (N, 7), values 1-10 → logit space."""
    p = (arr.astype(np.float32) - 1.0) / 9.0
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def denormalise_array(arr: np.ndarray) -> np.ndarray:
    """arr shape (N, 7), logit → [1, 10]."""
    p = 1.0 / (1.0 + np.exp(-arr.astype(np.float32)))
    return p * 9.0 + 1.0
