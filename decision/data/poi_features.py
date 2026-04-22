"""
Build a fixed-length POI count feature vector from parsed POI dicts.

Usage (fit on train set, transform any split):
    fitter = PoiFitter()
    fitter.fit(train_records)          # learns vocab + z-score stats
    v = fitter.transform(record)       # → np.ndarray shape (vocab_size,)
    fitter.save(path) / PoiFitter.load(path)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

_TOP_K = 15  # keep top-K poi types, rest → "other"


class PoiFitter:
    def __init__(self, top_k: int = _TOP_K) -> None:
        self.top_k = top_k
        self.vocab: list[str] = []      # ordered list of types (incl. "other")
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, records: list[dict]) -> "PoiFitter":
        """records: list of parsed perception records with 'poi_counts' key."""
        freq: Counter[str] = Counter()
        for r in records:
            freq.update(r.get("poi_counts", {}))
        top_types = [t for t, _ in freq.most_common(self.top_k)]
        self.vocab = top_types + ["other"]

        vecs = np.array([self._raw(r) for r in records], dtype=np.float32)  # (N, V)
        self._mean = vecs.mean(axis=0)
        self._std = np.where(vecs.std(axis=0) > 0, vecs.std(axis=0), 1.0)
        self._fitted = True
        return self

    def transform(self, record: dict) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        raw = self._raw(record)
        return ((raw - self._mean) / self._std).astype(np.float32)

    def transform_batch(self, records: list[dict]) -> np.ndarray:
        return np.stack([self.transform(r) for r in records])

    @property
    def dim(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    def _raw(self, record: dict) -> np.ndarray:
        counts = record.get("poi_counts", {})
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        other_idx = len(self.vocab) - 1
        for poi_type, cnt in counts.items():
            if poi_type in self.vocab:
                vec[self.vocab.index(poi_type)] += cnt
            else:
                vec[other_idx] += cnt
        return np.log1p(vec)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "top_k": self.top_k,
            "vocab": self.vocab,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
        }
        path.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PoiFitter":
        state = json.loads(Path(path).read_text())
        obj = cls(top_k=state["top_k"])
        obj.vocab = state["vocab"]
        if state["mean"] is not None:
            obj._mean = np.array(state["mean"], dtype=np.float32)
            obj._std = np.array(state["std"], dtype=np.float32)
            obj._fitted = True
        return obj
