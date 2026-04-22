"""
Route A' — LightGBM decision layer.

SegmentPCA   : per-segment PCA (4 independent PCAs, one per text segment).
MultiDomainLGBM : 7 independent LGBMRegressors, one per SIMD domain, each with
                  its own early stopping on the validation set.

Feature assembly (caller's responsibility):
    X = concat([pca(e_sat), pca(e_ntl), pca(e_sv), pca(e_poi_text), v_poi])

Targets:
    Y is in logit space (targets.normalise_array); invert at evaluation time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.decomposition import PCA

from decision.data.targets import DOMAINS


_SEGMENT_ORDER = ("sat", "ntl", "sv", "poi_text")


class SegmentPCA:
    """Fit one PCA per text segment; concatenate the reduced vectors on transform."""

    def __init__(self, pca_dim: int = 32, seed: int = 42) -> None:
        self.pca_dim = pca_dim
        self.seed = seed
        self._pcas: dict[str, PCA] = {}

    def fit(self, segments: dict[str, np.ndarray]) -> "SegmentPCA":
        for seg in _SEGMENT_ORDER:
            arr = segments[seg]
            n_comp = min(self.pca_dim, arr.shape[0], arr.shape[1])
            pca = PCA(n_components=n_comp, random_state=self.seed)
            pca.fit(arr)
            self._pcas[seg] = pca
        return self

    def transform(self, segments: dict[str, np.ndarray]) -> np.ndarray:
        parts = [self._pcas[seg].transform(segments[seg]) for seg in _SEGMENT_ORDER]
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def out_dim(self) -> int:
        return sum(p.n_components_ for p in self._pcas.values())


class MultiDomainLGBM:
    """7 independent LGBMRegressors (one per SIMD domain) sharing the same features."""

    def __init__(self, lgbm_params: dict, domains: list[str] | None = None) -> None:
        self.domains = list(domains) if domains else list(DOMAINS)
        self.lgbm_params = dict(lgbm_params)
        self._models: dict[str, LGBMRegressor] = {}
        self.best_iters: dict[str, int] = {}

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        early_stopping_rounds: int = 100,
        verbose_eval: int = 0,
    ) -> "MultiDomainLGBM":
        for k, d in enumerate(self.domains):
            model = LGBMRegressor(**self.lgbm_params)
            model.fit(
                X_train, Y_train[:, k],
                eval_set=[(X_val, Y_val[:, k])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(early_stopping_rounds, verbose=False),
                    log_evaluation(verbose_eval),
                ],
            )
            self._models[d] = model
            self.best_iters[d] = int(model.best_iteration_ or model.n_estimators)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        cols = [self._models[d].predict(X, num_iteration=self.best_iters[d]) for d in self.domains]
        return np.stack(cols, axis=1).astype(np.float32)

    # -- persistence ------------------------------------------------------
    def save(self, fold_dir: str | Path) -> None:
        fold_dir = Path(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        for d, m in self._models.items():
            m.booster_.save_model(str(fold_dir / f"lgbm_{d.lower()}.txt"))
        joblib.dump(
            {"best_iters": self.best_iters, "domains": self.domains, "params": self.lgbm_params},
            fold_dir / "lgbm_meta.joblib",
        )
