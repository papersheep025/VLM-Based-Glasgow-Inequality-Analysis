"""
Train Route A' (LightGBM) for a single (train_samples, val_samples) fold.

Per fold:
  1. Fit PoiFitter on train (log1p + z-score, no leakage).
  2. Index train/val segment embeddings from the shared cache.
  3. Fit SegmentPCA (per-segment PCA) on train, transform train+val.
  4. Concat [pca_segments, poi_vec] as LGBM input.
  5. Fit 7 LGBMRegressors on logit-space targets with val early stopping.
  6. Return val predictions in raw 1-10 space for OOF aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch

from decision.data.poi_features import PoiFitter
from decision.data.targets import DOMAINS, denormalise_array, normalise_array
from decision.models.route_a_lgbm.model import MultiDomainLGBM, SegmentPCA
from src.glasgow_vlm.metrics import spearmanr


_SEGMENTS = ("sat", "ntl", "sv", "poi_text")


@dataclass
class RouteALGBMConfig:
    pca_dim: int = 32
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l2: float = 1.0
    objective: str = "regression"
    early_stopping_rounds: int = 100
    seed: int = 42
    domains: list[str] = field(default_factory=lambda: list(DOMAINS))


def _lgbm_params(cfg: RouteALGBMConfig) -> dict:
    return {
        "objective": cfg.objective,
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.lambda_l2,
        "random_state": cfg.seed,
        "verbose": -1,
    }


def _index_segments(
    samples: list[dict],
    segments: dict[str, torch.Tensor],
    dz_index: dict[str, int],
) -> dict[str, np.ndarray]:
    idx = np.array([dz_index[s["datazone"]] for s in samples], dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for seg in _SEGMENTS:
        t = segments[seg]
        arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        out[seg] = arr[idx].astype(np.float32)
    return out


def _poi_vectors(samples: list[dict], fitter: PoiFitter) -> np.ndarray:
    return np.stack([fitter.transform(s) for s in samples]).astype(np.float32)


def _mean_spearman(pred: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
    per = {d: spearmanr(target[:, k].tolist(), pred[:, k].tolist())
           for k, d in enumerate(DOMAINS)}
    return float(np.mean(list(per.values()))), per


def train_fold_lgbm(
    train_samples: list[dict],
    val_samples: list[dict],
    segments: dict[str, torch.Tensor],
    dz_index: dict[str, int],
    cfg: RouteALGBMConfig | None = None,
    fold_dir: str | Path | None = None,
) -> dict:
    cfg = cfg or RouteALGBMConfig()

    poi_fitter = PoiFitter().fit(train_samples)
    v_poi_tr = _poi_vectors(train_samples, poi_fitter)
    v_poi_va = _poi_vectors(val_samples, poi_fitter)

    seg_tr = _index_segments(train_samples, segments, dz_index)
    seg_va = _index_segments(val_samples, segments, dz_index)

    pca = SegmentPCA(pca_dim=cfg.pca_dim, seed=cfg.seed).fit(seg_tr)
    X_tr = np.concatenate([pca.transform(seg_tr), v_poi_tr], axis=1)
    X_va = np.concatenate([pca.transform(seg_va), v_poi_va], axis=1)

    y_tr_raw = np.array([[s["targets_raw"][d] for d in DOMAINS] for s in train_samples], dtype=np.float32)
    y_va_raw = np.array([[s["targets_raw"][d] for d in DOMAINS] for s in val_samples], dtype=np.float32)
    Y_tr = normalise_array(y_tr_raw)
    Y_va = normalise_array(y_va_raw)

    model = MultiDomainLGBM(_lgbm_params(cfg), domains=cfg.domains)
    model.fit(X_tr, Y_tr, X_va, Y_va, early_stopping_rounds=cfg.early_stopping_rounds)

    val_pred_logit = model.predict(X_va)
    val_pred_raw = denormalise_array(val_pred_logit)
    val_score, per_domain = _mean_spearman(val_pred_raw, y_va_raw)

    if fold_dir is not None:
        fold_dir = Path(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save(fold_dir)
        joblib.dump(pca, fold_dir / "pca.joblib")
        poi_fitter.save(fold_dir / "poi_fitter.json")

    return {
        "val_datazones": [s["datazone"] for s in val_samples],
        "val_pred_logit": val_pred_logit,
        "val_pred_raw": val_pred_raw,
        "val_target_raw": y_va_raw,
        "val_mean_spearman": val_score,
        "val_per_domain_spearman": per_domain,
        "best_iters": model.best_iters,
    }
