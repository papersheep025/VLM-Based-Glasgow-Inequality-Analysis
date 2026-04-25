"""
Train Route C (caption-embedding + linear regressor) for a single CV fold.

Per fold:
  1. Index pre-computed caption embeddings for train / val samples.
  2. If use_spatial_lag: concat mean-pooled neighbour embeddings (same dim).
  3. If use_poi_vec: fit PoiFitter on train, concat POI vector.
  4. Clone the configured regressor for each of the 7 domains and fit on raw
     SIMD scores [1, 10] directly (no logit normalisation).
  5. Predict on val, report per-domain Spearman.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from decision.data.poi_features import PoiFitter
from decision.data.targets import DOMAINS
from decision.models.route_c.regressors import build_regressor
from src.glasgow_vlm.metrics import spearmanr


@dataclass
class RouteCConfig:
    caption_mode: str = "concat"                        # {concat, templated, modality_sep}
    encoder_backend: str = "sbert"                      # {bert, sbert}
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_batch_size: int = 32
    encoder_max_length: int = 256
    regressor: str = "ridge_cv"                         # {ridge_cv, lasso_cv, elasticnet_cv}
    use_poi_vec: bool = False
    use_spatial_lag: bool = False
    use_ego_gap: bool = False   # append (X_self − X_nbr_mean) as extra features
    use_sar_lag: bool = False   # append mean(y_train_neighbors) per domain as features
    use_dz_agg: bool = False    # mean-pool duplicate samples per datazone before indexing
    use_domain_indicators: bool = False  # append 17 VLM Likert scores + 1 missing flag
    use_latlon: bool = False    # append EPSG:27700 (centroid_x, centroid_y) in meters
    ablate_modalities: list[str] = field(default_factory=list)  # zero out these modality slices
    keep_modalities: list[str] = field(default_factory=list)    # if non-empty, zero out all OTHERS
    seed: int = 42
    domains: list[str] = field(default_factory=lambda: list(DOMAINS))


def _index_rows(samples: list[dict], X: np.ndarray, dz_index: dict[str, int]) -> np.ndarray:
    idx = np.array([dz_index[s["datazone"]] for s in samples], dtype=np.int64)
    return X[idx].astype(np.float32)


def _neighbor_features(
    samples: list[dict],
    X_all: np.ndarray,
    dz_index: dict[str, int],
    neighbors: dict[str, list[str]],
) -> np.ndarray:
    out = np.zeros((len(samples), X_all.shape[1]), dtype=np.float32)
    for i, s in enumerate(samples):
        dz = s["datazone"]
        nbr_dzs = [n for n in neighbors.get(dz, []) if n in dz_index]
        if nbr_dzs:
            nbr_idx = np.array([dz_index[n] for n in nbr_dzs])
            out[i] = X_all[nbr_idx].mean(axis=0)
        else:
            out[i] = X_all[dz_index[dz]]  # fallback: self-embedding
    return out


def _target_lag_features(
    samples: list[dict],
    neighbors: dict[str, list[str]],
    dz_to_y: dict[str, np.ndarray],
    global_mean: np.ndarray,
) -> np.ndarray:
    """Return mean(y_train_neighbors) for each sample — uses only training-set labels."""
    n_domains = global_mean.shape[0]
    out = np.zeros((len(samples), n_domains), dtype=np.float32)
    for i, s in enumerate(samples):
        dz = s["datazone"]
        train_nbrs = [n for n in neighbors.get(dz, []) if n in dz_to_y]
        if train_nbrs:
            out[i] = np.mean([dz_to_y[n] for n in train_nbrs], axis=0)
        else:
            out[i] = global_mean
    return out


def _poi_vectors(samples: list[dict], fitter: PoiFitter) -> np.ndarray:
    return np.stack([fitter.transform(s) for s in samples]).astype(np.float32)


def _indicator_features(samples: list[dict]) -> np.ndarray:
    """Stack 17 Likert scores + 1 missing flag per sample → shape (n, 18).

    A sample with indicators_missing=1 yields a zero score vector + flag=1, so
    the Ridge can learn to down-weight it without leaking arbitrary defaults.
    """
    rows = []
    for s in samples:
        missing = int(s.get("indicators_missing", 1))
        if missing:
            vec = np.zeros(17, dtype=np.float32)
        else:
            vec = np.asarray(s.get("indicators_vec", [0.0] * 17), dtype=np.float32)
            if vec.shape[0] != 17:
                vec = np.zeros(17, dtype=np.float32)
                missing = 1
        rows.append(np.concatenate([vec, np.array([float(missing)], dtype=np.float32)]))
    return np.stack(rows).astype(np.float32)


def _latlon_features(samples: list[dict]) -> np.ndarray:
    """Stack EPSG:27700 (centroid_x, centroid_y) per sample → shape (n, 2)."""
    return np.array(
        [[float(s.get("centroid_x", 0.0)), float(s.get("centroid_y", 0.0))] for s in samples],
        dtype=np.float32,
    )


def _mean_spearman(pred: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
    per = {
        d: spearmanr(target[:, k].tolist(), pred[:, k].tolist())
        for k, d in enumerate(DOMAINS)
    }
    return float(np.mean(list(per.values()))), per


def train_fold_caption(
    train_samples: list[dict],
    val_samples: list[dict],
    X_all: np.ndarray,
    dz_index: dict[str, int],
    cfg: RouteCConfig | None = None,
    fold_dir: str | Path | None = None,
    neighbors: dict[str, list[str]] | None = None,
) -> dict:
    cfg = cfg or RouteCConfig()

    X_tr = _index_rows(train_samples, X_all, dz_index)
    X_va = _index_rows(val_samples, X_all, dz_index)

    if neighbors is not None:
        lag_tr = _neighbor_features(train_samples, X_all, dz_index, neighbors)
        lag_va = _neighbor_features(val_samples, X_all, dz_index, neighbors)
        if cfg.use_ego_gap:
            # Use [self, self−nbr] instead of [self, nbr, self−nbr]:
            # avoids rank deficiency (self − nbr − (self−nbr) = 0).
            X_tr = np.concatenate([X_tr, X_tr - lag_tr], axis=1)
            X_va = np.concatenate([X_va, X_va - lag_va], axis=1)
        else:
            X_tr = np.concatenate([X_tr, lag_tr], axis=1)
            X_va = np.concatenate([X_va, lag_va], axis=1)

    poi_fitter: PoiFitter | None = None
    if cfg.use_poi_vec:
        poi_fitter = PoiFitter().fit(train_samples)
        v_poi_tr = _poi_vectors(train_samples, poi_fitter)
        v_poi_va = _poi_vectors(val_samples, poi_fitter)
        X_tr = np.concatenate([X_tr, v_poi_tr], axis=1)
        X_va = np.concatenate([X_va, v_poi_va], axis=1)

    y_tr_raw = np.array(
        [[s["targets_raw"][d] for d in DOMAINS] for s in train_samples],
        dtype=np.float32,
    )
    y_va_raw = np.array(
        [[s["targets_raw"][d] for d in DOMAINS] for s in val_samples],
        dtype=np.float32,
    )

    if cfg.use_sar_lag and neighbors is not None:
        # Build per-DZ mean y from training set only (no val labels touched).
        from collections import defaultdict as _dd
        dz_y_accum: dict = _dd(list)
        for s, y_row in zip(train_samples, y_tr_raw):
            dz_y_accum[s["datazone"]].append(y_row)
        dz_to_y = {dz: np.mean(ys, axis=0) for dz, ys in dz_y_accum.items()}
        global_mean = y_tr_raw.mean(axis=0)
        sar_tr = _target_lag_features(train_samples, neighbors, dz_to_y, global_mean)
        sar_va = _target_lag_features(val_samples, neighbors, dz_to_y, global_mean)
        X_tr = np.concatenate([X_tr, sar_tr], axis=1)
        X_va = np.concatenate([X_va, sar_va], axis=1)

    if cfg.use_domain_indicators:
        ind_tr = _indicator_features(train_samples)
        ind_va = _indicator_features(val_samples)
        X_tr = np.concatenate([X_tr, ind_tr], axis=1)
        X_va = np.concatenate([X_va, ind_va], axis=1)

    if cfg.use_latlon:
        xy_tr = _latlon_features(train_samples)
        xy_va = _latlon_features(val_samples)
        X_tr = np.concatenate([X_tr, xy_tr], axis=1)
        X_va = np.concatenate([X_va, xy_va], axis=1)

    base = build_regressor(cfg.regressor, seed=cfg.seed)
    fitted_pipelines = []
    val_pred_raw = np.zeros_like(y_va_raw, dtype=np.float32)
    for k in range(len(DOMAINS)):
        model = clone(base)
        model.fit(X_tr, y_tr_raw[:, k])
        val_pred_raw[:, k] = np.asarray(model.predict(X_va)).reshape(-1)
        fitted_pipelines.append(model)

    val_score, per_domain = _mean_spearman(val_pred_raw, y_va_raw)

    if fold_dir is not None:
        fold_dir = Path(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        for d, pipe in zip(DOMAINS, fitted_pipelines):
            joblib.dump(pipe, fold_dir / f"reg_{d}.joblib")
        if poi_fitter is not None:
            poi_fitter.save(fold_dir / "poi_fitter.json")
        (fold_dir / "fold_meta.json").write_text(json.dumps({
            "config": asdict(cfg),
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "val_mean_spearman": val_score,
            "val_per_domain_spearman": per_domain,
        }, indent=2))

    return {
        "val_datazones": [s["datazone"] for s in val_samples],
        "val_pred_raw": val_pred_raw,
        "val_target_raw": y_va_raw,
        "val_mean_spearman": val_score,
        "val_per_domain_spearman": per_domain,
    }
