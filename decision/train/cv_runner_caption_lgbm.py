"""5-fold IZ-grouped CV runner for Route C features with LightGBM regressors.

Reuses Route C's full feature-assembly pipeline (SBERT modality_sep + spatial
lag + ego-gap + POI + SAR lag + 17-dim indicators) by importing the private
helpers from ``decision.train.route_c_train``. Only the regressor swaps from
RidgeCV → ``MultiDomainLGBM`` (existing in models/route_a_lgbm/model.py),
to test whether non-linear interactions break the linear ceiling we observed
(~0.53 pooled ρ).

CLI:
    python -m decision.train.cv_runner_caption_lgbm \
        --config decision/configs/route_c_modality_sep_v1_lgbm.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from decision.data.poi_features import PoiFitter
from decision.data.spatial_neighbors import build_neighbors
from decision.data.targets import DOMAINS, denormalise_array, normalise_array
from decision.train.cv_runner import load_dataset
from decision.train.cv_runner_caption import (
    _aggregate_X_by_dz,
    _load_yaml,
    build_or_load_captions,
)
from decision.train.route_c_train import (
    RouteCConfig,
    _index_rows,
    _indicator_features,
    _latlon_features,
    _neighbor_features,
    _poi_vectors,
    _target_lag_features,
)
from src.glasgow_vlm.metrics import spearmanr
from src.glasgow_vlm.splits import DEFAULT_SHP, group_kfold_by_iz


@dataclass
class LGBMHyperparams:
    backend: str = "lgbm"           # {"lgbm", "hgb"} — hgb = sklearn HistGradientBoostingRegressor (no libomp dep)
    sbert_pca_dim: int = 0          # 0 = no PCA; >0 = reduce SBERT block (and lag) to this dim before regressor
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


def _lgbm_params(h: LGBMHyperparams, seed: int) -> dict:
    return {
        "objective": h.objective,
        "n_estimators": h.n_estimators,
        "learning_rate": h.learning_rate,
        "num_leaves": h.num_leaves,
        "min_data_in_leaf": h.min_data_in_leaf,
        "feature_fraction": h.feature_fraction,
        "bagging_fraction": h.bagging_fraction,
        "bagging_freq": h.bagging_freq,
        "lambda_l2": h.lambda_l2,
        "random_state": seed,
        "verbose": -1,
    }


class _MultiDomainHGB:
    """sklearn HistGradientBoostingRegressor wrapper mirroring MultiDomainLGBM
    (per-domain training + early stopping on the val set)."""

    def __init__(self, h: LGBMHyperparams, seed: int):
        self.h = h
        self.seed = seed
        self.domains = list(DOMAINS)
        self._models: dict = {}
        self.best_iters: dict = {}

    def fit(self, X_tr, Y_tr, X_va, Y_va, early_stopping_rounds: int = 100):
        from sklearn.ensemble import HistGradientBoostingRegressor
        # HGB does its own early stopping when validation_fraction is set; we
        # instead concatenate (X_tr, X_va) and pass an explicit val mask via
        # the validation_fraction trick — simpler: just rely on n_iter_no_change.
        # To honour the exact (X_va, Y_va) split, use HGB's `early_stopping=True`
        # with validation_fraction sliced from train, then fit a second pass on
        # full train with the chosen iters. Keep it simple: just early-stop on
        # an internal 15% slice of train and fit normally.
        for k, d in enumerate(self.domains):
            model = HistGradientBoostingRegressor(
                max_iter=self.h.n_estimators,
                learning_rate=self.h.learning_rate,
                max_leaf_nodes=self.h.num_leaves,
                min_samples_leaf=self.h.min_data_in_leaf,
                l2_regularization=self.h.lambda_l2,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=early_stopping_rounds,
                random_state=self.seed,
            )
            model.fit(X_tr, Y_tr[:, k])
            self._models[d] = model
            self.best_iters[d] = int(model.n_iter_)
        return self

    def predict(self, X):
        cols = [self._models[d].predict(X) for d in self.domains]
        return np.stack(cols, axis=1).astype(np.float32)

    def save(self, fold_dir):
        import joblib
        fold_dir = Path(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"models": self._models, "best_iters": self.best_iters,
             "domains": self.domains, "backend": "hgb"},
            fold_dir / "hgb_models.joblib",
        )


def _build_model(h: LGBMHyperparams, seed: int):
    if h.backend == "hgb":
        return _MultiDomainHGB(h, seed)
    if h.backend == "lgbm":
        from decision.models.route_a_lgbm.model import MultiDomainLGBM
        return MultiDomainLGBM(_lgbm_params(h, seed), domains=list(DOMAINS))
    raise ValueError(f"unknown backend {h.backend!r}")


def _apply_sbert_pca(X_tr, X_va, cfg, pca_dim: int, seed: int):
    """Reduce the SBERT block (and the lag/ego-gap block, if present) to
    ``pca_dim`` via per-fold PCA fit on train. Tail features (POI/SAR/
    indicators/latlon) are passed through untouched.

    Layout (mirrors train_fold_caption / _build_features):
        [SBERT_self : 1536]  [lag_or_ego : 1536 if use_spatial_lag]  [tail ...]
    """
    if pca_dim <= 0:
        return X_tr, X_va
    from sklearn.decomposition import PCA
    D_SBERT = 1536  # modality_sep encoder output
    if X_tr.shape[1] < D_SBERT:
        return X_tr, X_va  # nothing to reduce
    blocks_tr, blocks_va = [], []
    pca = PCA(n_components=pca_dim, random_state=seed).fit(X_tr[:, :D_SBERT])
    blocks_tr.append(pca.transform(X_tr[:, :D_SBERT]).astype(np.float32))
    blocks_va.append(pca.transform(X_va[:, :D_SBERT]).astype(np.float32))
    cur = D_SBERT
    if cfg.use_spatial_lag and X_tr.shape[1] >= cur + D_SBERT:
        pca2 = PCA(n_components=pca_dim, random_state=seed).fit(X_tr[:, cur:cur + D_SBERT])
        blocks_tr.append(pca2.transform(X_tr[:, cur:cur + D_SBERT]).astype(np.float32))
        blocks_va.append(pca2.transform(X_va[:, cur:cur + D_SBERT]).astype(np.float32))
        cur += D_SBERT
    if cur < X_tr.shape[1]:
        blocks_tr.append(X_tr[:, cur:])
        blocks_va.append(X_va[:, cur:])
    return np.concatenate(blocks_tr, axis=1), np.concatenate(blocks_va, axis=1)


def _build_features(train_samples, val_samples, X_all, dz_index, cfg, neighbors):
    """Mirror of train_fold_caption's feature assembly, minus the regressor."""
    X_tr = _index_rows(train_samples, X_all, dz_index)
    X_va = _index_rows(val_samples, X_all, dz_index)

    if neighbors is not None:
        lag_tr = _neighbor_features(train_samples, X_all, dz_index, neighbors)
        lag_va = _neighbor_features(val_samples, X_all, dz_index, neighbors)
        if cfg.use_ego_gap:
            X_tr = np.concatenate([X_tr, X_tr - lag_tr], axis=1)
            X_va = np.concatenate([X_va, X_va - lag_va], axis=1)
        else:
            X_tr = np.concatenate([X_tr, lag_tr], axis=1)
            X_va = np.concatenate([X_va, lag_va], axis=1)

    poi_fitter = None
    if cfg.use_poi_vec:
        poi_fitter = PoiFitter().fit(train_samples)
        X_tr = np.concatenate([X_tr, _poi_vectors(train_samples, poi_fitter)], axis=1)
        X_va = np.concatenate([X_va, _poi_vectors(val_samples, poi_fitter)], axis=1)

    y_tr_raw = np.array(
        [[s["targets_raw"][d] for d in DOMAINS] for s in train_samples], dtype=np.float32,
    )
    y_va_raw = np.array(
        [[s["targets_raw"][d] for d in DOMAINS] for s in val_samples], dtype=np.float32,
    )

    if cfg.use_sar_lag and neighbors is not None:
        dz_y_accum: dict = defaultdict(list)
        for s, y_row in zip(train_samples, y_tr_raw):
            dz_y_accum[s["datazone"]].append(y_row)
        dz_to_y = {dz: np.mean(ys, axis=0) for dz, ys in dz_y_accum.items()}
        global_mean = y_tr_raw.mean(axis=0)
        sar_tr = _target_lag_features(train_samples, neighbors, dz_to_y, global_mean)
        sar_va = _target_lag_features(val_samples, neighbors, dz_to_y, global_mean)
        X_tr = np.concatenate([X_tr, sar_tr], axis=1)
        X_va = np.concatenate([X_va, sar_va], axis=1)

    if cfg.use_domain_indicators:
        X_tr = np.concatenate([X_tr, _indicator_features(train_samples)], axis=1)
        X_va = np.concatenate([X_va, _indicator_features(val_samples)], axis=1)

    if cfg.use_latlon:
        X_tr = np.concatenate([X_tr, _latlon_features(train_samples)], axis=1)
        X_va = np.concatenate([X_va, _latlon_features(val_samples)], axis=1)

    return X_tr, X_va, y_tr_raw, y_va_raw, poi_fitter


def _mean_spearman(pred, target):
    per = {d: spearmanr(target[:, k].tolist(), pred[:, k].tolist())
           for k, d in enumerate(DOMAINS)}
    return float(np.mean(list(per.values()))), per


def run_cv_lgbm(
    dataset_path, out_dir, cfg: RouteCConfig, lgbm_h: LGBMHyperparams,
    n_splits: int = 5, shp_path=DEFAULT_SHP,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    print(f"[data] loaded {len(samples)} samples from {dataset_path}")

    cache_path = out_dir / "caption_cache.pt"
    cache_dzs, X = build_or_load_captions(samples, cache_path, cfg)
    if cfg.use_dz_agg:
        cache_dzs, X, dz_index = _aggregate_X_by_dz(samples, X)
        print(f"[dz_agg] aggregated to {len(cache_dzs)} unique datazones (X={X.shape})")
    else:
        dz_index = {dz: i for i, dz in enumerate(cache_dzs)}

    neighbors = None
    if cfg.use_spatial_lag:
        neighbors = build_neighbors(shp_path)
        print(f"[spatial] neighbour map: {len(neighbors)} datazones")

    by_dz = {s["datazone"]: s for s in samples}
    dataset_dzs = [s["datazone"] for s in samples]

    oof_rows: list[dict] = []
    fold_summaries: list[dict] = []

    for fold_idx, (train_dz, val_dz) in enumerate(
        group_kfold_by_iz(dataset_dzs, n_splits=n_splits, seed=cfg.seed)
    ):
        train_samples = [by_dz[d] for d in train_dz if d in by_dz]
        val_samples = [by_dz[d] for d in val_dz if d in by_dz]
        print(f"[fold {fold_idx}] train={len(train_samples)}  val={len(val_samples)}")

        X_tr, X_va, y_tr, y_va, poi_fitter = _build_features(
            train_samples, val_samples, X, dz_index, cfg, neighbors,
        )
        if lgbm_h.sbert_pca_dim > 0:
            d_before = X_tr.shape[1]
            X_tr, X_va = _apply_sbert_pca(
                X_tr, X_va, cfg, lgbm_h.sbert_pca_dim, cfg.seed,
            )
            print(f"[fold {fold_idx}] SBERT PCA→{lgbm_h.sbert_pca_dim}: "
                  f"X dim {d_before} → {X_tr.shape[1]}")
        Y_tr = normalise_array(y_tr)
        Y_va = normalise_array(y_va)

        model = _build_model(lgbm_h, cfg.seed)
        model.fit(X_tr, Y_tr, X_va, Y_va,
                  early_stopping_rounds=lgbm_h.early_stopping_rounds)

        val_pred_raw = denormalise_array(model.predict(X_va))
        val_score, per_domain = _mean_spearman(val_pred_raw, y_va)
        print(f"[fold {fold_idx}] val mean Spearman={val_score:.4f}  "
              f"best_iters={model.best_iters}")

        fold_dir = out_dir / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save(fold_dir)
        if poi_fitter is not None:
            poi_fitter.save(fold_dir / "poi_fitter.json")

        for dz, pred_raw, tgt_raw in zip(
            [s["datazone"] for s in val_samples], val_pred_raw, y_va,
        ):
            oof_rows.append({
                "datazone": dz,
                "fold": fold_idx,
                "prediction_json": {
                    d: float(round(float(pred_raw[k]), 4)) for k, d in enumerate(DOMAINS)
                },
                "target_raw": {d: float(tgt_raw[k]) for k, d in enumerate(DOMAINS)},
            })

        fold_summaries.append({
            "fold": fold_idx,
            "val_mean_spearman": val_score,
            "val_per_domain_spearman": per_domain,
            "best_iters": model.best_iters,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
        })

    oof_path = out_dir / "oof_predictions.jsonl"
    with open(oof_path, "w") as f:
        for row in oof_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "dataset": str(dataset_path),
        "n_samples": len(samples),
        "n_folds": n_splits,
        "d_caption": int(X.shape[1]),
        "config": asdict(cfg),
        "lgbm": asdict(lgbm_h),
        "folds": fold_summaries,
        "oof_mean_spearman": float(np.mean([f["val_mean_spearman"] for f in fold_summaries])),
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(oof_rows)} OOF rows → {oof_path}")
    print(f"[done] OOF mean Spearman = {summary['oof_mean_spearman']:.4f}")
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    y = _load_yaml(args.config)
    cap_y = y.get("caption", {})
    enc_y = y.get("encoder", {})
    cv_y = y.get("cv", {})
    tr_y = y.get("train", {})
    lgbm_y = y.get("lgbm", {})

    def _g(yv, default):
        return yv if yv is not None else default

    cfg = RouteCConfig(
        caption_mode=_g(cap_y.get("mode"), "concat"),
        encoder_backend=_g(enc_y.get("backend"), "sbert"),
        encoder_name=_g(enc_y.get("name"),
                        "sentence-transformers/all-MiniLM-L6-v2"),
        encoder_batch_size=_g(enc_y.get("batch_size"), 32),
        encoder_max_length=_g(enc_y.get("max_length"), 256),
        regressor="lgbm",  # informational only
        use_poi_vec=_g(tr_y.get("use_poi_vec"), False),
        use_spatial_lag=_g(tr_y.get("use_spatial_lag"), False),
        use_ego_gap=_g(tr_y.get("use_ego_gap"), False),
        use_sar_lag=_g(tr_y.get("use_sar_lag"), False),
        use_dz_agg=_g(tr_y.get("use_dz_agg"), False),
        use_domain_indicators=_g(tr_y.get("use_domain_indicators"), False),
        use_latlon=_g(tr_y.get("use_latlon"), False),
        seed=_g(tr_y.get("seed"), 42),
    )

    lgbm_h = LGBMHyperparams(
        backend=_g(lgbm_y.get("backend"), "lgbm"),
        sbert_pca_dim=_g(lgbm_y.get("sbert_pca_dim"), 0),
        n_estimators=_g(lgbm_y.get("n_estimators"), 2000),
        learning_rate=_g(lgbm_y.get("learning_rate"), 0.03),
        num_leaves=_g(lgbm_y.get("num_leaves"), 31),
        min_data_in_leaf=_g(lgbm_y.get("min_data_in_leaf"), 20),
        feature_fraction=_g(lgbm_y.get("feature_fraction"), 0.8),
        bagging_fraction=_g(lgbm_y.get("bagging_fraction"), 0.8),
        bagging_freq=_g(lgbm_y.get("bagging_freq"), 5),
        lambda_l2=_g(lgbm_y.get("lambda_l2"), 1.0),
        objective=_g(lgbm_y.get("objective"), "regression"),
        early_stopping_rounds=_g(lgbm_y.get("early_stopping_rounds"), 100),
    )
    n_splits = _g(cv_y.get("n_splits"), 5)

    run_cv_lgbm(
        dataset_path=y["dataset"],
        out_dir=y["out_dir"],
        cfg=cfg,
        lgbm_h=lgbm_h,
        n_splits=n_splits,
    )


if __name__ == "__main__":
    _main()
