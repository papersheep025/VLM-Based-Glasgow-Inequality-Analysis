"""SAR-only and residual-after-SAR diagnostics for Route C.

The diagnostics answer two related questions:

* sar_only: how much of the decision-layer score is explained by train-neighbour
  target lag alone?
* residual_after_sar: after a SAR-only first stage, can non-SAR Route C features
  explain the remaining target residual?

Both modes write the standard decision ``oof_predictions.jsonl`` schema so they
can be compared with ``python -m decision.eval.compare``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.base import clone
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", message="Singular matrix.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision.data.poi_features import PoiFitter  # noqa: E402
from decision.data.spatial_neighbors import build_neighbors  # noqa: E402
from decision.data.targets import DOMAINS  # noqa: E402
from decision.models.route_c.regressors import build_regressor  # noqa: E402
from decision.train.cv_runner import load_dataset  # noqa: E402
from decision.train.cv_runner_caption import (  # noqa: E402
    _aggregate_X_by_dz,
    build_or_load_captions,
)
from decision.train.route_c_train import (  # noqa: E402
    RouteCConfig,
    _indicator_features,
    _latlon_features,
    _neighbor_features,
    _poi_vectors,
    _target_lag_features,
)
from src.glasgow_vlm.metrics import spearmanr  # noqa: E402
from src.glasgow_vlm.splits import DEFAULT_SHP, group_kfold_by_iz  # noqa: E402


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _cfg_from_yaml(path: Path) -> tuple[RouteCConfig, Path, int]:
    y = _load_yaml(path)
    cap_y = y.get("caption", {})
    enc_y = y.get("encoder", {})
    tr_y = y.get("train", {})
    cv_y = y.get("cv", {})
    cfg = RouteCConfig(
        caption_mode=cap_y.get("mode", "modality_sep"),
        encoder_backend=enc_y.get("backend", "sbert"),
        encoder_name=enc_y.get("name", "sentence-transformers/all-MiniLM-L6-v2"),
        encoder_batch_size=enc_y.get("batch_size", 32),
        encoder_max_length=enc_y.get("max_length", 256),
        regressor=tr_y.get("regressor", "ridge_cv"),
        use_poi_vec=tr_y.get("use_poi_vec", True),
        use_spatial_lag=tr_y.get("use_spatial_lag", True),
        use_ego_gap=tr_y.get("use_ego_gap", True),
        use_sar_lag=False,
        use_dz_agg=tr_y.get("use_dz_agg", True),
        use_domain_indicators=tr_y.get("use_domain_indicators", False),
        use_latlon=tr_y.get("use_latlon", False),
        seed=tr_y.get("seed", 42),
    )
    return cfg, Path(y["dataset"]), int(cv_y.get("n_splits", 5))


def _mean_spearman(pred: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
    per = {
        d: float(spearmanr(target[:, k].tolist(), pred[:, k].tolist()))
        for k, d in enumerate(DOMAINS)
    }
    return float(np.mean(list(per.values()))), per


def _targets(samples: list[dict]) -> np.ndarray:
    return np.array(
        [[s["targets_raw"][d] for d in DOMAINS] for s in samples],
        dtype=np.float32,
    )


def _sar_matrix(
    train_samples: list[dict],
    eval_samples: list[dict],
    neighbors: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_tr = _targets(train_samples)
    y_ev = _targets(eval_samples)

    from collections import defaultdict

    dz_y_accum: dict[str, list[np.ndarray]] = defaultdict(list)
    for s, y_row in zip(train_samples, y_tr):
        dz_y_accum[s["datazone"]].append(y_row)
    dz_to_y = {dz: np.mean(ys, axis=0) for dz, ys in dz_y_accum.items()}
    global_mean = y_tr.mean(axis=0)

    sar_tr = _target_lag_features(train_samples, neighbors, dz_to_y, global_mean)
    sar_ev = _target_lag_features(eval_samples, neighbors, dz_to_y, global_mean)
    return sar_tr, sar_ev, y_tr, y_ev


def _non_sar_features(
    train_samples: list[dict],
    val_samples: list[dict],
    X_all: np.ndarray,
    dz_index: dict[str, int],
    cfg: RouteCConfig,
    neighbors: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Assemble the Route C feature stack with SAR intentionally excluded."""
    def idx(samples: list[dict]) -> np.ndarray:
        rows = [dz_index[s["datazone"]] for s in samples]
        return X_all[np.array(rows, dtype=np.int64)].astype(np.float32)

    X_tr = idx(train_samples)
    X_va = idx(val_samples)
    slices = {"text_self": [0, int(X_tr.shape[1])]}

    if cfg.use_spatial_lag:
        lag_tr = _neighbor_features(train_samples, X_all, dz_index, neighbors)
        lag_va = _neighbor_features(val_samples, X_all, dz_index, neighbors)
        start = X_tr.shape[1]
        if cfg.use_ego_gap:
            X_tr = np.concatenate([X_tr, X_tr - lag_tr], axis=1)
            X_va = np.concatenate([X_va, X_va - lag_va], axis=1)
            slices["text_ego_gap"] = [start, int(X_tr.shape[1])]
        else:
            X_tr = np.concatenate([X_tr, lag_tr], axis=1)
            X_va = np.concatenate([X_va, lag_va], axis=1)
            slices["text_spatial_lag"] = [start, int(X_tr.shape[1])]

    if cfg.use_poi_vec:
        start = X_tr.shape[1]
        poi_fitter = PoiFitter().fit(train_samples)
        X_tr = np.concatenate([X_tr, _poi_vectors(train_samples, poi_fitter)], axis=1)
        X_va = np.concatenate([X_va, _poi_vectors(val_samples, poi_fitter)], axis=1)
        slices["poi_vec"] = [start, int(X_tr.shape[1])]

    if cfg.use_domain_indicators:
        start = X_tr.shape[1]
        X_tr = np.concatenate([X_tr, _indicator_features(train_samples)], axis=1)
        X_va = np.concatenate([X_va, _indicator_features(val_samples)], axis=1)
        slices["domain_indicators"] = [start, int(X_tr.shape[1])]

    if cfg.use_latlon:
        start = X_tr.shape[1]
        X_tr = np.concatenate([X_tr, _latlon_features(train_samples)], axis=1)
        X_va = np.concatenate([X_va, _latlon_features(val_samples)], axis=1)
        slices["latlon"] = [start, int(X_tr.shape[1])]

    return X_tr, X_va, slices


def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray,
                 regressor: str, seed: int) -> np.ndarray:
    base = build_regressor(regressor, seed=seed)
    pred = np.zeros((X_va.shape[0], len(DOMAINS)), dtype=np.float32)
    for k in range(len(DOMAINS)):
        model = clone(base)
        model.fit(X_tr, y_tr[:, k])
        pred[:, k] = np.asarray(model.predict(X_va)).reshape(-1)
    return pred


def run_diagnostic(
    mode: str,
    config_path: Path,
    out_dir: Path,
    cache_source: Path | None = None,
    shp_path: Path = Path(DEFAULT_SHP),
) -> dict:
    cfg, dataset_path, n_splits = _cfg_from_yaml(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    by_dz = {s["datazone"]: s for s in samples}
    dataset_dzs = [s["datazone"] for s in samples]
    neighbors = build_neighbors(shp_path)
    print(f"[data] loaded {len(samples)} rows from {dataset_path}")
    print(f"[spatial] neighbour map={len(neighbors)} datazones")

    X_all = None
    dz_index = None
    feature_slices = {"sar_lag": [0, len(DOMAINS)]}
    if mode == "residual_after_sar":
        cache_path = out_dir / "caption_cache.pt"
        if cache_source and cache_source.exists() and not cache_path.exists():
            shutil.copy2(cache_source, cache_path)
            print(f"[cache] copied {cache_source} -> {cache_path}")
        cache_dzs, X_text = build_or_load_captions(samples, cache_path, cfg)
        if cfg.use_dz_agg:
            cache_dzs, X_text, dz_index = _aggregate_X_by_dz(samples, X_text)
        else:
            dz_index = {dz: i for i, dz in enumerate(cache_dzs)}
        X_all = X_text.astype(np.float32)
        print(f"[features] residual feature base X={X_all.shape}")

    oof_rows: list[dict] = []
    fold_summaries: list[dict] = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_sar_pred: list[np.ndarray] = []

    for fold_idx, (train_dz, val_dz) in enumerate(
        group_kfold_by_iz(dataset_dzs, n_splits=n_splits, seed=cfg.seed)
    ):
        train_samples = [by_dz[d] for d in train_dz if d in by_dz]
        val_samples = [by_dz[d] for d in val_dz if d in by_dz]
        sar_tr, sar_va, y_tr, y_va = _sar_matrix(train_samples, val_samples, neighbors)

        sar_tr_pred = _fit_predict(sar_tr, y_tr, sar_tr, cfg.regressor, cfg.seed)
        sar_va_pred = _fit_predict(sar_tr, y_tr, sar_va, cfg.regressor, cfg.seed)
        if mode == "sar_only":
            pred = sar_va_pred
        elif mode == "residual_after_sar":
            assert X_all is not None and dz_index is not None
            X_tr, X_va, slices = _non_sar_features(
                train_samples, val_samples, X_all, dz_index, cfg, neighbors,
            )
            feature_slices = {"sar_first_stage": [0, len(DOMAINS)], **slices}
            residual_tr = y_tr - sar_tr_pred
            residual_pred = _fit_predict(X_tr, residual_tr, X_va, cfg.regressor, cfg.seed)
            pred = sar_va_pred + residual_pred
        else:
            raise ValueError(f"Unknown mode: {mode}")

        fold_mean, fold_per = _mean_spearman(pred, y_va)
        sar_mean, sar_per = _mean_spearman(sar_va_pred, y_va)
        print(
            f"[fold {fold_idx}] train={len(train_samples)} val={len(val_samples)} "
            f"sar ρ={sar_mean:.4f} final ρ={fold_mean:.4f}"
        )

        for dz, p_row, y_row in zip(val_dz, pred, y_va):
            oof_rows.append({
                "datazone": dz,
                "fold": fold_idx,
                "prediction_json": {
                    d: float(round(float(p_row[k]), 4)) for k, d in enumerate(DOMAINS)
                },
                "target_raw": {
                    d: float(y_row[k]) for k, d in enumerate(DOMAINS)
                },
            })

        fold_summaries.append({
            "fold": fold_idx,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "sar_mean_spearman": sar_mean,
            "sar_per_domain_spearman": sar_per,
            "val_mean_spearman": fold_mean,
            "val_per_domain_spearman": fold_per,
        })
        all_pred.append(pred)
        all_true.append(y_va)
        all_sar_pred.append(sar_va_pred)

    pred_all = np.concatenate(all_pred, axis=0)
    true_all = np.concatenate(all_true, axis=0)
    sar_pred_all = np.concatenate(all_sar_pred, axis=0)
    pooled_mean, pooled_per = _mean_spearman(pred_all, true_all)
    sar_pooled_mean, sar_pooled_per = _mean_spearman(sar_pred_all, true_all)

    oof_path = out_dir / "oof_predictions.jsonl"
    with open(oof_path, "w") as f:
        for row in oof_rows:
            f.write(json.dumps(row) + "\n")
    (out_dir / "feature_slices.json").write_text(json.dumps(feature_slices, indent=2))

    summary = {
        "mode": mode,
        "dataset": str(dataset_path),
        "n_samples": len(samples),
        "n_folds": n_splits,
        "config": asdict(cfg),
        "folds": fold_summaries,
        "sar_stage_pooled_mean_spearman": sar_pooled_mean,
        "sar_stage_pooled_per_domain_spearman": sar_pooled_per,
        "pooled_mean_spearman": pooled_mean,
        "pooled_per_domain_spearman": pooled_per,
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(oof_rows)} rows -> {oof_path}")
    print(f"[done] pooled final ρ={pooled_mean:.4f}; sar-stage ρ={sar_pooled_mean:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["sar_only", "residual_after_sar"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cache-source", default=None, type=Path,
                        help="Optional caption_cache.pt to copy before encoding.")
    parser.add_argument("--shp", default=Path(DEFAULT_SHP), type=Path)
    args = parser.parse_args()
    run_diagnostic(
        mode=args.mode,
        config_path=args.config,
        out_dir=args.out_dir,
        cache_source=args.cache_source,
        shp_path=args.shp,
    )


if __name__ == "__main__":
    main()
