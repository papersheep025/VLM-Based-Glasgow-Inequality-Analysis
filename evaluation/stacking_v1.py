"""Stacking ensemble (Level-2 Ridge over Level-1 SBERT-Ridge + structured-HGB).

Goal: combine two complementary signal paths
  Level 1a: SBERT-RidgeCV (already cached at modality_sep_v1/oof_predictions)
  Level 1b: HGB on structured features only (POI + SAR_lag + indicators + SVF)
  Level 2:  Ridge meta-learner on concat([L1a_pred, L1b_pred]) = 14-dim

Since both L1a and L1b OOF are leak-safe (each datazone's prediction comes from
a fold that did not see it), the L2 meta can re-use the same 5-fold partition
without leakage — train rows see only OOF predictions, never their own folds.

Output:
  outputs/decision/route_c/stacking_v1/oof_predictions.jsonl
  outputs/decision/route_c/stacking_v1/cv_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision.data.poi_features import PoiFitter  # noqa: E402
from decision.data.spatial_neighbors import build_neighbors  # noqa: E402
from decision.data.targets import (  # noqa: E402
    DOMAINS, denormalise_array, normalise_array,
)
from decision.train.cv_runner import load_dataset  # noqa: E402
from decision.train.route_c_train import (  # noqa: E402
    _indicator_features, _poi_vectors, _target_lag_features,
)
from src.glasgow_vlm.metrics import spearmanr  # noqa: E402
from src.glasgow_vlm.splits import DEFAULT_SHP, group_kfold_by_iz  # noqa: E402


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_ridge_oof(path: Path) -> dict[str, np.ndarray]:
    """Load Level-1a OOF predictions → {datazone: (7,) array in raw 1-10 space}."""
    out: dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dz = obj["datazone"]
            pred = np.array(
                [float(obj["prediction_json"][d]) for d in DOMAINS], dtype=np.float32
            )
            out[dz] = pred
    return out


def _load_svf(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    return df.set_index("datazone")


def _svf_block(samples: list[dict], svf_dfs: list[pd.DataFrame],
               cols_per_df: list[list[str]]) -> np.ndarray:
    """Return (n_samples, sum(cols)) matrix; missing → 0 placeholder filled later."""
    n = len(samples)
    total_cols = sum(len(c) for c in cols_per_df)
    out = np.full((n, total_cols), np.nan, dtype=np.float32)
    for i, s in enumerate(samples):
        cur = 0
        for df, cols in zip(svf_dfs, cols_per_df):
            if s["datazone"] in df.index:
                vals = df.loc[s["datazone"], cols].to_numpy(dtype=np.float32)
                out[i, cur:cur + len(cols)] = vals
            cur += len(cols)
    return out


# ---------------------------------------------------------------------------
# Feature assembly (structured-only path)
# ---------------------------------------------------------------------------

def _build_structured(
    train_samples, val_samples, neighbors,
    svf_dfs: list[pd.DataFrame], svf_cols_per_df: list[list[str]],
):
    """[POI(13) + SAR_lag(7) + indicators(18) + SVF(...)]   — no SBERT."""
    poi_fitter = PoiFitter().fit(train_samples)
    poi_tr = _poi_vectors(train_samples, poi_fitter)
    poi_va = _poi_vectors(val_samples, poi_fitter)

    y_tr_raw = np.array(
        [[s["targets_raw"][d] for d in DOMAINS] for s in train_samples],
        dtype=np.float32,
    )
    y_va_raw = np.array(
        [[s["targets_raw"][d] for d in DOMAINS] for s in val_samples],
        dtype=np.float32,
    )

    dz_y_accum: dict = defaultdict(list)
    for s, y_row in zip(train_samples, y_tr_raw):
        dz_y_accum[s["datazone"]].append(y_row)
    dz_to_y = {dz: np.mean(ys, axis=0) for dz, ys in dz_y_accum.items()}
    global_mean = y_tr_raw.mean(axis=0)
    sar_tr = _target_lag_features(train_samples, neighbors, dz_to_y, global_mean)
    sar_va = _target_lag_features(val_samples, neighbors, dz_to_y, global_mean)

    ind_tr = _indicator_features(train_samples)
    ind_va = _indicator_features(val_samples)

    svf_tr = _svf_block(train_samples, svf_dfs, svf_cols_per_df)
    svf_va = _svf_block(val_samples, svf_dfs, svf_cols_per_df)
    # Impute SVF NaNs with train-fold column mean (leak-safe)
    if svf_tr.size > 0:
        col_mean = np.nanmean(svf_tr, axis=0)
        col_mean = np.nan_to_num(col_mean, nan=0.0)
        for c in range(svf_tr.shape[1]):
            svf_tr[np.isnan(svf_tr[:, c]), c] = col_mean[c]
            svf_va[np.isnan(svf_va[:, c]), c] = col_mean[c]

    X_tr = np.concatenate([poi_tr, sar_tr, ind_tr, svf_tr], axis=1)
    X_va = np.concatenate([poi_va, sar_va, ind_va, svf_va], axis=1)
    return X_tr, X_va, y_tr_raw, y_va_raw


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_stacking(
    dataset_path: Path,
    ridge_oof_path: Path,
    svf_parquets: list[Path],
    out_dir: Path,
    n_splits: int = 5,
    shp_path: Path = Path(DEFAULT_SHP),
    seed: int = 42,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    print(f"[data] {len(samples)} samples")
    ridge_oof = _load_ridge_oof(ridge_oof_path)
    print(f"[L1a] loaded ridge OOF for {len(ridge_oof)} datazones")

    svf_dfs, svf_cols_per_df = [], []
    for pq in svf_parquets:
        df = _load_svf(pq)
        cols = sorted([c for c in df.columns if c.endswith("_mean")])
        svf_dfs.append(df)
        svf_cols_per_df.append(cols)
        print(f"[SVF] {pq.name}: {len(cols)} cols")

    neighbors = build_neighbors(shp_path)
    by_dz = {s["datazone"]: s for s in samples}
    dataset_dzs = [s["datazone"] for s in samples]

    # ----- Pass 1: produce L1b (HGB-on-structured) OOF -----
    n = len(samples)
    n_dom = len(DOMAINS)
    l1b_pred = np.full((n, n_dom), np.nan, dtype=np.float32)
    l1a_pred = np.full((n, n_dom), np.nan, dtype=np.float32)
    y_raw_all = np.zeros((n, n_dom), dtype=np.float32)
    sample_dzs = [s["datazone"] for s in samples]
    sample_idx = {dz_pos: pos for pos, dz_pos in enumerate(sample_dzs)}

    for fold_idx, (train_dz, val_dz) in enumerate(
        group_kfold_by_iz(dataset_dzs, n_splits=n_splits, seed=seed)
    ):
        train_samples = [by_dz[d] for d in train_dz if d in by_dz]
        val_samples = [by_dz[d] for d in val_dz if d in by_dz]
        X_tr, X_va, y_tr, y_va = _build_structured(
            train_samples, val_samples, neighbors, svf_dfs, svf_cols_per_df,
        )
        Y_tr = normalise_array(y_tr)

        # HGB per-domain
        hgb_pred = np.zeros_like(y_va)
        for k in range(n_dom):
            m = HistGradientBoostingRegressor(
                max_iter=2000, learning_rate=0.03, max_leaf_nodes=31,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=100, random_state=seed,
            )
            m.fit(X_tr, Y_tr[:, k])
            hgb_pred[:, k] = m.predict(X_va)
        hgb_pred_raw = denormalise_array(hgb_pred)

        # Write into global OOF arrays (L1b on val rows; L1a from cache)
        # Build a per-row index since dataset_v1 has duplicates (_extra patches)
        positions = []
        for s in val_samples:
            # Find ALL rows in samples with this datazone (might be multiple)
            # Just match the first unfilled occurrence to keep 1:1 with val_samples
            positions.append(s)

        # Use a flat index over samples list (preserve order)
        pos_in_samples = {id(s): i for i, s in enumerate(samples)}
        for i_val, s in enumerate(val_samples):
            pos = pos_in_samples[id(s)]
            l1b_pred[pos] = hgb_pred_raw[i_val]
            l1a_pred[pos] = ridge_oof.get(s["datazone"], np.zeros(n_dom, np.float32))
            y_raw_all[pos] = y_va[i_val]

        per_fold_score = float(np.mean([
            spearmanr(y_va[:, k].tolist(), hgb_pred_raw[:, k].tolist())
            for k in range(n_dom)
        ]))
        print(f"[L1b fold {fold_idx}] val Spearman (HGB structured)={per_fold_score:.4f}")

    # ----- Pass 2: meta Ridge on concat(L1a, L1b) -----
    meta_X = np.concatenate([l1a_pred, l1b_pred], axis=1)  # (n, 14)
    meta_pred = np.full_like(y_raw_all, np.nan)

    fold_summaries = []
    for fold_idx, (train_dz, val_dz) in enumerate(
        group_kfold_by_iz(dataset_dzs, n_splits=n_splits, seed=seed)
    ):
        train_samples = [by_dz[d] for d in train_dz if d in by_dz]
        val_samples = [by_dz[d] for d in val_dz if d in by_dz]
        train_pos = [pos_in_samples[id(s)] for s in train_samples]
        val_pos = [pos_in_samples[id(s)] for s in val_samples]

        Y_tr = normalise_array(y_raw_all[train_pos])
        # 7 independent meta-Ridges
        for k in range(n_dom):
            m = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=5)
            m.fit(meta_X[train_pos], Y_tr[:, k])
            meta_pred[val_pos, k] = m.predict(meta_X[val_pos])
        meta_pred_raw = denormalise_array(meta_pred[val_pos])

        score_per = {
            d: float(spearmanr(y_raw_all[val_pos, k].tolist(),
                               meta_pred_raw[:, k].tolist()))
            for k, d in enumerate(DOMAINS)
        }
        score = float(np.mean(list(score_per.values())))
        fold_summaries.append({"fold": fold_idx, "val_mean_spearman": score,
                               "val_per_domain_spearman": score_per})
        print(f"[L2  fold {fold_idx}] val mean Spearman (stacking)={score:.4f}")
        # Replace meta_pred[val_pos] with raw-space for output writing later
        meta_pred[val_pos] = meta_pred_raw

    # ----- Write OOF -----
    oof_path = out_dir / "oof_predictions.jsonl"
    with open(oof_path, "w") as f:
        for pos, s in enumerate(samples):
            row = {
                "datazone": s["datazone"],
                "fold": -1,  # stacking is not single-fold; preserved for compatibility
                "prediction_json": {
                    d: float(round(float(meta_pred[pos, k]), 4))
                    for k, d in enumerate(DOMAINS)
                },
                "target_raw": {
                    d: float(y_raw_all[pos, k]) for k, d in enumerate(DOMAINS)
                },
            }
            f.write(json.dumps(row) + "\n")

    pooled = {
        d: float(spearmanr(
            y_raw_all[:, k].tolist(), meta_pred[:, k].tolist()
        )) for k, d in enumerate(DOMAINS)
    }
    pooled_mean = float(np.mean(list(pooled.values())))
    summary = {
        "dataset": str(dataset_path),
        "ridge_oof": str(ridge_oof_path),
        "svf_parquets": [str(p) for p in svf_parquets],
        "n_samples": n,
        "n_folds": n_splits,
        "fold_summaries": fold_summaries,
        "pooled_per_domain": pooled,
        "pooled_mean": pooled_mean,
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] OOF rows={n} → {oof_path}")
    print(f"[done] pooled mean Spearman = {pooled_mean:.4f}")
    print(f"[done] per-domain: {pooled}")
    return summary


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/decision/dataset_v1.jsonl", type=Path)
    parser.add_argument("--ridge-oof", required=True, type=Path,
                        help="Level-1a OOF JSONL (e.g. modality_sep_v1)")
    parser.add_argument("--svf", nargs="*", default=[], type=Path,
                        help="Optional SVF parquet(s); use *_mean cols.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_stacking(
        dataset_path=args.dataset,
        ridge_oof_path=args.ridge_oof,
        svf_parquets=args.svf,
        out_dir=args.out_dir,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
