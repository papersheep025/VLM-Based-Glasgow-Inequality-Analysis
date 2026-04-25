"""5-fold IZ-grouped CV runner for Route C with semantic-segmentation SVF.

Design
------
- Reuses ``build_or_load_captions`` / ``_aggregate_X_by_dz`` / ``RouteCConfig``
  / ``train_fold_caption`` / ``group_kfold_by_iz`` / ``build_neighbors`` from
  the existing modules **without modification**.
- Adds a single step: after the text embedding matrix ``X_text`` is built, load
  one or more SVF parquet files and concatenate them to the column dimension of
  ``X_text``. The resulting ``X_all`` is handed to ``train_fold_caption``
  transparently, so existing spatial lag / ego-gap / dz-agg automatically
  cover SVF.

Missing-datazone policy (leak-safe):
    - If a datazone has no SVF row (no street view), it is filled with
      ``NaN`` at the global level.
    - In each fold, missing entries are then imputed with the per-column
      **train-fold mean**, computed from the fold's training datazones only.

CLI:
    python -m decision.train.cv_runner_svf \
        --config decision/configs/svf/route_c_modality_sep_svf_segformer_v0.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from decision.data.spatial_neighbors import build_neighbors
from decision.data.targets import DOMAINS
from decision.train.cv_runner import load_dataset
from decision.train.cv_runner_caption import (
    _aggregate_X_by_dz,
    _load_yaml,
    build_or_load_captions,
)
from decision.train.route_c_train import RouteCConfig, train_fold_caption
from src.glasgow_vlm.splits import DEFAULT_SHP, group_kfold_by_iz


# ---------------------------------------------------------------------------
# SVF loading + alignment
# ---------------------------------------------------------------------------

def _select_svf_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Resolve ``columns`` spec ({'mean', 'std', explicit names}) into actual
    parquet columns. Unknown specs raise.
    """
    available = set(df.columns)
    resolved: list[str] = []
    for spec in columns:
        if spec == "mean":
            resolved.extend(sorted([c for c in available if c.endswith("_mean")]))
        elif spec == "std":
            resolved.extend(sorted([c for c in available if c.endswith("_std")]))
        elif spec in available:
            resolved.append(spec)
        else:
            raise ValueError(f"SVF column spec {spec!r} not found in parquet.")
    # Preserve insertion order, de-duplicate.
    seen, uniq = set(), []
    for c in resolved:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def load_svf_matrix(
    parquet_paths: list[Path],
    column_specs: list[str],
    dz_index: dict[str, int],
) -> np.ndarray:
    """Build an (n_dz, D) SVF matrix aligned to ``dz_index``.

    - If multiple parquet files are given, their resolved columns are
      concatenated along axis=1 in the order supplied.
    - Datazones absent from a parquet receive NaN for that block.
    """
    n_dz = len(dz_index)
    blocks: list[np.ndarray] = []
    for pq_path in parquet_paths:
        df = pd.read_parquet(pq_path)
        cols = _select_svf_columns(df, column_specs)
        block = np.full((n_dz, len(cols)), np.nan, dtype=np.float32)
        dz_to_row = {dz: i for i, dz in enumerate(df["datazone"].tolist())}
        for dz, row in dz_index.items():
            j = dz_to_row.get(dz)
            if j is not None:
                block[row] = df.iloc[j][cols].to_numpy(dtype=np.float32)
        print(f"[svf] {pq_path.name}: {len(cols)} cols, "
              f"coverage={(~np.isnan(block).any(axis=1)).sum()}/{n_dz}")
        blocks.append(block)
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((n_dz, 0), np.float32)


def _impute_fold(
    X_full: np.ndarray,
    svf_start: int,
    dz_index: dict[str, int],
    train_dz: list[str],
) -> np.ndarray:
    """Copy ``X_full`` and fill NaNs in the SVF block (columns ``>= svf_start``)
    with the train-fold column mean."""
    X = X_full.copy()
    svf = X[:, svf_start:]
    if svf.size == 0:
        return X
    train_rows = [dz_index[d] for d in train_dz if d in dz_index]
    train_block = svf[train_rows]
    col_means = np.nanmean(train_block, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)   # all-NaN columns → 0
    nan_mask = np.isnan(svf)
    for c in range(svf.shape[1]):
        col_nan = nan_mask[:, c]
        if col_nan.any():
            svf[col_nan, c] = col_means[c]
    X[:, svf_start:] = svf
    return X


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------

def run_cv_svf(
    dataset_path: str | Path,
    out_dir: str | Path,
    cfg: RouteCConfig,
    svf_parquets: list[Path],
    svf_columns: list[str],
    n_splits: int = 5,
    shp_path: str | Path = DEFAULT_SHP,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    print(f"[data] loaded {len(samples)} samples from {dataset_path}")

    cache_path = out_dir / "caption_cache.pt"
    cache_dzs, X_text = build_or_load_captions(samples, cache_path, cfg)

    if cfg.use_dz_agg:
        cache_dzs, X_text, dz_index = _aggregate_X_by_dz(samples, X_text)
        print(f"[dz_agg] aggregated to {len(cache_dzs)} unique datazones "
              f"(X shape={X_text.shape})")
    else:
        dz_index = {dz: i for i, dz in enumerate(cache_dzs)}

    # --- SVF injection (new) ---
    text_dim = X_text.shape[1]
    svf_matrix = load_svf_matrix(svf_parquets, svf_columns, dz_index)
    X_all = np.concatenate([X_text, svf_matrix], axis=1).astype(np.float32)
    svf_start = text_dim
    print(f"[svf] X_text={X_text.shape}  svf={svf_matrix.shape}  "
          f"X_all={X_all.shape}")
    # ---------------------------

    neighbors = None
    if cfg.use_spatial_lag:
        neighbors = build_neighbors(shp_path)
        n_with_neighbors = sum(1 for v in neighbors.values() if v)
        print(f"[spatial] neighbour map: {len(neighbors)} datazones, "
              f"{n_with_neighbors} have >=1 neighbour")

    dataset_dzs = [s["datazone"] for s in samples]
    by_dz = {s["datazone"]: s for s in samples}

    oof_rows: list[dict] = []
    fold_summaries: list[dict] = []

    for fold_idx, (train_dz, val_dz) in enumerate(
        group_kfold_by_iz(dataset_dzs, n_splits=n_splits, seed=cfg.seed)
    ):
        train_samples = [by_dz[d] for d in train_dz if d in by_dz]
        val_samples = [by_dz[d] for d in val_dz if d in by_dz]
        print(f"[fold {fold_idx}] train={len(train_samples)}  val={len(val_samples)}")

        # Impute SVF NaNs with this fold's train mean before handing off.
        X_fold = _impute_fold(X_all, svf_start, dz_index, train_dz)

        fold_dir = out_dir / f"fold{fold_idx}"
        result = train_fold_caption(
            train_samples, val_samples, X_fold, dz_index,
            cfg=cfg, fold_dir=fold_dir, neighbors=neighbors,
        )
        print(f"[fold {fold_idx}] val mean Spearman={result['val_mean_spearman']:.4f}")

        for dz, pred_raw, tgt_raw in zip(
            result["val_datazones"], result["val_pred_raw"], result["val_target_raw"]
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
            "val_mean_spearman": result["val_mean_spearman"],
            "val_per_domain_spearman": result["val_per_domain_spearman"],
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
        "d_caption": int(text_dim),
        "d_svf": int(svf_matrix.shape[1]),
        "svf_parquets": [str(p) for p in svf_parquets],
        "svf_columns_spec": list(svf_columns),
        "config": asdict(cfg),
        "folds": fold_summaries,
        "oof_mean_spearman": float(
            np.mean([f["val_mean_spearman"] for f in fold_summaries])
        ),
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(oof_rows)} OOF rows → {oof_path}")
    print(f"[done] OOF mean Spearman = {summary['oof_mean_spearman']:.4f}")
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    y = _load_yaml(args.config)
    cap_y = y.get("caption", {})
    enc_y = y.get("encoder", {})
    cv_y = y.get("cv", {})
    tr_y = y.get("train", {})
    svf_y = y.get("svf", {})

    def _pick(yv, default):
        return yv if yv is not None else default

    dataset = y.get("dataset")
    out_dir = y.get("out_dir")
    if not dataset or not out_dir:
        parser.error("YAML must specify 'dataset' and 'out_dir'.")

    cfg = RouteCConfig(
        caption_mode=_pick(cap_y.get("mode"), "concat"),
        encoder_backend=_pick(enc_y.get("backend"), "sbert"),
        encoder_name=_pick(enc_y.get("name"),
                           "sentence-transformers/all-MiniLM-L6-v2"),
        encoder_batch_size=_pick(enc_y.get("batch_size"), 32),
        encoder_max_length=_pick(enc_y.get("max_length"), 256),
        regressor=_pick(tr_y.get("regressor"), "ridge_cv"),
        use_poi_vec=_pick(tr_y.get("use_poi_vec"), False),
        use_spatial_lag=_pick(tr_y.get("use_spatial_lag"), False),
        use_ego_gap=_pick(tr_y.get("use_ego_gap"), False),
        use_sar_lag=_pick(tr_y.get("use_sar_lag"), False),
        use_dz_agg=_pick(tr_y.get("use_dz_agg"), False),
        use_domain_indicators=_pick(tr_y.get("use_domain_indicators"), False),
        use_latlon=_pick(tr_y.get("use_latlon"), False),
        seed=_pick(tr_y.get("seed"), 42),
    )
    n_splits = _pick(cv_y.get("n_splits"), 5)

    svf_parquets = [Path(p) for p in svf_y.get("parquets", [])]
    svf_columns = list(svf_y.get("columns", ["mean"]))
    if not svf_parquets:
        print("[warn] no SVF parquets given; running baseline (X_all == X_text).")

    run_cv_svf(
        dataset_path=dataset,
        out_dir=out_dir,
        cfg=cfg,
        svf_parquets=svf_parquets,
        svf_columns=svf_columns,
        n_splits=n_splits,
    )


if __name__ == "__main__":
    _main()
