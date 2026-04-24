"""
5-fold IZ-grouped CV runner for Route C (caption embedding + linear regressor).

Flow
----
1. Load ``dataset_v0.jsonl``.
2. Build one caption string per sample (concat or CityLens-style templated).
3. Encode captions once with BERT mean-pool or Sentence-BERT, cache to
   ``caption_cache.pt``.  Cache is reused across folds and across reruns when
   backend / model / caption-mode match.
4. If use_spatial_lag: build Queen-contiguity neighbour map from shapefile.
5. For each of 5 folds from ``group_kfold_by_iz`` (IZ groups, SIMD-strata):
     - optionally concat neighbour-averaged embeddings (spatial lag),
     - optionally fit a per-fold POI scaler on train,
     - clone a regressor per domain (RidgeCV / LassoCV / ElasticNetCV),
     - fit on raw SIMD scores [1, 10] (no logit normalisation),
     - collect val predictions.
6. Aggregate OOF predictions into ``oof_predictions.jsonl`` using the project's
   standard ``prediction_json`` schema (same as Route A / A').

CLI:
    python -m decision.train.cv_runner_caption \
        --config decision/configs/route_c_caption_embed.yaml

    # enable spatial lag:
    python -m decision.train.cv_runner_caption \
        --config decision/configs/route_c_caption_embed.yaml \
        --use-spatial-lag \
        --out-dir outputs/decision/route_c/sbert_spatial_v0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from decision.data.targets import DOMAINS
from decision.models.route_c.captioner import build_captions, build_captions_modality_sep
from decision.models.route_c.encoder import (
    encode_modality_sep,
    encode_texts,
    load_caption_cache,
    save_caption_cache,
)
from decision.train.cv_runner import load_dataset
from decision.train.route_c_train import RouteCConfig, train_fold_caption
from src.glasgow_vlm.splits import DEFAULT_SHP, group_kfold_by_iz


def _aggregate_X_by_dz(
    samples: list[dict],
    X: np.ndarray,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Mean-pool duplicate rows that share the same datazone.

    Returns (dz_list, X_agg, dz_index) where X_agg is (n_unique_dz, d).
    """
    dz_rows: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        dz_rows[s["datazone"]].append(i)
    dz_list = list(dz_rows.keys())
    X_agg = np.stack(
        [X[idxs].mean(axis=0) for idxs in dz_rows.values()]
    ).astype(np.float32)
    dz_index = {dz: i for i, dz in enumerate(dz_list)}
    return dz_list, X_agg, dz_index


def build_or_load_captions(
    samples: list[dict],
    cache_path: Path,
    cfg: RouteCConfig,
) -> tuple[list[str], np.ndarray]:
    meta = {
        "backend": cfg.encoder_backend,
        "model_name": cfg.encoder_name,
        "caption_mode": cfg.caption_mode,
        "max_length": cfg.encoder_max_length,
    }
    if cache_path.exists():
        try:
            dzs_cached, X_cached, meta_cached = load_caption_cache(cache_path)
        except Exception as exc:
            print(f"[cache] could not read {cache_path}: {exc}; re-encoding")
        else:
            sample_dzs = {s["datazone"] for s in samples}
            if set(dzs_cached) == sample_dzs and meta_cached == meta:
                print(f"[cache] loaded caption embeddings from {cache_path}")
                return dzs_cached, X_cached
            print(f"[cache] stale ({len(dzs_cached)}→{len(sample_dzs)} samples "
                  f"or meta mismatch); re-encoding")

    if cfg.caption_mode == "modality_sep":
        segments = build_captions_modality_sep(samples)
        print(f"[caption] modality_sep: {len(segments)} samples, "
              f"sat/ntl/sv/poi encoded separately")
        print(f"[encode] backend={cfg.encoder_backend}  model={cfg.encoder_name}")
        X = encode_modality_sep(
            segments,
            backend=cfg.encoder_backend,
            model_name=cfg.encoder_name,
            batch_size=cfg.encoder_batch_size,
            max_length=cfg.encoder_max_length,
        )
    else:
        captions = build_captions(samples, mode=cfg.caption_mode)
        print(f"[caption] built {len(captions)} captions (mode={cfg.caption_mode}); "
              f"sample[0]={captions[0][:160]!r}")
        print(f"[encode] backend={cfg.encoder_backend}  model={cfg.encoder_name}")
        X = encode_texts(
            captions,
            backend=cfg.encoder_backend,
            model_name=cfg.encoder_name,
            batch_size=cfg.encoder_batch_size,
            max_length=cfg.encoder_max_length,
        )

    dzs = [s["datazone"] for s in samples]
    save_caption_cache(cache_path, dzs, X, meta)
    print(f"[encode] X shape={X.shape}  cached → {cache_path}")
    return dzs, X


def run_cv(
    dataset_path: str | Path,
    out_dir: str | Path,
    cfg: RouteCConfig,
    n_splits: int = 5,
    shp_path: str | Path = DEFAULT_SHP,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    print(f"[data] loaded {len(samples)} samples from {dataset_path}")

    cache_path = out_dir / "caption_cache.pt"
    cache_dzs, X = build_or_load_captions(samples, cache_path, cfg)

    if cfg.use_dz_agg:
        cache_dzs, X, dz_index = _aggregate_X_by_dz(samples, X)
        print(f"[dz_agg] aggregated to {len(cache_dzs)} unique datazones "
              f"(X shape={X.shape})")
    else:
        dz_index = {dz: i for i, dz in enumerate(cache_dzs)}

    neighbors = None
    if cfg.use_spatial_lag:
        from decision.data.spatial_neighbors import build_neighbors
        neighbors = build_neighbors(shp_path)
        n_with_neighbors = sum(1 for v in neighbors.values() if v)
        print(f"[spatial] neighbour map: {len(neighbors)} datazones, "
              f"{n_with_neighbors} have ≥1 neighbour")

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

        fold_dir = out_dir / f"fold{fold_idx}"
        result = train_fold_caption(
            train_samples, val_samples, X, dz_index,
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
        "d_caption": int(X.shape[1]),
        "config": asdict(cfg),
        "folds": fold_summaries,
        "oof_mean_spearman": float(np.mean([f["val_mean_spearman"] for f in fold_summaries])),
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[done] wrote {len(oof_rows)} OOF rows → {oof_path}")
    print(f"[done] OOF mean Spearman = {summary['oof_mean_spearman']:.4f}")
    return summary


def _load_yaml(path: str | Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML config (CLI flags override)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--caption-mode", default=None,
                        choices=["concat", "templated", "modality_sep"])
    parser.add_argument("--encoder-backend", default=None, choices=["bert", "sbert"])
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--encoder-max-length", type=int, default=None)
    parser.add_argument("--regressor", default=None,
                        choices=["ridge_cv", "lasso_cv", "elasticnet_cv"])
    parser.add_argument("--use-poi-vec", action="store_true", default=None)
    parser.add_argument("--no-poi-vec", dest="use_poi_vec", action="store_false")
    parser.add_argument("--use-spatial-lag", action="store_true", default=None)
    parser.add_argument("--no-spatial-lag", dest="use_spatial_lag", action="store_false")
    parser.add_argument("--use-ego-gap", action="store_true", default=None)
    parser.add_argument("--no-ego-gap", dest="use_ego_gap", action="store_false")
    parser.add_argument("--use-sar-lag", action="store_true", default=None)
    parser.add_argument("--no-sar-lag", dest="use_sar_lag", action="store_false")
    parser.add_argument("--use-dz-agg", action="store_true", default=None)
    parser.add_argument("--no-dz-agg", dest="use_dz_agg", action="store_false")
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    y = _load_yaml(args.config) if args.config else {}
    cap_y = y.get("caption", {})
    enc_y = y.get("encoder", {})
    cv_y = y.get("cv", {})
    tr_y = y.get("train", {})

    def pick(cli, yaml_val, default):
        return cli if cli is not None else (yaml_val if yaml_val is not None else default)

    dataset = pick(args.dataset, y.get("dataset"), None)
    out_dir = pick(args.out_dir, y.get("out_dir"), None)
    if not dataset or not out_dir:
        parser.error("--dataset and --out-dir are required (via CLI or YAML)")

    cfg = RouteCConfig(
        caption_mode=pick(args.caption_mode, cap_y.get("mode"), "concat"),
        encoder_backend=pick(args.encoder_backend, enc_y.get("backend"), "sbert"),
        encoder_name=pick(args.encoder_name, enc_y.get("name"),
                          "sentence-transformers/all-MiniLM-L6-v2"),
        encoder_batch_size=pick(args.encoder_batch_size, enc_y.get("batch_size"), 32),
        encoder_max_length=pick(args.encoder_max_length, enc_y.get("max_length"), 256),
        regressor=pick(args.regressor, tr_y.get("regressor"), "ridge_cv"),
        use_poi_vec=pick(args.use_poi_vec, tr_y.get("use_poi_vec"), False),
        use_spatial_lag=pick(args.use_spatial_lag, tr_y.get("use_spatial_lag"), False),
        use_ego_gap=pick(args.use_ego_gap, tr_y.get("use_ego_gap"), False),
        use_sar_lag=pick(args.use_sar_lag, tr_y.get("use_sar_lag"), False),
        use_dz_agg=pick(args.use_dz_agg, tr_y.get("use_dz_agg"), False),
        seed=pick(args.seed, tr_y.get("seed"), 42),
    )
    n_splits = pick(args.n_splits, cv_y.get("n_splits"), 5)

    run_cv(dataset_path=dataset, out_dir=out_dir, cfg=cfg, n_splits=n_splits)


if __name__ == "__main__":
    _main()
