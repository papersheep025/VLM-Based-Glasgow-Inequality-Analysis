"""
5-fold IZ-grouped CV runner for Route A' (LightGBM).

Mirrors decision.train.cv_runner but swaps the MLP trainer for LightGBM.
Reuses the segment embedding cache (`segments_cache.pt`) — point `out_dir`
at an existing MLP run to skip BGE encoding.

CLI:
    python -m decision.train.cv_runner_lgbm --config decision/configs/route_a_lgbm.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from decision.data.targets import DOMAINS
from decision.train.cv_runner import build_or_load_encodings, load_dataset
from decision.train.route_a_lgbm_train import RouteALGBMConfig, train_fold_lgbm
from src.glasgow_vlm.splits import group_kfold_by_iz


def run_cv(
    dataset_path: str | Path,
    out_dir: str | Path,
    model_name: str = "BAAI/bge-m3",
    n_splits: int = 5,
    encode_batch_size: int = 32,
    cfg: RouteALGBMConfig | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    print(f"[data] loaded {len(samples)} samples from {dataset_path}")

    cache_path = out_dir / "segments_cache.pt"
    cache_dzs, segments = build_or_load_encodings(
        samples, cache_path, model_name=model_name,
        device=None, batch_size=encode_batch_size,
    )
    dz_index = {dz: i for i, dz in enumerate(cache_dzs)}

    cfg = cfg or RouteALGBMConfig()
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
        result = train_fold_lgbm(
            train_samples, val_samples, segments, dz_index, cfg=cfg, fold_dir=fold_dir,
        )
        print(
            f"[fold {fold_idx}] val mean Spearman={result['val_mean_spearman']:.4f}  "
            f"best_iters={result['best_iters']}"
        )

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
            "best_iters": result["best_iters"],
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
        "encoder": model_name,
        "d_txt": next(iter(segments.values())).shape[1],
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
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-leaves", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    y = _load_yaml(args.config) if args.config else {}
    enc_y = y.get("encoder", {})
    cv_y = y.get("cv", {})
    tr_y = y.get("train", {})

    def pick(cli, yaml_val, default):
        return cli if cli is not None else (yaml_val if yaml_val is not None else default)

    dataset = pick(args.dataset, y.get("dataset"), None)
    out_dir = pick(args.out_dir, y.get("out_dir"), None)
    if not dataset or not out_dir:
        parser.error("--dataset and --out-dir are required (via CLI or YAML)")

    encoder = pick(args.encoder, enc_y.get("name"), "BAAI/bge-m3")
    encode_bs = pick(args.encode_batch_size, enc_y.get("batch_size"), 32)
    n_splits = pick(args.n_splits, cv_y.get("n_splits"), 5)

    cfg = RouteALGBMConfig(
        pca_dim=pick(args.pca_dim, tr_y.get("pca_dim"), 32),
        n_estimators=pick(args.n_estimators, tr_y.get("n_estimators"), 2000),
        learning_rate=pick(args.learning_rate, tr_y.get("learning_rate"), 0.03),
        num_leaves=pick(args.num_leaves, tr_y.get("num_leaves"), 31),
        min_data_in_leaf=tr_y.get("min_data_in_leaf", 20),
        feature_fraction=tr_y.get("feature_fraction", 0.8),
        bagging_fraction=tr_y.get("bagging_fraction", 0.8),
        bagging_freq=tr_y.get("bagging_freq", 5),
        lambda_l2=tr_y.get("lambda_l2", 1.0),
        objective=tr_y.get("objective", "regression"),
        early_stopping_rounds=tr_y.get("early_stopping_rounds", 100),
        seed=pick(args.seed, tr_y.get("seed"), 42),
    )
    run_cv(
        dataset_path=dataset,
        out_dir=out_dir,
        model_name=encoder,
        n_splits=n_splits,
        encode_batch_size=encode_bs,
        cfg=cfg,
    )


if __name__ == "__main__":
    _main()
