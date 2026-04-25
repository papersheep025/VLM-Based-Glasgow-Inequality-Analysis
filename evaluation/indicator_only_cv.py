"""Indicators-only CV baseline for the decision layer.

This script reads a perception JSONL with ``domain_indicators`` populated,
aggregates valid indicator vectors to one row per datazone, and trains a
RidgeCV regressor using only those indicator scores.

Output is compatible with the existing decision OOF evaluators:

    python evaluation/indicator_only_cv.py \
        --perception outputs/perception/qwen3vl_8b_perception_v2.jsonl \
        --out-dir outputs/decision/route_c/indicators_only_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.base import clone

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision.data.parse_perception import extract_domain_indicators  # noqa: E402
from decision.data.targets import DOMAINS  # noqa: E402
from decision.models.route_c.regressors import build_regressor  # noqa: E402
from perception.prompts.perception import INDICATOR_KEYS  # noqa: E402
from src.glasgow_vlm.metrics import spearmanr  # noqa: E402
from src.glasgow_vlm.splits import group_kfold_by_iz  # noqa: E402


def load_simd(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dz = row["datazone"]
            out[dz] = np.array([float(row[d]) for d in DOMAINS], dtype=np.float32)
    return out


def load_indicator_matrix(
    path: Path,
    include_missing_flag: bool = False,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Return (datazones, X, stats) aggregated to one row per datazone."""
    valid: dict[str, list[np.ndarray]] = defaultdict(list)
    missing_count = 0
    total_count = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            row = json.loads(line)
            dz = row["datazone"]
            vec, missing = extract_domain_indicators(row)
            if missing:
                missing_count += 1
                if include_missing_flag:
                    valid[dz].append(
                        np.concatenate([vec, np.array([1.0], dtype=np.float32)])
                    )
                continue
            if include_missing_flag:
                vec = np.concatenate([vec, np.array([0.0], dtype=np.float32)])
            valid[dz].append(vec.astype(np.float32))

    datazones = sorted(valid)
    X = np.stack([np.mean(valid[dz], axis=0) for dz in datazones]).astype(np.float32)
    stats = {
        "n_jsonl_rows": total_count,
        "n_missing_indicator_rows": missing_count,
        "n_datazones_with_valid_indicators": len(datazones),
        "indicator_dim": int(X.shape[1]) if len(datazones) else 0,
    }
    return datazones, X, stats


def mean_spearman(pred: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
    per = {
        d: float(spearmanr(target[:, k].tolist(), pred[:, k].tolist()))
        for k, d in enumerate(DOMAINS)
    }
    return float(np.mean(list(per.values()))), per


def run_cv(
    perception_path: Path,
    simd_path: Path,
    out_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
    regressor: str = "ridge_cv",
    include_missing_flag: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    datazones, X_all, indicator_stats = load_indicator_matrix(
        perception_path,
        include_missing_flag=include_missing_flag,
    )
    simd = load_simd(simd_path)
    keep = [dz for dz in datazones if dz in simd]
    dz_index = {dz: datazones.index(dz) for dz in keep}

    print(f"[data] perception rows={indicator_stats['n_jsonl_rows']}")
    print(f"[data] valid indicator datazones={len(datazones)}; SIMD overlap={len(keep)}")
    print(f"[data] X shape={(len(keep), X_all.shape[1])}")

    y_all = {dz: simd[dz] for dz in keep}
    base_model = build_regressor(regressor, seed=seed)

    oof_rows: list[dict] = []
    fold_summaries: list[dict] = []
    all_pred = []
    all_true = []

    for fold_idx, (train_dz, val_dz) in enumerate(
        group_kfold_by_iz(keep, n_splits=n_splits, seed=seed)
    ):
        train_dz = [dz for dz in train_dz if dz in dz_index]
        val_dz = [dz for dz in val_dz if dz in dz_index]

        X_tr = X_all[[dz_index[dz] for dz in train_dz]]
        X_va = X_all[[dz_index[dz] for dz in val_dz]]
        y_tr = np.stack([y_all[dz] for dz in train_dz]).astype(np.float32)
        y_va = np.stack([y_all[dz] for dz in val_dz]).astype(np.float32)

        pred = np.zeros_like(y_va, dtype=np.float32)
        for k in range(len(DOMAINS)):
            model = clone(base_model)
            model.fit(X_tr, y_tr[:, k])
            pred[:, k] = np.asarray(model.predict(X_va)).reshape(-1)

        fold_mean, fold_per = mean_spearman(pred, y_va)
        print(f"[fold {fold_idx}] train={len(train_dz)} val={len(val_dz)} mean ρ={fold_mean:.4f}")

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
            "n_train": len(train_dz),
            "n_val": len(val_dz),
            "val_mean_spearman": fold_mean,
            "val_per_domain_spearman": fold_per,
        })
        all_pred.append(pred)
        all_true.append(y_va)

    pred_all = np.concatenate(all_pred, axis=0)
    true_all = np.concatenate(all_true, axis=0)
    pooled_mean, pooled_per = mean_spearman(pred_all, true_all)

    oof_path = out_dir / "oof_predictions.jsonl"
    with open(oof_path, "w") as f:
        for row in oof_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "perception": str(perception_path),
        "simd": str(simd_path),
        "n_samples": len(keep),
        "n_folds": n_splits,
        "seed": seed,
        "regressor": regressor,
        "feature_block": "domain_indicators_only",
        "indicator_keys": list(INDICATOR_KEYS),
        "include_missing_flag": include_missing_flag,
        "indicator_stats": indicator_stats,
        "folds": fold_summaries,
        "oof_mean_spearman": float(np.mean([f["val_mean_spearman"] for f in fold_summaries])),
        "pooled_mean_spearman": pooled_mean,
        "pooled_per_domain_spearman": pooled_per,
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "feature_slices.json").write_text(json.dumps({
        "domain_indicators": [0, len(INDICATOR_KEYS)],
        **({"missing_flag": [len(INDICATOR_KEYS), len(INDICATOR_KEYS) + 1]}
           if include_missing_flag else {}),
    }, indent=2))

    print(f"[done] wrote {len(oof_rows)} OOF rows -> {oof_path}")
    print(f"[done] pooled mean ρ={pooled_mean:.4f}")
    for d in DOMAINS:
        print(f"  {d:<10} ρ={pooled_per[d]:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", required=True, type=Path)
    parser.add_argument("--simd", default=ROOT / "dataset/SIMD/SIMD_score.csv", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regressor", default="ridge_cv",
                        choices=["ridge_cv", "lasso_cv", "elasticnet_cv"])
    parser.add_argument("--include-missing-flag", action="store_true")
    args = parser.parse_args()
    run_cv(
        perception_path=args.perception,
        simd_path=args.simd,
        out_dir=args.out_dir,
        n_splits=args.n_splits,
        seed=args.seed,
        regressor=args.regressor,
        include_missing_flag=args.include_missing_flag,
    )


if __name__ == "__main__":
    main()
