"""
5-fold IZ-grouped CV runner for Route A.

Flow
----
1. Load ``dataset_v0.jsonl`` (produced by ``build_dataset.py``).
2. Encode 4 text segments once with the frozen encoder (BGE-m3 by default).
3. For each of 5 folds from ``group_kfold_by_iz`` (Intermediate Zone groups,
   stratified on SIMD Overall):
     - fit a per-fold POI scaler on train,
     - train Route A (Huber + Pearson + SoftSpearman, MTL uncertainty weighting),
     - collect val predictions.
4. Aggregate OOF predictions into ``oof_predictions.jsonl`` using the
   project's standard ``prediction_json`` schema so existing evaluators work.

CLI:
    python -m decision.train.cv_runner \\
        --dataset outputs/decision/dataset_v0.jsonl \\
        --out-dir outputs/decision/route_a/bge_m3_v0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from decision.data.targets import DOMAINS
from decision.models.route_a.encoder import (
    FrozenTextEncoder,
    encode_segments,
    load_segment_cache,
    save_segment_cache,
)
from decision.train.route_a_train import RouteATrainConfig, train_fold
from src.glasgow_vlm.splits import group_kfold_by_iz


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_or_load_encodings(
    samples: list[dict],
    cache_path: Path,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    if cache_path.exists():
        dzs, segs = load_segment_cache(cache_path)
        cache_dzs = set(dzs)
        sample_dzs = {s["datazone"] for s in samples}
        if cache_dzs == sample_dzs:
            print(f"[cache] loaded segment embeddings from {cache_path}")
            return dzs, segs
        print(f"[cache] mismatch ({len(cache_dzs)} vs {len(sample_dzs)}); re-encoding")

    print(f"[encode] {model_name}  on  {len(samples)} samples × 4 segments")
    enc = FrozenTextEncoder(model_name=model_name, device=device)
    segs = encode_segments(enc, samples, batch_size=batch_size)
    dzs = [s["datazone"] for s in samples]
    save_segment_cache(segs, dzs, cache_path)
    print(f"[encode] cached → {cache_path}")
    return dzs, segs


# ---------------------------------------------------------------------------
# CV orchestration
# ---------------------------------------------------------------------------

def run_cv(
    dataset_path: str | Path,
    out_dir: str | Path,
    model_name: str = "BAAI/bge-m3",
    n_splits: int = 5,
    encode_batch_size: int = 32,
    cfg: RouteATrainConfig | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path)
    print(f"[data] loaded {len(samples)} samples from {dataset_path}")

    cache_path = out_dir / "segments_cache.pt"
    cache_dzs, segments = build_or_load_encodings(
        samples, cache_path, model_name=model_name,
        device=cfg.device if cfg else None, batch_size=encode_batch_size,
    )
    dz_index = {dz: i for i, dz in enumerate(cache_dzs)}

    actual_d_txt = next(iter(segments.values())).shape[1]
    cfg = cfg or RouteATrainConfig(d_txt=actual_d_txt)
    # Always sync d_txt from actual encoder output (overrides YAML/CLI default).
    cfg.d_txt = actual_d_txt

    dataset_dzs = [s["datazone"] for s in samples]
    by_dz = {s["datazone"]: s for s in samples}

    oof_rows: list[dict] = []
    fold_summaries: list[dict] = []

    for fold_idx, (train_dz, val_dz) in enumerate(
        group_kfold_by_iz(dataset_dzs, n_splits=n_splits, seed=cfg.seed)
    ):
        train_samples = [by_dz[d] for d in train_dz if d in by_dz]
        val_samples = [by_dz[d] for d in val_dz if d in by_dz]
        print(
            f"[fold {fold_idx}] train={len(train_samples)}  val={len(val_samples)}"
        )

        result = train_fold(train_samples, val_samples, segments, dz_index, cfg=cfg)
        print(
            f"[fold {fold_idx}] best epoch={result['best_epoch']}  "
            f"val mean Spearman={result['val_mean_spearman']:.4f}"
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
            "best_epoch": result["best_epoch"],
            "val_mean_spearman": result["val_mean_spearman"],
            "n_train": len(train_samples),
            "n_val": len(val_samples),
        })

        torch.save(result["best_state"], out_dir / f"fold{fold_idx}_best.pt")

    # ---------- persist outputs ----------
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML config (CLI flags override)")
    parser.add_argument("--dataset", default=None, help="Path to dataset_v0.jsonl")
    parser.add_argument("--out-dir", default=None, help="Output directory for this experiment")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
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

    cfg = RouteATrainConfig(
        d_poi=tr_y.get("d_poi", 32),
        d_trunk=tr_y.get("d_trunk", 256),
        head_hidden=tr_y.get("head_hidden", 64),
        dropout=tr_y.get("dropout", 0.2),
        lr=pick(args.lr, tr_y.get("lr"), 2e-4),
        weight_decay=tr_y.get("weight_decay", 1e-4),
        epochs=pick(args.epochs, tr_y.get("epochs"), 60),
        batch_size=pick(args.batch_size, tr_y.get("batch_size"), 64),
        grad_clip=tr_y.get("grad_clip", 1.0),
        huber_beta=tr_y.get("huber_beta", 1.0),
        lam_pearson=tr_y.get("lam_pearson", 0.5),
        lam_spearman=tr_y.get("lam_spearman", 0.5),
        soft_beta=tr_y.get("soft_beta", 5.0),
        patience=pick(args.patience, tr_y.get("patience"), 10),
        seed=pick(args.seed, tr_y.get("seed"), 42),
        device=pick(args.device, tr_y.get("device"), None),
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
