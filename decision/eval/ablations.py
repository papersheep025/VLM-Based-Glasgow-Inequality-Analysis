"""
Segment ablations for Route A: zero one input at a time, re-predict val set
for each fold, report per-domain Spearman drop vs the baseline OOF.

Segments ablated (one at a time):
    - sat       : e_sat      → zeros
    - ntl       : e_ntl      → zeros
    - sv        : e_sv       → zeros
    - poi_text  : e_poi_text → zeros
    - v_poi     : v_poi      → zeros

CLI:
    python -m decision.eval.ablations \\
        --dataset outputs/decision/dataset_v0.jsonl \\
        --run-dir outputs/decision/route_a/bge_m3_v0 \\
        --out-csv outputs/decision/route_a/bge_m3_v0/ablation_report.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from decision.data.poi_features import PoiFitter
from decision.data.targets import DOMAINS, denormalise_array
from decision.models.route_a.heads import RouteAModel
from decision.models.route_a.encoder import load_segment_cache
from decision.train.cv_runner import load_dataset
from decision.train.dataset import build_tensors
from src.glasgow_vlm.metrics import spearmanr
from src.glasgow_vlm.splits import group_kfold_by_iz


SEGMENTS = ["sat", "ntl", "sv", "poi_text", "v_poi"]


def _restore_poi_fitter(state: dict) -> PoiFitter:
    f = PoiFitter(top_k=state["top_k"])
    f.vocab = list(state["vocab"])
    f._mean = np.array(state["mean"], dtype=np.float32)
    f._std = np.array(state["std"], dtype=np.float32)
    f._fitted = True
    return f


def _auto_device(pref: str | None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _predict_ablated(
    model: RouteAModel,
    loader: DataLoader,
    device: str,
    ablate: str | None,
) -> np.ndarray:
    """Run val with one segment zeroed (or no ablation if `ablate` is None)."""
    model.eval()
    outs = []
    for e_sat, e_ntl, e_sv, e_poi, v_poi, _ in loader:
        e_sat = e_sat.to(device)
        e_ntl = e_ntl.to(device)
        e_sv = e_sv.to(device)
        e_poi = e_poi.to(device)
        v_poi = v_poi.to(device)
        if ablate == "sat":      e_sat = torch.zeros_like(e_sat)
        elif ablate == "ntl":    e_ntl = torch.zeros_like(e_ntl)
        elif ablate == "sv":     e_sv = torch.zeros_like(e_sv)
        elif ablate == "poi_text": e_poi = torch.zeros_like(e_poi)
        elif ablate == "v_poi":  v_poi = torch.zeros_like(v_poi)
        pred = model(e_sat, e_ntl, e_sv, e_poi, v_poi)
        outs.append(pred.cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.empty((0, len(DOMAINS)))


def _mean_spearman(pred: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
    per = {d: spearmanr(target[:, k].tolist(), pred[:, k].tolist())
           for k, d in enumerate(DOMAINS)}
    return float(np.mean(list(per.values()))), per


def run_ablations(
    dataset_path: str | Path,
    run_dir: str | Path,
    out_csv: str | Path,
    batch_size: int = 64,
    device: str | None = None,
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "cv_summary.json").read_text())
    d_txt = summary["d_txt"]
    cfg = summary["config"]
    seed = cfg["seed"]
    n_splits = summary["n_folds"]   # must match training to reproduce the same fold boundaries

    samples = load_dataset(dataset_path)
    cache_dzs, segments = load_segment_cache(run_dir / "segments_cache.pt")
    dz_index = {dz: i for i, dz in enumerate(cache_dzs)}
    by_dz = {s["datazone"]: s for s in samples}
    dataset_dzs = [s["datazone"] for s in samples]

    dev = _auto_device(device)
    rows: list[dict] = []

    for fold_idx, (_, val_dz) in enumerate(group_kfold_by_iz(dataset_dzs, n_splits=n_splits, seed=seed)):
        ckpt_path = run_dir / f"fold{fold_idx}_best.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        poi_fitter = _restore_poi_fitter(state["poi_fitter"])

        val_samples = [by_dz[d] for d in val_dz if d in by_dz]
        val_ds, y_raw, _ = build_tensors(val_samples, segments, dz_index, poi_fitter)
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = RouteAModel(
            d_txt=d_txt,
            d_poi_in=poi_fitter.dim,
            d_poi=cfg["d_poi"],
            d_trunk=cfg["d_trunk"],
            head_hidden=cfg["head_hidden"],
            dropout=cfg["dropout"],
            domains=cfg["domains"],
        ).to(dev)
        model.load_state_dict(state["model"])

        y_np = y_raw.numpy()
        base_pred = denormalise_array(_predict_ablated(model, loader, dev, None))
        base_mean, base_per = _mean_spearman(base_pred, y_np)
        rows.append({"fold": fold_idx, "ablation": "none",
                     "mean_spearman": base_mean, **base_per})

        for ab in SEGMENTS:
            ab_pred = denormalise_array(_predict_ablated(model, loader, dev, ab))
            ab_mean, ab_per = _mean_spearman(ab_pred, y_np)
            rows.append({"fold": fold_idx, "ablation": ab,
                         "mean_spearman": ab_mean, **ab_per})
        print(f"[fold {fold_idx}] baseline mean ρ = {base_mean:.4f}")

    df = pd.DataFrame(rows)

    agg = df.groupby("ablation").mean(numeric_only=True).reset_index()
    base_row = agg[agg["ablation"] == "none"].iloc[0]
    drop_cols = ["mean_spearman"] + list(DOMAINS)
    for c in drop_cols:
        agg[f"Δ_{c}"] = agg[c] - base_row[c]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    per_fold_path = out_csv.with_name(out_csv.stem + "_per_fold.csv")
    df.to_csv(per_fold_path, index=False)

    print(f"[done] wrote {out_csv}  (per-fold → {per_fold_path})")
    print(agg[["ablation", "mean_spearman", "Δ_mean_spearman"]].to_string(index=False))
    return agg


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-dir", required=True, help="CV run dir with fold{k}_best.pt + cv_summary.json + segments_cache.pt")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    run_ablations(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    _main()
