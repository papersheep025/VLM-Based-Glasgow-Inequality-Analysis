"""Aggregate modality-ablation OOF predictions into a single comparison table.

For each run dir we compute, on DZ-aggregated predictions:
  - pooled Spearman rho (concat all 7 domains)
  - macro R^2 (mean across 7 domains)
  - per-domain Spearman rho

Baseline: outputs/decision/route_c/modality_sep_v1/oof_predictions.jsonl
Ablations: outputs/decision/route_c/ablations/{loo_no_*, single_*}/oof_predictions.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

DOMAINS = ["Income", "Employment", "Health", "Education", "Access", "Crime", "Housing"]
ROOT = Path(__file__).resolve().parents[2]


def load_oof(p: Path) -> tuple[np.ndarray, np.ndarray]:
    pred_per_dz: dict[str, list[list[float]]] = defaultdict(list)
    tgt_per_dz: dict[str, list[float]] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        dz = r["datazone"]
        pred_per_dz[dz].append([r["prediction_json"][d] for d in DOMAINS])
        tgt_per_dz[dz] = [r["target_raw"][d] for d in DOMAINS]
    dzs = sorted(pred_per_dz)
    pred = np.array([np.mean(pred_per_dz[d], axis=0) for d in dzs], dtype=np.float64)
    tgt = np.array([tgt_per_dz[d] for d in dzs], dtype=np.float64)
    return pred, tgt


def metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    pooled_rho = spearmanr(pred.ravel(), tgt.ravel()).statistic
    macro_r2 = float(np.mean([r2_score(tgt[:, k], pred[:, k]) for k in range(7)]))
    per_dom_rho = {d: float(spearmanr(pred[:, k], tgt[:, k]).statistic)
                   for k, d in enumerate(DOMAINS)}
    return {"pooled_rho": float(pooled_rho), "macro_r2": macro_r2, **per_dom_rho}


def main() -> None:
    runs = [
        ("baseline (sat+ntl+sv+poi)", ROOT / "outputs/decision/route_c/modality_sep_v1/oof_predictions.jsonl"),
        ("LOO -sat", ROOT / "outputs/decision/route_c/ablations/loo_no_sat/oof_predictions.jsonl"),
        ("LOO -ntl", ROOT / "outputs/decision/route_c/ablations/loo_no_ntl/oof_predictions.jsonl"),
        ("LOO -sv",  ROOT / "outputs/decision/route_c/ablations/loo_no_sv/oof_predictions.jsonl"),
        ("LOO -poi", ROOT / "outputs/decision/route_c/ablations/loo_no_poi/oof_predictions.jsonl"),
        ("single sat", ROOT / "outputs/decision/route_c/ablations/single_sat/oof_predictions.jsonl"),
        ("single ntl", ROOT / "outputs/decision/route_c/ablations/single_ntl/oof_predictions.jsonl"),
        ("single sv",  ROOT / "outputs/decision/route_c/ablations/single_sv/oof_predictions.jsonl"),
        ("single poi", ROOT / "outputs/decision/route_c/ablations/single_poi/oof_predictions.jsonl"),
    ]

    rows = []
    base_pooled = base_r2 = None
    for name, p in runs:
        if not p.exists():
            print(f"[skip] {name}: {p} missing")
            continue
        pred, tgt = load_oof(p)
        m = metrics(pred, tgt)
        if name.startswith("baseline"):
            base_pooled = m["pooled_rho"]
            base_r2 = m["macro_r2"]
        rows.append({"run": name, **m})

    df = pd.DataFrame(rows)
    df["d_pooled_rho"] = df["pooled_rho"] - base_pooled
    df["d_macro_r2"] = df["macro_r2"] - base_r2

    cols = ["run", "pooled_rho", "d_pooled_rho", "macro_r2", "d_macro_r2"] + DOMAINS
    df = df[cols]

    out_dir = ROOT / "outputs/evaluation/modality_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\n[saved] {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
