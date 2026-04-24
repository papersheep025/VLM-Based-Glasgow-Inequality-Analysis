# -*- coding: utf-8 -*-
"""
Compare datazone-level VLM predictions against SIMD 2020 ground truth.

Metrics per domain:
  RMSE, MAE, Accuracy (exact), Within-1 Accuracy, Pearson r, Spearman ρ, R², QWK
  Each metric reported with 95% bootstrap CI (e.g. RMSE_lo / RMSE_hi).

Additional outputs:
  - Per-quintile breakdown CSV
  - Mean baseline and spatial lag baseline comparison
  - Moran's I spatial autocorrelation on residuals (requires libpysal + esda)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DOMAIN_MAP_STANDARD = {
    "income": "income_score",
    "employment": "employment_score",
    "health": "health_score",
    "education": "education_score",
    "housing": "housing_score",
    "access": "access_score",
    "crime": "crime_score",
    "overall": "overall_score",
}

DOMAIN_MAP_PRECISE = {
    "income": "Income",
    "employment": "Employment",
    "health": "Health",
    "education": "Education",
    "housing": "Housing",
    "access": "Access",
    "crime": "Crime",
    "overall": "Overall",
}


def _detect_domain_map(simd_df: pd.DataFrame) -> dict:
    if "income_score" in simd_df.columns:
        return DOMAIN_MAP_STANDARD
    return DOMAIN_MAP_PRECISE


DOMAIN_MAP = DOMAIN_MAP_STANDARD

N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate domain-level VLM predictions vs SIMD ground truth.")
    parser.add_argument("--pred-csv", type=Path, required=True, help="Aggregated datazone predictions CSV")
    parser.add_argument(
        "--simd-csv",
        type=Path,
        default=Path("dataset/SIMD/SIMD_data.csv"),
        help="SIMD ground truth CSV",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=Path("dataset/glasgow_datazone/glasgow_datazone.shp"),
        help="Shapefile for spatial weights (Moran's I and spatial lag baseline)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/domain_evaluation_report.csv"),
        help="Output metrics CSV path",
    )
    parser.add_argument(
        "--rank-csv",
        type=Path,
        default=Path("dataset/SIMD/SIMD_data.csv"),
        help="CSV containing SIMD2020v2_Rank column (used for rank-band breakdown)",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Skip spatial analysis (Moran's I, spatial lag baseline)",
    )
    return parser.parse_args()


def _qwk(true_int: np.ndarray, pred_int: np.ndarray) -> float:
    labels = sorted(set(true_int.tolist()) | set(pred_int.tolist()))
    try:
        return float(cohen_kappa_score(true_int, pred_int, labels=labels, weights="quadratic"))
    except Exception:
        return float("nan")


def compute_metrics(pred: np.ndarray, true: np.ndarray, rng: np.random.Generator | None = None) -> dict:
    mask = ~(np.isnan(pred) | np.isnan(true))
    pred, true = pred[mask], true[mask]
    n = len(pred)

    _metric_keys = [
        "N",
        "RMSE", "RMSE_lo", "RMSE_hi",
        "MAE", "MAE_lo", "MAE_hi",
        "Acc", "Acc_lo", "Acc_hi",
        "Within1Acc", "Within1Acc_lo", "Within1Acc_hi",
        "Pearson_r", "Pearson_r_lo", "Pearson_r_hi",
        "Spearman_rho", "Spearman_rho_lo", "Spearman_rho_hi",
        "R2", "R2_lo", "R2_hi",
        "QWK", "QWK_lo", "QWK_hi",
    ]
    if n < 2:
        return {k: float("nan") for k in _metric_keys}

    true_int = np.clip(np.round(true).astype(int), 1, 10)
    pred_int = np.clip(np.round(pred).astype(int), 1, 10)

    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    acc = float(np.mean(pred_int == true_int))
    within1 = float(np.mean(np.abs(pred - true) <= 1.0))
    pearson_r, _ = stats.pearsonr(pred, true)
    spearman_rho, _ = stats.spearmanr(pred, true)
    r2 = float(r2_score(true, pred))
    qwk = _qwk(true_int, pred_int)

    if rng is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)

    boot_rmse, boot_mae, boot_acc, boot_w1 = [], [], [], []
    boot_pr, boot_sr, boot_r2, boot_qwk = [], [], [], []

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        p_b, t_b = pred[idx], true[idx]
        ti_b = np.clip(np.round(t_b).astype(int), 1, 10)
        pi_b = np.clip(np.round(p_b).astype(int), 1, 10)
        boot_rmse.append(np.sqrt(mean_squared_error(t_b, p_b)))
        boot_mae.append(mean_absolute_error(t_b, p_b))
        boot_acc.append(float(np.mean(pi_b == ti_b)))
        boot_w1.append(float(np.mean(np.abs(p_b - t_b) <= 1.0)))
        try:
            pr_b, _ = stats.pearsonr(p_b, t_b)
        except Exception:
            pr_b = float("nan")
        boot_pr.append(pr_b)
        sr_b, _ = stats.spearmanr(p_b, t_b)
        boot_sr.append(sr_b)
        try:
            boot_r2.append(r2_score(t_b, p_b))
        except Exception:
            boot_r2.append(float("nan"))
        boot_qwk.append(_qwk(ti_b, pi_b))

    def ci(arr):
        a = np.array(arr, dtype=float)
        return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))

    def r4(v):
        return round(float(v), 4)

    return {
        "N": n,
        "RMSE": r4(rmse), "RMSE_lo": r4(ci(boot_rmse)[0]), "RMSE_hi": r4(ci(boot_rmse)[1]),
        "MAE": r4(mae), "MAE_lo": r4(ci(boot_mae)[0]), "MAE_hi": r4(ci(boot_mae)[1]),
        "Acc": r4(acc), "Acc_lo": r4(ci(boot_acc)[0]), "Acc_hi": r4(ci(boot_acc)[1]),
        "Within1Acc": r4(within1), "Within1Acc_lo": r4(ci(boot_w1)[0]), "Within1Acc_hi": r4(ci(boot_w1)[1]),
        "Pearson_r": r4(pearson_r), "Pearson_r_lo": r4(ci(boot_pr)[0]), "Pearson_r_hi": r4(ci(boot_pr)[1]),
        "Spearman_rho": r4(spearman_rho), "Spearman_rho_lo": r4(ci(boot_sr)[0]), "Spearman_rho_hi": r4(ci(boot_sr)[1]),
        "R2": r4(r2), "R2_lo": r4(ci(boot_r2)[0]), "R2_hi": r4(ci(boot_r2)[1]),
        "QWK": r4(qwk), "QWK_lo": r4(ci(boot_qwk)[0]), "QWK_hi": r4(ci(boot_qwk)[1]),
    }


def compute_metrics_by_rank_band(merged: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    for pred_col, true_col in DOMAIN_MAP.items():
        for band in sorted(merged["rank_band"].dropna().unique()):
            sub = merged[merged["rank_band"] == band]
            m = compute_metrics(sub[pred_col].to_numpy(float), sub[true_col].to_numpy(float), rng)
            m["domain"] = pred_col
            m["rank_band"] = int(band)
            records.append(m)
    df = pd.DataFrame(records)
    cols = ["domain", "rank_band"] + [c for c in df.columns if c not in ("domain", "rank_band")]
    return df[cols]


def compute_mean_baseline(merged: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    for pred_col, true_col in DOMAIN_MAP.items():
        true_vals = merged[true_col].to_numpy(float)
        mask = ~np.isnan(true_vals)
        true_vals = true_vals[mask]
        if len(true_vals) < 2:
            continue
        baseline_pred = np.full_like(true_vals, true_vals.mean())
        m = compute_metrics(baseline_pred, true_vals, rng)
        m["domain"] = pred_col
        m["source"] = "baseline_mean"
        records.append(m)
    return records


def compute_spatial_lag_baseline(merged: pd.DataFrame, gdf) -> list[dict]:
    try:
        import libpysal.weights as lps_w
    except ImportError:
        print("[warn] libpysal not installed; skipping spatial lag baseline")
        return []

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gdf_merged = gdf.merge(merged[["datazone"] + list(DOMAIN_MAP.values())], on="datazone", how="inner")
    w = lps_w.Queen.from_dataframe(gdf_merged, silence_warnings=True)
    w.transform = "r"

    records = []
    for pred_col, true_col in DOMAIN_MAP.items():
        true_vals = gdf_merged[true_col].to_numpy(float)
        lag_pred = lps_w.lag_spatial(w, true_vals)
        m = compute_metrics(lag_pred, true_vals, rng)
        m["domain"] = pred_col
        m["source"] = "baseline_spatial_lag"
        records.append(m)
    return records


def compute_spatial_autocorrelation(merged: pd.DataFrame, gdf):
    try:
        import libpysal.weights as lps_w
        import esda
    except ImportError:
        print("[warn] libpysal/esda not installed; skipping spatial autocorrelation")
        return None, merged

    all_cols = ["datazone"] + list(DOMAIN_MAP.keys()) + list(DOMAIN_MAP.values())
    existing = [c for c in all_cols if c in merged.columns]
    gdf_merged = gdf.merge(merged[existing], on="datazone", how="inner")
    w = lps_w.Queen.from_dataframe(gdf_merged, silence_warnings=True)
    w.transform = "r"

    moran_rows = []
    for pred_col, true_col in DOMAIN_MAP.items():
        pred_vals = gdf_merged[pred_col].to_numpy(float) if pred_col in gdf_merged.columns else np.full(len(gdf_merged), np.nan)
        true_vals = gdf_merged[true_col].to_numpy(float)
        residuals = pred_vals - true_vals
        residuals = np.where(np.isnan(residuals), 0.0, residuals)

        mi = esda.Moran(residuals, w, permutations=999)
        moran_rows.append({
            "domain": pred_col,
            "Moran_I": round(float(mi.I), 4),
            "Moran_EI": round(float(mi.EI), 4),
            "Moran_p_sim": round(float(mi.p_sim), 4),
            "Moran_z_sim": round(float(mi.z_sim), 4),
        })

        lisa = esda.Moran_Local(residuals, w, permutations=999, seed=42)
        gdf_merged[f"{pred_col}_lisa_q"] = lisa.q
        gdf_merged[f"{pred_col}_lisa_p"] = lisa.p_sim

    moran_df = pd.DataFrame(moran_rows)
    return moran_df, gdf_merged


def main():
    args = parse_args()

    pred_df = pd.read_csv(args.pred_csv)
    simd_df = pd.read_csv(args.simd_csv)

    domain_map = _detect_domain_map(simd_df)

    if "SIMD2020v2_Rank" not in simd_df.columns:
        rank_df = pd.read_csv(args.rank_csv, usecols=["datazone", "SIMD2020v2_Rank"])
        simd_df = simd_df.merge(rank_df, on="datazone", how="left")
    simd_df["rank_band"] = pd.qcut(
        simd_df["SIMD2020v2_Rank"], q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    merged = pred_df.merge(
        simd_df[["datazone"] + list(domain_map.values()) + ["SIMD2020v2_Rank", "rank_band"]],
        on="datazone", how="inner",
    )
    print(f"Merged {len(merged)} datazones (pred={len(pred_df)}, simd={len(simd_df)})")

    global DOMAIN_MAP
    DOMAIN_MAP = domain_map

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    for pred_col, true_col in DOMAIN_MAP.items():
        m = compute_metrics(merged[pred_col].to_numpy(float), merged[true_col].to_numpy(float), rng)
        m["domain"] = pred_col
        m["source"] = "vlm"
        records.append(m)

    records.extend(compute_mean_baseline(merged))

    gdf = None
    if not args.no_spatial and args.shapefile.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(args.shapefile)
            id_col = next(
                (c for c in gdf.columns if "zone" in c.lower() or "dz" in c.lower()
                 or c.lower() in ("code", "datazone", "data_zone")),
                gdf.columns[0],
            )
            gdf = gdf.rename(columns={id_col: "datazone"})
            records.extend(compute_spatial_lag_baseline(merged, gdf))
        except ImportError:
            print("[warn] geopandas not installed; skipping spatial components")

    col_order = ["domain", "source", "N",
                 "RMSE", "RMSE_lo", "RMSE_hi",
                 "MAE", "MAE_lo", "MAE_hi",
                 "Acc", "Acc_lo", "Acc_hi",
                 "Within1Acc", "Within1Acc_lo", "Within1Acc_hi",
                 "Pearson_r", "Pearson_r_lo", "Pearson_r_hi",
                 "Spearman_rho", "Spearman_rho_lo", "Spearman_rho_hi",
                 "R2", "R2_lo", "R2_hi",
                 "QWK", "QWK_lo", "QWK_hi"]
    report = pd.DataFrame(records)[col_order]

    vlm_report = report[report["source"] == "vlm"].set_index("domain")
    print("\n=== Domain Evaluation Report (VLM, 95% CI) ===")
    disp_cols = ["N", "RMSE", "RMSE_lo", "RMSE_hi", "MAE", "Pearson_r", "Spearman_rho", "R2", "QWK"]
    print(vlm_report[disp_cols].to_string())

    vlm_avg = vlm_report[["RMSE", "MAE", "Acc", "Within1Acc", "Pearson_r", "Spearman_rho", "R2", "QWK"]].mean()
    print("\n=== Macro Average (VLM) ===")
    for k, v in vlm_avg.items():
        print(f"  {k}: {v:.4f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\nSaved main report → {args.output_csv}")

    rank_band_report = compute_metrics_by_rank_band(merged)
    stem = args.output_csv.stem
    base = args.output_csv.parent
    rank_band_path = base / (stem.replace("report", "by_rank_band") if "report" in stem else stem + "_by_rank_band")
    rank_band_path = rank_band_path.with_suffix(".csv")
    rank_band_report.to_csv(rank_band_path, index=False, encoding="utf-8")
    print(f"Saved per-rank-band report → {rank_band_path}")

    merged_out = base / (stem + "_merged.csv")
    if gdf is not None and not args.no_spatial:
        try:
            moran_df, gdf_lisa = compute_spatial_autocorrelation(merged, gdf)
            if moran_df is not None:
                print("\n=== Spatial Autocorrelation (Moran's I on residuals) ===")
                print(moran_df.to_string(index=False))
                moran_path = base / "domain_moran_results.csv"
                moran_df.to_csv(moran_path, index=False, encoding="utf-8")
                print(f"Saved Moran's I results → {moran_path}")

                lisa_cols = [c for c in gdf_lisa.columns if "_lisa_" in c]
                merged_lisa = merged.merge(gdf_lisa[["datazone"] + lisa_cols], on="datazone", how="left")
                merged_lisa.to_csv(merged_out, index=False, encoding="utf-8")
                print(f"Saved merged dataframe with LISA → {merged_out}")
                return
        except Exception as e:
            print(f"[warn] Spatial autocorrelation failed: {e}")

    merged.to_csv(merged_out, index=False, encoding="utf-8")
    print(f"Saved merged dataframe → {merged_out}")


if __name__ == "__main__":
    main()
