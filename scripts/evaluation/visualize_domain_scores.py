# -*- coding: utf-8 -*-
"""
Visualize domain-level VLM predictions vs SIMD ground truth.

Static figures (matplotlib/seaborn, publication-ready):
  fig1_domain_scatter       — 2×4 scatter grid, Spearman ρ + RMSE per domain
  fig2_domain_error_bar     — RMSE + MAE bar chart per domain
  fig3_error_by_quintile    — Violin plots of (pred - true) × SIMD quintile
  fig4_correlation_heatmap  — Spearman cross-correlation matrix (pred domain × true domain)
  fig4b_diagonal_bar        — Same-domain correlation vs SIMD inter-domain baseline

Interactive:
  fig5_glasgow_map.html  — Plotly choropleth with dropdown
                           (predicted / true / error / overestimation / underestimation)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Style ─────────────────────────────────────────────────────────────────────
FONT_FAMILY = "DejaVu Sans"
mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

DOMAIN_MAP = {
    "income": "income_score",
    "employment": "employment_score",
    "health": "health_score",
    "education": "education_score",
    "housing": "housing_score",
    "access": "access_score",
    "crime": "crime_score",
    "overall": "overall_score",
}
DOMAIN_LABELS = {
    "income": "Income",
    "employment": "Employment",
    "health": "Health",
    "education": "Education",
    "housing": "Housing",
    "access": "Access",
    "crime": "Crime",
    "overall": "Overall",
}
QUINTILE_PALETTE = sns.color_palette("RdYlGn", 5)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize domain predictions vs SIMD ground truth.")
    parser.add_argument("--pred-csv", type=Path, required=True)
    parser.add_argument("--simd-csv", type=Path, default=Path("dataset/SIMD/SIMD_data.csv"))
    parser.add_argument("--shapefile", type=Path, default=Path("dataset/glasgow_datazone/glasgow_datazone.shp"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF output (faster)")
    return parser.parse_args()


def load_merged(pred_csv: Path, simd_csv: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    simd = pd.read_csv(simd_csv)
    keep = ["datazone"] + list(DOMAIN_MAP.values()) + ["SIMD2020v2_Quintile", "SIMD2020v2_Rank"]
    return pred.merge(simd[keep], on="datazone", how="inner")


def save(fig: plt.Figure, path_stem: Path, no_pdf: bool):
    fig.savefig(str(path_stem) + ".png")
    if not no_pdf:
        fig.savefig(str(path_stem) + ".pdf")
    plt.close(fig)
    print(f"  Saved {path_stem}.png{'  (PDF skipped)' if no_pdf else f' + .pdf'}")


# ── Figure 1: Domain scatter matrix ───────────────────────────────────────────
def fig1_domain_scatter(df: pd.DataFrame, out: Path, no_pdf: bool):
    domains = list(DOMAIN_MAP.keys())
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
    fig.suptitle("Predicted vs True Deprivation Scores by Domain", fontsize=14, fontweight="bold", y=1.01)

    for ax, domain in zip(axes.flat, domains):
        pred_col = domain
        true_col = DOMAIN_MAP[domain]
        mask = df[pred_col].notna() & df[true_col].notna()
        x = df.loc[mask, true_col].to_numpy(float)
        y = df.loc[mask, pred_col].to_numpy(float)
        quintiles = df.loc[mask, "SIMD2020v2_Quintile"].to_numpy(int)

        for q in range(1, 6):
            idx = quintiles == q
            ax.scatter(x[idx], y[idx], c=[QUINTILE_PALETTE[q - 1]], s=14, alpha=0.75, linewidths=0, label=f"Q{q}")

        lo, hi = min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        rho, _ = stats.spearmanr(x, y)
        rmse = np.sqrt(mean_squared_error(x, y))
        ax.set_title(DOMAIN_LABELS[domain], fontsize=11, fontweight="bold")
        ax.text(0.04, 0.93, f"ρ={rho:.2f}  RMSE={rmse:.2f}", transform=ax.transAxes,
                fontsize=8, va="top", color="#333333",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.8))
        ax.set_xlabel("True score", fontsize=9)
        ax.set_ylabel("Predicted score", fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    handles = [mpl.patches.Patch(color=QUINTILE_PALETTE[q], label=f"Quintile {q+1}") for q in range(5)]
    fig.legend(handles=handles, title="SIMD Quintile\n(1=most deprived)", loc="lower center",
               ncol=5, frameon=True, fontsize=8, title_fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, out / "fig1_domain_scatter", no_pdf)


# ── Figure 2: RMSE + MAE bar chart ────────────────────────────────────────────
def fig2_domain_error_bar(df: pd.DataFrame, out: Path, no_pdf: bool):
    domains = list(DOMAIN_MAP.keys())
    rmse_vals, mae_vals = [], []
    rmse_ci, mae_ci = [], []

    rng = np.random.default_rng(42)
    for domain in domains:
        pred = df[domain].to_numpy(float)
        true = df[DOMAIN_MAP[domain]].to_numpy(float)
        mask = ~(np.isnan(pred) | np.isnan(true))
        pred, true = pred[mask], true[mask]
        rmse_vals.append(np.sqrt(mean_squared_error(true, pred)))
        mae_vals.append(mean_absolute_error(true, pred))

        boots_r, boots_m = [], []
        for _ in range(500):
            idx = rng.integers(0, len(pred), len(pred))
            boots_r.append(np.sqrt(mean_squared_error(true[idx], pred[idx])))
            boots_m.append(mean_absolute_error(true[idx], pred[idx]))
        rmse_ci.append(1.96 * np.std(boots_r))
        mae_ci.append(1.96 * np.std(boots_m))

    order = np.argsort(rmse_vals)[::-1]
    domain_labels = [DOMAIN_LABELS[domains[i]] for i in order]
    rmse_sorted = [rmse_vals[i] for i in order]
    mae_sorted = [mae_vals[i] for i in order]
    rmse_ci_s = [rmse_ci[i] for i in order]
    mae_ci_s = [mae_ci[i] for i in order]

    x = np.arange(len(domains))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, rmse_sorted, width, yerr=rmse_ci_s, capsize=3,
                   color="#E07B54", label="RMSE", error_kw={"elinewidth": 0.8})
    bars2 = ax.bar(x + width / 2, mae_sorted, width, yerr=mae_ci_s, capsize=3,
                   color="#5B9BD5", label="MAE", error_kw={"elinewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=10)
    ax.set_ylabel("Error (score units, 1–10)", fontsize=10)
    ax.set_title("Per-Domain Prediction Error (RMSE & MAE)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save(fig, out / "fig2_domain_error_bar", no_pdf)


# ── Figure 3: Error × SIMD Quintile violin ────────────────────────────────────
def fig3_error_by_quintile(df: pd.DataFrame, out: Path, no_pdf: bool):
    domain_cols = [d for d in DOMAIN_MAP if d != "overall"]
    records = []
    for domain in domain_cols:
        pred = df[domain].to_numpy(float)
        true = df[DOMAIN_MAP[domain]].to_numpy(float)
        quintile = df["SIMD2020v2_Quintile"].to_numpy()
        mask = ~(np.isnan(pred) | np.isnan(true))
        for p, t, q in zip(pred[mask], true[mask], quintile[mask]):
            records.append({"domain": DOMAIN_LABELS[domain], "error": p - t, "quintile": int(q)})

    long_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(
        data=long_df, x="domain", y="error", hue="quintile",
        palette="RdYlGn", inner="quartile", linewidth=0.7,
        density_norm="width", cut=0, ax=ax, legend=True,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Domain", fontsize=10)
    ax.set_ylabel("Prediction Error (Pred − True)", fontsize=10)
    ax.set_title("Prediction Error Distribution by SIMD Quintile", fontsize=12, fontweight="bold")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles, [f"Q{l}" for l in labels_], title="SIMD Quintile\n(1=most deprived)",
              fontsize=8, title_fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, out / "fig3_error_by_quintile", no_pdf)


# ── Figure 4: Cross-domain Spearman heatmap ───────────────────────────────────
def fig4_correlation_heatmap(df: pd.DataFrame, out: Path, no_pdf: bool):
    domains = list(DOMAIN_MAP.keys())
    matrix = np.zeros((len(domains), len(domains)))
    for i, di in enumerate(domains):
        for j, dj in enumerate(domains):
            x = df[di].to_numpy(float)
            y = df[DOMAIN_MAP[dj]].to_numpy(float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 2:
                rho, _ = stats.spearmanr(x[mask], y[mask])
                matrix[i, j] = rho
            else:
                matrix[i, j] = np.nan

    labels = [DOMAIN_LABELS[d] for d in domains]
    fig, ax = plt.subplots(figsize=(9, 7))
    mask_upper = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap=cmap,
        vmin=-1, vmax=1, center=0, square=True,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, linecolor="#dddddd",
        cbar_kws={"shrink": 0.75, "label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_xlabel("True domain (SIMD)", fontsize=10)
    ax.set_ylabel("Predicted domain (VLM)", fontsize=10)
    ax.set_title("Cross-Domain Spearman Correlation\n(Predicted vs True)", fontsize=12, fontweight="bold")
    for i in range(len(domains)):
        ax.add_patch(mpl.patches.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", linewidth=1.5))
    fig.tight_layout()
    save(fig, out / "fig4_correlation_heatmap", no_pdf)


# ── Figure 4b: Diagonal correlation bar chart ─────────────────────────────────
def fig4b_diagonal_bar(df: pd.DataFrame, out: Path, no_pdf: bool):
    domains = list(DOMAIN_MAP.keys())
    labels = [DOMAIN_LABELS[d] for d in domains]

    same_domain_rho = []
    for d in domains:
        x = df[d].to_numpy(float)
        y = df[DOMAIN_MAP[d]].to_numpy(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 2:
            rho, _ = stats.spearmanr(x[mask], y[mask])
        else:
            rho = float("nan")
        same_domain_rho.append(rho)

    simd_true_cols = list(DOMAIN_MAP.values())
    simd_matrix = np.zeros((len(domains), len(domains)))
    for i, ci in enumerate(simd_true_cols):
        for j, cj in enumerate(simd_true_cols):
            x = df[ci].to_numpy(float)
            y = df[cj].to_numpy(float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 2 and i != j:
                rho, _ = stats.spearmanr(x[mask], y[mask])
                simd_matrix[i, j] = rho
            else:
                simd_matrix[i, j] = np.nan

    mean_interdom_rho = np.nanmean(simd_matrix + np.where(np.eye(len(domains), dtype=bool), np.nan, 0), axis=1)

    order = np.argsort(same_domain_rho)
    ordered_labels = [labels[i] for i in order]
    ordered_same = [same_domain_rho[i] for i in order]
    ordered_inter = [mean_interdom_rho[i] for i in order]

    x = np.arange(len(domains))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(x - width / 2, ordered_same, width, color="#5B9BD5", label="VLM Pred vs True (same domain)")
    ax.barh(x + width / 2, ordered_inter, width, color="#E07B54", alpha=0.7, label="SIMD inter-domain mean ρ")
    ax.set_yticks(x)
    ax.set_yticklabels(ordered_labels, fontsize=10)
    ax.set_xlabel("Spearman ρ", fontsize=10)
    ax.set_title("Same-Domain Prediction Correlation vs SIMD Inter-Domain Baseline", fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlim(-0.1, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, out / "fig4b_diagonal_correlation", no_pdf)


# ── Figure 5: Interactive Glasgow choropleth ──────────────────────────────────
def fig5_glasgow_map(df: pd.DataFrame, shapefile: Path, out: Path):
    gdf = gpd.read_file(shapefile).to_crs(epsg=4326)
    id_col = None
    for c in gdf.columns:
        if "zone" in c.lower() or "dz" in c.lower() or c.lower() in ("code", "datazone", "data_zone"):
            id_col = c
            break
    if id_col is None:
        id_col = gdf.columns[0]
        print(f"  [warn] shapefile ID column not found, using '{id_col}'")

    gdf = gdf.rename(columns={id_col: "datazone"})
    df = df.copy()
    df["error_overall"] = df["overall"] - df["overall_score"]
    df["overestimation"] = (df["error_overall"] > 1).astype(float)
    df["underestimation"] = (df["error_overall"] < -1).astype(float)

    merged_geo = gdf.merge(df, on="datazone", how="left")

    err_vals = merged_geo["error_overall"].dropna()
    err_abs_max = max(abs(err_vals.quantile(0.02)), abs(err_vals.quantile(0.98)))

    layers = {
        "Predicted Overall Score": {
            "col": "overall",
            "colorscale": "RdYlGn",
            "reversescale": False,
            "zmin": merged_geo["overall"].quantile(0.02),
            "zmax": merged_geo["overall"].quantile(0.98),
        },
        "True Overall Score (SIMD)": {
            "col": "overall_score",
            "colorscale": "RdYlGn",
            "reversescale": False,
            "zmin": merged_geo["overall_score"].quantile(0.02),
            "zmax": merged_geo["overall_score"].quantile(0.98),
        },
        "Prediction Error (Pred − True)": {
            "col": "error_overall",
            "colorscale": "RdBu",
            "reversescale": False,
            "zmin": -err_abs_max,
            "zmax": err_abs_max,
        },
        "Overestimation (error > +1)": {
            "col": "overestimation",
            "colorscale": [[0, "#dddddd"], [1, "#d62728"]],
            "reversescale": False,
            "zmin": 0,
            "zmax": 1,
        },
        "Underestimation (error < −1)": {
            "col": "underestimation",
            "colorscale": [[0, "#dddddd"], [1, "#1f77b4"]],
            "reversescale": False,
            "zmin": 0,
            "zmax": 1,
        },
    }

    custom_data_cols = ["datazone", "n_samples", "overall", "overall_score", "error_overall",
                        "income", "employment", "health", "education",
                        "housing", "access", "crime"]
    existing_cols = [c for c in custom_data_cols if c in merged_geo.columns]
    custom_data = merged_geo[existing_cols]

    hover_lines = ["<b>%{customdata[0]}</b><br>"]
    for i, col in enumerate(existing_cols[1:], start=1):
        hover_lines.append(f"{col}: %{{customdata[{i}]}}<br>")
    hover_template = "".join(hover_lines) + "<extra></extra>"

    geojson = merged_geo.__geo_interface__
    fig = go.Figure()
    for i, (label, cfg) in enumerate(layers.items()):
        col = cfg["col"]
        if col not in merged_geo.columns:
            continue
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=list(range(len(merged_geo))),
                z=merged_geo[col].fillna(-999).tolist(),
                colorscale=cfg["colorscale"],
                reversescale=cfg["reversescale"],
                zmin=cfg["zmin"],
                zmax=cfg["zmax"],
                marker_line_width=0.3,
                marker_line_color="white",
                colorbar_title=label,
                customdata=custom_data.values,
                hovertemplate=hover_template,
                visible=(i == 0),
                name=label,
            )
        )

    n_traces = len(layers)
    buttons = []
    for i, label in enumerate(layers.keys()):
        visible = [j == i for j in range(n_traces)]
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": visible}, {"title": f"Glasgow Deprivation — {label}"}]))

    fig.update_layout(
        title="Glasgow Deprivation — Predicted Overall Score",
        geo=dict(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)"),
        updatemenus=[dict(type="dropdown", x=0.01, y=0.99, xanchor="left", yanchor="top",
                          buttons=buttons, showactive=True)],
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=650,
        paper_bgcolor="#f8f8f8",
    )

    html_path = out / "fig5_glasgow_map.html"
    fig.write_html(str(html_path))
    print(f"  Saved {html_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_merged(args.pred_csv, args.simd_csv)
    print(f"  {len(df)} datazones loaded")

    print("\n[Fig 1] Domain scatter matrix...")
    fig1_domain_scatter(df, args.output_dir, args.no_pdf)

    print("[Fig 2] Domain error bar chart...")
    fig2_domain_error_bar(df, args.output_dir, args.no_pdf)

    print("[Fig 3] Error × quintile violins...")
    fig3_error_by_quintile(df, args.output_dir, args.no_pdf)

    print("[Fig 4] Spearman correlation heatmap...")
    fig4_correlation_heatmap(df, args.output_dir, args.no_pdf)

    print("[Fig 4b] Diagonal correlation bar chart...")
    fig4b_diagonal_bar(df, args.output_dir, args.no_pdf)

    print("[Fig 5] Interactive Glasgow map...")
    fig5_glasgow_map(df, args.shapefile, args.output_dir)

    print(f"\nAll outputs written to {args.output_dir}/")


if __name__ == "__main__":
    main()
