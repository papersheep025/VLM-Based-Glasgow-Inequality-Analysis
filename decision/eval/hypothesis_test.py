"""
Hypothesis testing: pairwise comparison of Route A, A', C on OOF predictions.

For each domain and for each pair of routes, we test:
  H0: The two routes have equal prediction error (median difference = 0)
using the Wilcoxon signed-rank test on paired per-datazone absolute errors.

Additionally, for Spearman ρ difference we compute 95% bootstrap CIs.

Multiple-comparison correction: Bonferroni across 3 pairs × 8 domains (7 + overall).

Usage:
    python -m decision.eval.hypothesis_test \
        --out-dir outputs/decision/compare/hypothesis_v0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DOMAINS = ["Income", "Employment", "Health", "Education", "Access", "Crime", "Housing"]

ROUTES: dict[str, str] = {
    "A_MLP":   "outputs/decision/route_a/bge_m3_v0/oof_predictions.jsonl",
    "A_LGBM":  "outputs/decision/route_a_lgbm/bge_m3_v0/oof_predictions.jsonl",
    "C_Ridge": "outputs/decision/route_c/sbert_minilm_v0/oof_predictions.jsonl",
}

N_BOOT = 2000
BOOT_SEED = 42
ALPHA = 0.05
N_PAIRS = 3  # (A_MLP, A_LGBM), (A_MLP, C_Ridge), (A_LGBM, C_Ridge)
N_DOMAINS = len(DOMAINS) + 1  # +1 for overall


def _load_oof(path: str | Path) -> dict[str, dict[str, float]]:
    """Returns {datazone: {domain: pred_value, ..., 'target': {domain: gt}}}"""
    by_dz: dict[str, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_dz[r["datazone"]].append(r)

    result: dict[str, dict] = {}
    for dz, rows in by_dz.items():
        pred = {d: float(np.mean([ro["prediction_json"][d] for ro in rows])) for d in DOMAINS}
        tgt = {d: float(np.mean([ro["target_raw"][d] for ro in rows])) for d in DOMAINS}
        pred["overall"] = float(np.mean(list(pred.values())))
        tgt["overall"] = float(np.mean(list(tgt.values())))
        result[dz] = {"pred": pred, "tgt": tgt}
    return result


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr as sp
    return float(sp(x, y).statistic)


def _boot_spearman_diff(a_err: np.ndarray, b_err: np.ndarray,
                        tgt: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap CI for (spearman_b - spearman_a) where higher spearman = better route."""
    rng = np.random.default_rng(seed)
    diffs = []
    N = len(tgt)
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        rho_a = _spearman(a_err[idx], tgt[idx])
        rho_b = _spearman(b_err[idx], tgt[idx])
        diffs.append(rho_b - rho_a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    observed = _spearman(b_err, tgt) - _spearman(a_err, tgt)
    return float(observed), float(lo), float(hi)


def _wilcoxon_pvalue(a_ae: np.ndarray, b_ae: np.ndarray) -> float:
    """Wilcoxon signed-rank on absolute errors: H0 = no difference."""
    diff = a_ae - b_ae
    if np.all(diff == 0):
        return 1.0
    try:
        return float(wilcoxon(diff, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def run(out_dir: Path, routes: dict[str, str] = ROUTES) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] loading OOF predictions …")
    data: dict[str, dict] = {name: _load_oof(path) for name, path in routes.items()}

    # Intersect datazones present in all routes
    shared_dz = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
    print(f"[data] shared datazones: {len(shared_dz)}")

    all_domains = DOMAINS + ["overall"]
    route_names = list(routes.keys())
    pairs = [(route_names[i], route_names[j])
             for i in range(len(route_names)) for j in range(i + 1, len(route_names))]

    n_tests = len(pairs) * len(all_domains)
    bonferroni_alpha = ALPHA / n_tests
    print(f"[test] {len(pairs)} pairs × {len(all_domains)} domains = {n_tests} tests")
    print(f"[test] Bonferroni corrected α = {bonferroni_alpha:.5f}")

    rows = []
    for dom in all_domains:
        tgt = np.array([data[route_names[0]][dz]["tgt"][dom] for dz in shared_dz])
        preds = {
            name: np.array([data[name][dz]["pred"][dom] for dz in shared_dz])
            for name in route_names
        }
        aes = {name: np.abs(preds[name] - tgt) for name in route_names}
        rhos = {name: _spearman(preds[name], tgt) for name in route_names}
        rmses = {name: float(np.sqrt(np.mean((preds[name] - tgt) ** 2))) for name in route_names}

        for (r_a, r_b) in pairs:
            p_raw = _wilcoxon_pvalue(aes[r_a], aes[r_b])
            p_corr = min(p_raw * n_tests, 1.0)  # Bonferroni
            obs_diff, ci_lo, ci_hi = _boot_spearman_diff(
                preds[r_a], preds[r_b], tgt, N_BOOT, BOOT_SEED
            )
            rows.append({
                "domain": dom,
                "route_a": r_a,
                "route_b": r_b,
                "spearman_a": round(rhos[r_a], 4),
                "spearman_b": round(rhos[r_b], 4),
                "delta_spearman_b_minus_a": round(obs_diff, 4),
                "boot_ci_lo": round(ci_lo, 4),
                "boot_ci_hi": round(ci_hi, 4),
                "rmse_a": round(rmses[r_a], 4),
                "rmse_b": round(rmses[r_b], 4),
                "wilcoxon_p_raw": round(p_raw, 6),
                "wilcoxon_p_bonferroni": round(p_corr, 6),
                "significant": p_corr < ALPHA,
            })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "pairwise_hypothesis_test.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {len(df)} rows → {out_csv}")

    _print_summary(df, route_names, bonferroni_alpha)
    _save_spearman_table(data, shared_dz, all_domains, route_names, out_dir)


def _print_summary(df: pd.DataFrame, route_names: list[str], alpha: float) -> None:
    print("\n" + "=" * 78)
    print("PAIRWISE HYPOTHESIS TEST SUMMARY  (Wilcoxon signed-rank on |pred - gt|)")
    print(f"Bonferroni corrected α = {alpha:.5f}")
    print("=" * 78)

    for (r_a, r_b), grp in df.groupby(["route_a", "route_b"]):
        print(f"\n{r_a}  vs  {r_b}")
        print(f"{'Domain':<12} {'ρ_A':>7} {'ρ_B':>7} {'Δρ(B−A)':>10} {'95% CI':>18} "
              f"{'p_raw':>10} {'p_Bonf':>10} {'sig':>5}")
        print("-" * 88)
        for _, row in grp.iterrows():
            ci = f"[{row['boot_ci_lo']:+.3f},{row['boot_ci_hi']:+.3f}]"
            sig = "***" if row["wilcoxon_p_bonferroni"] < 0.001 else \
                  "**"  if row["wilcoxon_p_bonferroni"] < 0.01  else \
                  "*"   if row["wilcoxon_p_bonferroni"] < 0.05  else ""
            print(f"{row['domain']:<12} {row['spearman_a']:>7.4f} {row['spearman_b']:>7.4f} "
                  f"{row['delta_spearman_b_minus_a']:>+10.4f} {ci:>18} "
                  f"{row['wilcoxon_p_raw']:>10.4f} {row['wilcoxon_p_bonferroni']:>10.4f} {sig:>5}")

    print("\n" + "=" * 78)
    print("ROUTE RANKING BY OVERALL OOF SPEARMAN ρ")
    print("=" * 78)


def _save_spearman_table(
    data: dict[str, dict],
    shared_dz: list[str],
    all_domains: list[str],
    route_names: list[str],
    out_dir: Path,
) -> None:
    rows = []
    for dom in all_domains:
        tgt = np.array([data[route_names[0]][dz]["tgt"][dom] for dz in shared_dz])
        row = {"domain": dom}
        for name in route_names:
            preds = np.array([data[name][dz]["pred"][dom] for dz in shared_dz])
            row[f"spearman_{name}"] = round(_spearman(preds, tgt), 4)
            row[f"rmse_{name}"] = round(float(np.sqrt(np.mean((preds - tgt) ** 2))), 4)
            ss_res = float(np.sum((tgt - preds) ** 2))
            ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
            row[f"r2_{name}"] = round(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = out_dir / "route_comparison_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote comparison table → {out_csv}")
    print("\n" + df.to_string(index=False))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/decision/compare/hypothesis_v0")
    parser.add_argument("--route-a-mlp",
                        default="outputs/decision/route_a/bge_m3_v0/oof_predictions.jsonl")
    parser.add_argument("--route-a-lgbm",
                        default="outputs/decision/route_a_lgbm/bge_m3_v0/oof_predictions.jsonl")
    parser.add_argument("--route-c-ridge",
                        default="outputs/decision/route_c/sbert_minilm_v0/oof_predictions.jsonl")
    args = parser.parse_args()

    routes = {
        "A_MLP":   args.route_a_mlp,
        "A_LGBM":  args.route_a_lgbm,
        "C_Ridge": args.route_c_ridge,
    }
    run(Path(args.out_dir), routes=routes)


if __name__ == "__main__":
    _main()
