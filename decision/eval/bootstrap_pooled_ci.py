"""Bootstrap confidence intervals for pooled OOF metrics.

The bootstrap unit is the datazone-aggregated OOF row. For each resample,
all 7 domains are stacked into one vector before computing pooled Spearman
rho and pooled R2, matching ``decision.eval.compare``.

Example:
    python -m decision.eval.bootstrap_pooled_ci \
      --manifest decision/experiments/manifests/paper_v1.yaml \
      --out-csv outputs/evaluation/paper_v1_pooled_bootstrap_ci.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

from decision.data.targets import DOMAINS
from decision.eval.oof import aggregate_by_datazone, load_oof, pooled_r2, pooled_spearman
from src.glasgow_vlm.metrics import spearmanr


def _oof_path(path: Path) -> Path:
    return path if path.name == "oof_predictions.jsonl" else path / "oof_predictions.jsonl"


def _runs_from_manifest(path: Path) -> tuple[list[tuple[str, Path]], Path | None]:
    cfg = yaml.safe_load(path.read_text()) or {}
    base_dir = Path(cfg.get("base_dir", "."))
    runs = []
    for row in cfg.get("runs", []):
        name = row["name"]
        run_path = Path(row["path"])
        if not run_path.is_absolute():
            run_path = base_dir / run_path
        runs.append((name, _oof_path(run_path)))
    out_csv = cfg.get("bootstrap_ci_csv")
    if out_csv:
        out_csv = Path(out_csv)
        if not out_csv.is_absolute():
            out_csv = base_dir / out_csv
    return runs, out_csv


def _run_name(path: Path) -> str:
    if path.name == "oof_predictions.jsonl":
        return path.parent.name
    return path.name


def _runs_from_paths(paths: list[Path]) -> list[tuple[str, Path]]:
    return [(_run_name(p), _oof_path(p)) for p in paths]


def _stacked_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([df[f"target_{d}"].to_numpy(dtype=float) for d in DOMAINS])
    y_pred = np.concatenate([df[f"pred_{d}"].to_numpy(dtype=float) for d in DOMAINS])
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _pooled_metrics(df: pd.DataFrame) -> tuple[float, float]:
    y_true, y_pred = _stacked_arrays(df)
    rho = float(spearmanr(y_true.tolist(), y_pred.tolist())) if len(y_true) >= 2 else float("nan")
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return rho, r2


def _bootstrap_metric_ci(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_resamples: int,
) -> tuple[np.ndarray, np.ndarray]:
    rho_values = np.empty(n_resamples, dtype=float)
    r2_values = np.empty(n_resamples, dtype=float)
    n = len(df)
    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        sample = df.iloc[sample_idx]
        rho_values[i], r2_values[i] = _pooled_metrics(sample)
    return rho_values, r2_values


def _paired_delta_ci(
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    rng: np.random.Generator,
    n_resamples: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = baseline_df.set_index("datazone")
    run = run_df.set_index("datazone")
    common = base.index.intersection(run.index)
    base = base.loc[common].reset_index()
    run = run.loc[common].reset_index()
    n = len(common)
    d_rho = np.empty(n_resamples, dtype=float)
    d_r2 = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        base_sample = base.iloc[sample_idx]
        run_sample = run.iloc[sample_idx]
        base_rho, base_r2 = _pooled_metrics(base_sample)
        run_rho, run_r2 = _pooled_metrics(run_sample)
        d_rho[i] = run_rho - base_rho
        d_r2[i] = run_r2 - base_r2
    return d_rho, d_r2


def _ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_runs(
    runs: list[tuple[str, Path]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    if not runs:
        raise ValueError("At least one run is required.")

    dataframes = []
    for name, path in runs:
        df = aggregate_by_datazone(load_oof(path))
        dataframes.append((name, path, df))

    baseline_name, _, baseline_df = dataframes[0]
    baseline_rho, _ = pooled_spearman(baseline_df)
    baseline_r2, _ = pooled_r2(baseline_df)

    rows = []
    for run_idx, (name, path, df) in enumerate(dataframes):
        point_rho, _ = pooled_spearman(df)
        point_r2, _ = pooled_r2(df)

        rng = np.random.default_rng(seed + run_idx)
        rho_values, r2_values = _bootstrap_metric_ci(df, rng, n_resamples)
        rho_lo, rho_hi = _ci(rho_values)
        r2_lo, r2_hi = _ci(r2_values)

        if run_idx == 0:
            d_rho = np.zeros(n_resamples, dtype=float)
            d_r2 = np.zeros(n_resamples, dtype=float)
        else:
            rng_delta = np.random.default_rng(seed + 10000 + run_idx)
            d_rho, d_r2 = _paired_delta_ci(baseline_df, df, rng_delta, n_resamples)
        d_rho_lo, d_rho_hi = _ci(d_rho)
        d_r2_lo, d_r2_hi = _ci(d_r2)

        rows.append({
            "run": name,
            "pooled_rho": point_rho,
            "rho_ci_lo": rho_lo,
            "rho_ci_hi": rho_hi,
            "pooled_r2": point_r2,
            "r2_ci_lo": r2_lo,
            "r2_ci_hi": r2_hi,
            "delta_rho_vs_baseline": point_rho - baseline_rho,
            "delta_rho_ci_lo": d_rho_lo,
            "delta_rho_ci_hi": d_rho_hi,
            "delta_r2_vs_baseline": point_r2 - baseline_r2,
            "delta_r2_ci_lo": d_r2_lo,
            "delta_r2_ci_hi": d_r2_hi,
            "n_datazones": int(len(df)),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "baseline": baseline_name,
            "oof_path": str(path),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.manifest:
        runs, manifest_out_csv = _runs_from_manifest(args.manifest)
        out_csv = args.out_csv or manifest_out_csv
    else:
        runs = _runs_from_paths(args.runs)
        out_csv = args.out_csv

    if not runs:
        parser.error("Provide --runs or a manifest with at least one run.")

    missing = [str(path) for _, path in runs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing OOF files:\n" + "\n".join(missing))

    df = bootstrap_runs(runs, n_resamples=args.n_resamples, seed=args.seed)
    print(df[[
        "run", "pooled_rho", "rho_ci_lo", "rho_ci_hi",
        "pooled_r2", "r2_ci_lo", "r2_ci_hi",
        "delta_rho_vs_baseline", "delta_rho_ci_lo", "delta_rho_ci_hi",
    ]].to_string(index=False))

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, float_format="%.6f")
        print(f"\n[done] wrote {out_csv}")


if __name__ == "__main__":
    main()

