"""Shared helpers for decision-layer OOF prediction files.

All Route C-style experiments write ``oof_predictions.jsonl`` with one row per
held-out datazone. Keeping comparison logic here prevents each ablation script
from re-implementing slightly different Spearman calculations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from decision.data.targets import DOMAINS
from src.glasgow_vlm.metrics import spearmanr


def load_oof(path: str | Path) -> pd.DataFrame:
    """Load an OOF JSONL file into a flat dataframe."""
    path = Path(path)
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = obj["prediction_json"]
            target = obj["target_raw"]
            row = {
                "datazone": obj["datazone"],
                "fold": obj.get("fold", -1),
            }
            for d in DOMAINS:
                row[f"pred_{d}"] = float(pred[d])
                row[f"target_{d}"] = float(target[d])
            rows.append(row)
    if not rows:
        raise ValueError(f"No OOF rows found in {path}")
    return pd.DataFrame(rows)


def aggregate_by_datazone(df: pd.DataFrame) -> pd.DataFrame:
    """Average duplicate OOF rows for the same datazone, if any."""
    value_cols = [c for c in df.columns if c.startswith(("pred_", "target_"))]
    return df.groupby("datazone", as_index=False)[value_cols].mean(numeric_only=True)


def pooled_spearman(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Return stacked Spearman over all 7 domains and per-domain Spearman.

    The stacked metric concatenates (target, pred) across all domains and
    computes one Spearman correlation, matching the legacy
    ``evaluate_domain_scores.py`` ``overall`` row. Per-domain values are kept
    for diagnostics only.
    """
    per = {
        d: float(spearmanr(df[f"target_{d}"].tolist(), df[f"pred_{d}"].tolist()))
        for d in DOMAINS
    }
    y_true = np.concatenate([df[f"target_{d}"].to_numpy(dtype=float) for d in DOMAINS])
    y_pred = np.concatenate([df[f"pred_{d}"].to_numpy(dtype=float) for d in DOMAINS])
    pooled = float(spearmanr(y_true.tolist(), y_pred.tolist()))
    return pooled, per


def pooled_r2(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Return stacked R2 over all 7 domains and per-domain R2."""
    per = {}
    for d in DOMAINS:
        y_true = df[f"target_{d}"].to_numpy(dtype=float)
        y_pred = df[f"pred_{d}"].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        per[d] = float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() >= 2 else float("nan")
    y_true_all = np.concatenate([df[f"target_{d}"].to_numpy(dtype=float) for d in DOMAINS])
    y_pred_all = np.concatenate([df[f"pred_{d}"].to_numpy(dtype=float) for d in DOMAINS])
    mask = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
    pooled = float(r2_score(y_true_all[mask], y_pred_all[mask])) if mask.sum() >= 2 else float("nan")
    return pooled, per


def summarize_oof(path: str | Path) -> dict:
    """Summarize an OOF run using the legacy stacked + datazone-aggregated metric."""
    df = load_oof(path)
    df = aggregate_by_datazone(df)
    pooled, per = pooled_spearman(df)
    pooled_r2_value, per_r2 = pooled_r2(df)
    return {
        "n_rows": int(len(df)),
        "pooled_rho": pooled,
        "pooled_r2": pooled_r2_value,
        **{f"rho_{d}": per[d] for d in DOMAINS},
        **{f"r2_{d}": per_r2[d] for d in DOMAINS},
    }


def compare_oof_runs(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    """Compare runs against the first run as baseline.

    All runs are scored with the legacy stacked + datazone-aggregated metric
    (see :func:`summarize_oof`).
    """
    if not runs:
        raise ValueError("At least one run is required.")

    rows = []
    baseline_per: dict[str, float] | None = None
    baseline_pooled: float | None = None
    baseline_r2: float | None = None
    for i, (name, path) in enumerate(runs):
        summary = summarize_oof(path)
        per = {d: summary[f"rho_{d}"] for d in DOMAINS}
        pooled = summary["pooled_rho"]
        r2_value = summary["pooled_r2"]
        if i == 0:
            baseline_per = per
            baseline_pooled = pooled
            baseline_r2 = r2_value
        assert baseline_per is not None and baseline_pooled is not None and baseline_r2 is not None
        rows.append({
            "run": name,
            "pooled_rho": pooled,
            "delta_pooled": pooled - baseline_pooled,
            "pooled_r2": r2_value,
            "delta_r2": r2_value - baseline_r2,
            **{f"rho_{d}": per[d] for d in DOMAINS},
            **{f"r2_{d}": summary[f"r2_{d}"] for d in DOMAINS},
            **{f"d_rho_{d}": per[d] - baseline_per[d] for d in DOMAINS},
            "n_rows": summary["n_rows"],
            "oof_path": str(path),
        })
    return pd.DataFrame(rows)


def format_markdown_table(df: pd.DataFrame) -> str:
    """Render the high-signal columns as a compact markdown table."""
    cols = ["run", "pooled_rho", "delta_pooled", "pooled_r2", "delta_r2"] + [
        f"rho_{d}" for d in DOMAINS
    ]
    out = df[cols].copy()
    for c in out.columns:
        if c == "run":
            continue
        if c.startswith("delta"):
            out[c] = out[c].map(lambda v: f"{v:+.4f}")
        else:
            out[c] = out[c].map(lambda v: f"{v:.4f}")
    return out.to_markdown(index=False)
