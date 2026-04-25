"""Compare OOF predictions of baseline + SVF ablation runs.

Reads OOF JSONL files directly (independent from eval_decision_oof.py), computes
per-domain pooled Spearman, and emits:
  - a markdown table to stdout
  - a CSV with per-domain rho + deltas vs. baseline

CLI:
    python evaluation/compare_svf_ablation.py \
        --baseline outputs/decision/route_c/modality_sep_v0/oof_predictions.jsonl \
        --runs outputs/decision/route_c/modality_sep_svf_segformer_v0/oof_predictions.jsonl \
               outputs/decision/route_c/modality_sep_svf_mitpsp_v0/oof_predictions.jsonl \
               outputs/decision/route_c/modality_sep_svf_both_v0/oof_predictions.jsonl \
        --output outputs/evaluation/svf_ablation_summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision.data.targets import DOMAINS  # noqa: E402
from src.glasgow_vlm.metrics import spearmanr  # noqa: E402


def _load_oof(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = {"datazone": obj["datazone"], "fold": obj.get("fold", 0)}
            for d in DOMAINS:
                row[f"pred_{d}"] = float(obj["prediction_json"][d])
                row[f"tgt_{d}"] = float(obj["target_raw"][d])
            rows.append(row)
    return pd.DataFrame(rows)


def _pooled_rho(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    per: dict[str, float] = {}
    for d in DOMAINS:
        per[d] = float(spearmanr(df[f"tgt_{d}"].tolist(), df[f"pred_{d}"].tolist()))
    return float(np.mean(list(per.values()))), per


def _label_from_path(p: Path) -> str:
    # Expect `.../<run_name>/oof_predictions.jsonl`
    return p.parent.name


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path,
                        help="OOF JSONL of the baseline run.")
    parser.add_argument("--runs", nargs="+", required=True, type=Path,
                        help="OOF JSONL paths of SVF variant runs.")
    parser.add_argument("--output", required=True, type=Path,
                        help="CSV destination.")
    args = parser.parse_args()

    baseline_df = _load_oof(args.baseline)
    base_pooled, base_per = _pooled_rho(baseline_df)

    rows = [{
        "run": _label_from_path(args.baseline),
        "pooled_rho": base_pooled,
        **{f"rho_{d}": base_per[d] for d in DOMAINS},
        **{f"d_rho_{d}": 0.0 for d in DOMAINS},
        "delta_pooled": 0.0,
    }]

    for oof in args.runs:
        df = _load_oof(oof)
        pooled, per = _pooled_rho(df)
        row = {
            "run": _label_from_path(oof),
            "pooled_rho": pooled,
            **{f"rho_{d}": per[d] for d in DOMAINS},
            **{f"d_rho_{d}": per[d] - base_per[d] for d in DOMAINS},
            "delta_pooled": pooled - base_pooled,
        }
        rows.append(row)

    tbl = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.output, index=False, float_format="%.4f")

    # Markdown digest
    fmt_cols = ["run", "pooled_rho", "delta_pooled"] + [f"rho_{d}" for d in DOMAINS]
    digest = tbl[fmt_cols].copy()
    for c in digest.columns:
        if c != "run":
            digest[c] = digest[c].map(lambda v: f"{v:+.4f}" if "delta" in c else f"{v:.4f}")
    print("\n" + digest.to_markdown(index=False))
    print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    _main()
