# -*- coding: utf-8 -*-
"""
Aggregate decision-module OOF predictions to datazone level, then run the
standard evaluate_domain_scores + visualize_domain_scores pipeline.

OOF JSONL schema (from decision/train/cv_runner*.py):
    {
      "datazone": "S01...",
      "fold": 0,
      "prediction_json": {"Income": 5.3, ..., "Housing": 4.1},
      "target_raw":      {"Income": 6.0, ..., "Housing": 5.0}
    }

Multiple rows per datazone are averaged before evaluation.

Usage:
    python scripts/evaluation/eval_decision_oof.py \\
        --oof-jsonl outputs/decision/route_c/sbert_minilm_v0/oof_predictions.jsonl \\
        --output-dir outputs/evaluation/decision_route_c_sbert_minilm_v0 \\
        [--simd-csv dataset/SIMD/SIMD_data.csv] \\
        [--shapefile dataset/glasgow_datazone/glasgow_datazone.shp] \\
        [--no-spatial] [--no-pdf]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

DOMAINS = ["Income", "Employment", "Health", "Education", "Access", "Crime", "Housing"]


def load_and_aggregate(oof_path: Path) -> pd.DataFrame:
    rows = []
    for line in oof_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pred = r["prediction_json"]
        row = {"datazone": r["datazone"]}
        for d in DOMAINS:
            row[d.lower()] = float(pred[d])
        rows.append(row)

    df = pd.DataFrame(rows)
    # group by datazone, average predictions across folds/images
    df = df.groupby("datazone", as_index=False).mean(numeric_only=True)
    df["overall"] = df[[d.lower() for d in DOMAINS]].mean(axis=1)
    col_order = ["datazone"] + [d.lower() for d in DOMAINS] + ["overall"]
    return df[col_order]


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate decision OOF → evaluate + visualize.")
    p.add_argument("--oof-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--simd-csv", type=Path, default=Path("dataset/SIMD/SIMD_data.csv"))
    p.add_argument("--shapefile", type=Path, default=Path("dataset/glasgow_datazone/glasgow_datazone.shp"))
    p.add_argument("--no-spatial", action="store_true")
    p.add_argument("--no-pdf", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. aggregate
    print(f"Loading {args.oof_jsonl} ...")
    df = load_and_aggregate(args.oof_jsonl)
    print(f"  {len(df)} datazones after aggregation")

    pred_csv = args.output_dir / "pred_aggregated.csv"
    df.to_csv(pred_csv, index=False)
    print(f"  Saved → {pred_csv}")

    # 2. evaluate
    report_csv = args.output_dir / "domain_evaluation_report.csv"
    eval_cmd = [
        sys.executable,
        "scripts/evaluation/evaluate_domain_scores.py",
        "--pred-csv", str(pred_csv),
        "--simd-csv", str(args.simd_csv),
        "--shapefile", str(args.shapefile),
        "--output-csv", str(report_csv),
    ]
    if args.no_spatial:
        eval_cmd.append("--no-spatial")
    run(eval_cmd)

    # 3. visualize
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    viz_cmd = [
        sys.executable,
        "scripts/evaluation/visualize_domain_scores.py",
        "--pred-csv", str(pred_csv),
        "--simd-csv", str(args.simd_csv),
        "--shapefile", str(args.shapefile),
        "--output-dir", str(fig_dir),
    ]
    if args.no_pdf:
        viz_cmd.append("--no-pdf")
    run(viz_cmd)

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
